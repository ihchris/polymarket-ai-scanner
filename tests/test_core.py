import json
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
import httpx
from pydantic import ValidationError
from config.settings import Settings
from src.collectors.polymarket import parse_market, parse_book, PolymarketClient, Book, Level, demo_markets
from src.collectors.elections import DemoElection, BASE_TIME
from src.collectors.historical import as_of, load_history
from src.models.bayesian import forecast
from src.models.hierarchical import sample_posterior
from src.analysis.probability import ProbabilityEstimate
from src.analysis.opportunities import evaluate
from src.analysis.backtest import replay
from src.storage.database import Database
from src.utils.validation import normalize_probabilities, probability


@pytest.fixture
def raw_market():
    return {'id': '1', 'question': 'A wins?', 'outcomes': '["Yes", "No"]',
            'clobTokenIds': '["123", "456"]', 'outcomePrices': '["0.4", "0.6"]',
            'volume': '1000', 'liquidity': '500', 'active': True, 'closed': False,
            'acceptingOrders': True, 'updatedAt': BASE_TIME.isoformat(),
            'endDate': (BASE_TIME+timedelta(days=1)).isoformat()}


def test_parse(raw_market):
    m = parse_market(raw_market)
    assert m.token_ids == ['123', '456']
    assert m.indicative_prices == [.4, .6]
    assert not m.books


@pytest.mark.parametrize('price', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_prices(price):
    with pytest.raises((ValueError, ValidationError)):
        Level(price=price, size=10)
    with pytest.raises(ValueError):
        probability(price)


def test_missing_and_mapping(raw_market):
    del raw_market['updatedAt']
    with pytest.raises(KeyError):
        parse_market(raw_market)
    raw_market['updatedAt'] = BASE_TIME.isoformat()
    raw_market['clobTokenIds'] = '["123"]'
    with pytest.raises(ValueError):
        parse_market(raw_market)


def test_book_depth_and_asset():
    raw = {'asset_id': '123', 'timestamp': str(int(BASE_TIME.timestamp()*1000)),
           'bids': [{'price': '.4', 'size': '50'}],
           'asks': [{'price': '.6', 'size': '10'}, {'price': '.5', 'size': '5'}]}
    book = parse_book(raw, '123')
    assert book.bid == .4 and book.ask == .5
    assert book.buy_quote(10) == pytest.approx(.55)
    assert book.buy_quote(20) is None
    with pytest.raises(ValueError):
        parse_book(raw, 'wrong')


def test_retry_and_timeout():
    calls = []
    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={'Retry-After': '0'})
        if len(calls) == 2:
            raise httpx.ReadTimeout('timeout', request=request)
        return httpx.Response(200, json=[])
    c = PolymarketClient(transport=httpx.MockTransport(handle), sleep=lambda _: None)
    assert c.markets() == [] and len(calls) == 3
    c.close()


def test_no_retry_for_404():
    c = PolymarketClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)), sleep=lambda _: None)
    with pytest.raises(httpx.HTTPStatusError):
        c.get('https://example.test', '/missing')
    c.close()


def test_live_client_books(raw_market, monkeypatch):
    # Make the fixture active under its own virtual clock.
    monkeypatch.setattr('src.collectors.polymarket.utcnow', lambda: BASE_TIME)
    def handle(request):
        if request.url.path == '/markets':
            return httpx.Response(200, json=[raw_market])
        if request.url.path == '/fee-rate':
            return httpx.Response(200, json={'base_fee': 0})
        token = request.url.params['token_id']
        return httpx.Response(200, json={'asset_id': token, 'timestamp': str(int(BASE_TIME.timestamp()*1000)),
            'asks': [{'price': '.55', 'size': '100'}], 'bids': []})
    c = PolymarketClient(transport=httpx.MockTransport(handle), sleep=lambda _: None)
    markets = c.markets(max_markets=1)
    assert markets[0].books['YES'].ask == .55
    assert markets[0].books['NO'].token_id == '456'
    assert markets[0].books['YES'].fee_metadata == {'base_fee': 0}
    c.close()


def estimate(p=.8, low=.75):
    return ProbabilityEstimate(probability=p, lower=low, upper=.9, timestamp=BASE_TIME, model_version='test')


def test_ev_and_depth():
    market = demo_markets(0, BASE_TIME)[0]
    s = Settings(estimated_fee_per_share=.01, slippage_per_share=.005)
    r = evaluate(market, 'YES', estimate(), s, now=BASE_TIME)
    assert r['gross_edge'] == pytest.approx(.8-.505)
    assert r['net_edge'] == pytest.approx(.8-.505-.015)
    assert r['status'] == 'POSSIBLE_EDGE'
    s.order_size = 105
    r = evaluate(market, 'YES', estimate(), s, now=BASE_TIME)
    assert r['execution_price'] > r['ask_price']
    assert r['net_edge'] == pytest.approx(.8-r['execution_price']-.015)


@pytest.mark.parametrize('case, status', [('missing', 'INSUFFICIENT_DATA'), ('fees', 'INSUFFICIENT_DATA'),
    ('closed', 'INSUFFICIENT_DATA'), ('liquidity', 'INSUFFICIENT_DATA'), ('depth', 'INSUFFICIENT_DATA'),
    ('stale', 'STALE'), ('uncertain', 'UNCERTAIN'), ('noedge', 'NO_EDGE'), ('ambiguous', 'UNCERTAIN')])
def test_statuses(case, status):
    m = demo_markets(0, BASE_TIME)[0]
    s = Settings(estimated_fee_per_share=0)
    p, now = estimate(), BASE_TIME
    if case == 'missing': p = None
    if case == 'fees': s.estimated_fee_per_share = None
    if case == 'closed': m.closed = True
    if case == 'liquidity': m.liquidity = 0
    if case == 'depth': s.order_size = 10000
    if case == 'stale': now += timedelta(seconds=301)
    if case == 'uncertain': p = estimate(low=.1)
    if case == 'noedge': p = estimate(p=.2, low=.1)
    if case == 'ambiguous': m.ambiguous_resolution = True
    assert evaluate(m, 'YES', p, s, now=now)['status'] == status


def test_normalization():
    assert normalize_probabilities([2, 3]).sum() == pytest.approx(1)
    for values in [[], [0, 0], [-1, 2], [np.nan, 1]]:
        with pytest.raises(ValueError): normalize_probabilities(values)


def test_deterministic_hidden_and_nonrandom():
    demo = DemoElection()
    snapshot = demo.snapshot(.5)
    pd.testing.assert_frame_equal(snapshot, DemoElection().snapshot(.5))
    assert not any('final' in col for col in snapshot)
    assert snapshot.urban.corr(snapshot.counted_votes / snapshot.expected_votes) < -.3
    later = demo.snapshot(.6)
    assert (later.counted_votes >= snapshot.counted_votes).all()
    assert (later.candidate_a_votes >= snapshot.candidate_a_votes).all()


def test_forecast_determinism_and_completion():
    demo = DemoElection()
    df = demo.snapshot(.4)
    a, b = forecast(df, 500), forecast(df, 500)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert 0 <= a.win_probability <= 1
    assert a.credible_interval_95[0] < a.estimated_vote_share < a.credible_interval_95[1]
    final = forecast(demo.snapshot(1), 500)
    assert final.estimated_vote_share == pytest.approx(demo.outcome()['vote_share'])
    assert final.credible_interval_95[0] == pytest.approx(final.credible_interval_95[1])
    assert final.counted_votes_percentage == 100


def test_bayesian_evidence_and_pooled_covariance():
    demo = DemoElection()
    prior = forecast(demo.snapshot(0), 1000)
    later = forecast(demo.snapshot(.8), 1000)
    assert np.std(later.samples) < np.std(prior.samples)
    df = demo.snapshot(0)
    posterior = sample_posterior(df, draws=2000)
    assert np.corrcoef(posterior.logits[:, 0], posterior.logits[:, 3])[0, 1] > .3


@pytest.mark.parametrize('column, value', [('counted_votes', -1), ('previous_share', 1), ('expected_votes', np.nan), ('candidate_a_votes', 1)])
def test_bad_election(column, value):
    df = DemoElection().snapshot(0)
    df.loc[0, column] = value
    with pytest.raises(ValueError): forecast(df, 10)


def history():
    demo = DemoElection()
    return pd.concat([demo.snapshot(p) for p in [0, .3, .6, 1]], ignore_index=True)


def test_asof_and_csv():
    h = history()
    loaded = load_history(__import__('io').StringIO(h.to_csv(index=False)))
    assert len(loaded) == 400
    cut = BASE_TIME + timedelta(minutes=54)
    snap = as_of(h, cut)
    assert len(snap) == 100
    assert pd.to_datetime(snap.timestamp, utc=True).max() <= cut


def test_replay_no_future_leakage_and_label_isolation():
    h = history()
    times = sorted(h.timestamp.unique())[:2]
    r = replay(h, times, .52, draws=500)
    changed = h.copy()
    future = pd.to_datetime(changed.timestamp, utc=True) > pd.Timestamp(times[-1])
    changed.loc[future, 'candidate_a_votes'] = 0
    other = replay(changed, times, .48, draws=500)
    assert r['forecasts'] == other['forecasts']
    assert r['metrics']['net_return'] is None
    p = np.array([x['probability'] for x in r['forecasts']])
    assert r['metrics']['brier_score'] == pytest.approx(np.mean((p-1)**2))
    assert r['metrics']['log_loss'] == pytest.approx(-np.log(np.clip(p, 1e-9, 1-1e-9)).mean())


def test_future_prices_cannot_fill():
    h = history()
    times = sorted(h.timestamp.unique())[:2]
    prices = pd.DataFrame([{'timestamp': BASE_TIME+timedelta(days=1), 'ask': .01, 'depth': 100, 'source': 'DEMO'}])
    result = replay(h, times, .52, draws=500, price_history=prices, fee_per_share=0)
    assert result['trades'] == []


def test_realized_trade_arithmetic():
    h = history()
    # Construct strong observed A evidence before completion. The simulator's
    # default hidden outcome happens to favor B; never assume it favors A.
    h['candidate_a_votes'] = np.floor(h.counted_votes * .65).astype(int)
    times = sorted(h.timestamp.unique())[-2:-1]
    prices = pd.DataFrame([{'timestamp': times[0], 'ask': .1, 'depth': 100, 'source': 'DEMO'}])
    result = replay(h, times, .65, draws=500, price_history=prices, fee_per_share=.01, slippage=.002)
    assert len(result['trades']) == 1
    assert result['trades'][0]['profit'] == pytest.approx(10*(1-.1-.01-.002))


def test_database_append_only_and_utc(tmp_path):
    db = Database(tmp_path/'test.sqlite3')
    for _ in range(2): db.append('forecasts', {'p': .5}, BASE_TIME, 'v1')
    rows = db.read('forecasts')
    assert len(rows) == 2 and rows[0][0].endswith('+00:00')
    with pytest.raises(ValueError): db.append('bad; DROP TABLE forecasts', {})
    with pytest.raises(ValueError): db.append('forecasts', {}, '2026-01-01T00:00:00')
    db.close()


def test_no_trade_after_result_or_beyond_capital():
    h = history()
    h['candidate_a_votes'] = np.floor(h.counted_votes * .65).astype(int)
    times = sorted(h.timestamp.unique())
    prices = pd.DataFrame([{'timestamp': t, 'ask': .1, 'depth': 100000, 'source': 'DEMO'} for t in times])
    complete = replay(h, times[-1:], .65, draws=500, price_history=prices, fee_per_share=0)
    assert complete['trades'] == []
    unfunded = replay(h, times[-2:-1], .65, draws=500, price_history=prices,
                      fee_per_share=0, quantity=20000, initial_capital=1000)
    assert unfunded['trades'] == []
