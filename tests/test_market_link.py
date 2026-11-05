from datetime import timedelta
import io
import json
import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path
from src.collectors.market_link import parse_link, lookup_link, inspect_market
from src.collectors.polymarket import PolymarketClient, parse_market
from src.collectors.elections import DemoElection, BASE_TIME
from src.analysis.linked_election import analyze_election_csv
from config.settings import Settings

CONDITION = '0x' + 'a'*64


def raw_market(closed=False):
    return {'id': '101', 'question': 'Candidate A wins?', 'slug': 'candidate-a-wins',
            'conditionId': CONDITION, 'outcomes': '["Yes","No"]',
            'clobTokenIds': '["123","456"]', 'outcomePrices': '["1","0"]',
            'volume': 1000, 'liquidity': 500, 'closed': closed, 'active': True,
            'acceptingOrders': not closed, 'updatedAt': BASE_TIME.isoformat(),
            'endDate': (BASE_TIME+timedelta(days=1)).isoformat(),
            'description': 'Majority of valid votes for A', 'resolutionSource': 'https://official.example/election'}


def clob_market(closed=False, winner=False):
    return {'closed': closed, 'tokens': [{'token_id': '123', 'winner': winner}, {'token_id': '456', 'winner': False}]}


def client_for(raw=None, clob=None):
    raw = raw or raw_market()
    clob = clob or clob_market()
    def handle(request):
        path = request.url.path
        if path.startswith('/events/slug/'):
            return httpx.Response(200, json={'markets': [raw, {**raw, 'id': '102', 'slug': 'other-market'}]})
        if path.startswith('/markets/slug/'):
            return httpx.Response(200, json=raw)
        if path == f'/markets/{CONDITION}':
            return httpx.Response(200, json=clob)
        if path == '/book':
            return httpx.Response(200, json={'asset_id': request.url.params['token_id'],
                'timestamp': str(int(BASE_TIME.timestamp()*1000)),
                'bids': [{'price': '.4', 'size': '10'}], 'asks': [{'price': '.5', 'size': '20'}]})
        return httpx.Response(404)
    return PolymarketClient(transport=httpx.MockTransport(handle), retries=0)


@pytest.mark.parametrize('url', ['http://polymarket.com/event/x', 'https://evil.example/event/x',
    'https://polymarket.com.evil.example/event/x', 'https://user@polymarket.com/event/x',
    'https://polymarket.com:8080/event/x', 'https://polymarket.com/event/..',
    'https://polymarket.com/event/%2Fetc', 'https://polymarket.com/event/x/y/z',
    'https://polymarket.com/', 'https://polymarket.com/event/'])
def test_reject_unsafe_links(url):
    with pytest.raises(ValueError):
        parse_link(url)


def test_links_and_specific_event_selection():
    assert parse_link('https://www.polymarket.com/event/election/candidate-a-wins?tid=1').market_slug == 'candidate-a-wins'
    client = client_for()
    event = lookup_link(client, 'https://polymarket.com/event/election')
    assert len(event['markets']) == 2
    selected = lookup_link(client, 'https://polymarket.com/event/election/candidate-a-wins')
    assert len(selected['markets']) == 1 and selected['markets'][0].id == '101'
    direct = lookup_link(client, 'https://polymarket.com/market/candidate-a-wins')
    assert direct['markets'][0].source == 'LIVE_DATA'
    with pytest.raises(ValueError):
        lookup_link(client, 'https://polymarket.com/event/election/missing')
    client.close()


def test_price_one_is_not_a_result():
    client = client_for()
    market = parse_market(raw_market())
    report = inspect_market(client, market, now=BASE_TIME)
    assert market.indicative_prices == [1., 0.]
    assert report['status'] == 'UNRESOLVED'
    assert report['winner'] is None and report['independent_prediction'] is None
    assert len(report['quotes']) == 2
    client.close()


def test_resolved_requires_explicit_winner():
    client = client_for(raw_market(True), clob_market(True, True))
    report = inspect_market(client, parse_market(raw_market(True)), BASE_TIME)
    assert report['status'] == 'RESOLVED' and report['winner'] == 'Yes'
    assert report['quotes'] == []
    client.close()


@pytest.mark.parametrize('kind', ['no_winner', 'multiple_winners', 'pending', 'wrong_token', 'closure_disagreement', 'winner_on_open'])
def test_do_not_confirm_ambiguous_resolution(kind):
    raw, clob = raw_market(True), clob_market(True, True)
    if kind == 'no_winner': clob['tokens'][0]['winner'] = False
    if kind == 'multiple_winners': clob['tokens'][1]['winner'] = True
    if kind == 'pending': raw['umaResolutionStatus'] = 'disputed'
    if kind == 'wrong_token': clob['tokens'][0]['token_id'] = '999'
    if kind == 'closure_disagreement': clob['closed'] = False
    if kind == 'winner_on_open': raw['closed'], clob['closed'] = False, False
    client = client_for(raw, clob)
    report = inspect_market(client, parse_market(raw), BASE_TIME)
    assert report['winner'] is None
    assert report['status'] in {'AWAITING_RESOLUTION', 'INCONSISTENT'}
    client.close()


def test_lookup_failure_never_substitutes_demo():
    def offline(request):
        raise httpx.ConnectError('offline')
    client = PolymarketClient(transport=httpx.MockTransport(offline), retries=0)
    with pytest.raises(httpx.ConnectError):
        lookup_link(client, 'https://polymarket.com/event/election')
    client.close()


def test_real_csv_binding_and_no_future_leakage():
    demo = DemoElection()
    # Synthetic fixture for testing the user-upload path, not real evidence.
    history = pd.concat([demo.snapshot(0), demo.snapshot(.5), demo.snapshot(1)])
    content = history.to_csv(index=False).encode()
    client = client_for()
    market = parse_market(raw_market())
    inspect_market(client, market, BASE_TIME)
    now = BASE_TIME + timedelta(minutes=90)
    settings = Settings(simulations=500, estimated_fee_per_share=0, maximum_age_seconds=7200)
    args = dict(market=market, csv_bytes=content, cutoff=now, candidate_a='Alice', candidate_b='Bob',
                candidate_a_outcome='NO', source_url='https://official.example/results',
                confirmed=True, settings=settings, now=now)
    result = analyze_election_csv(**args)
    assert result['forecast']['candidate'] == 'Alice'
    assert result['candidate_a_outcome'] == 'NO'
    assert result['source'] == 'CSV_USUARIO_NAO_VERIFICADO_AUTOMATICAMENTE'
    assert len(result['source_sha256']) == 64
    changed = history.copy()
    changed.loc[pd.to_datetime(changed.timestamp, utc=True) > now, 'candidate_a_votes'] = 0
    result2 = analyze_election_csv(**{**args, 'csv_bytes': changed.to_csv(index=False).encode()})
    assert result['forecast'] == result2['forecast']
    no = next(r for r in result['opportunities'] if r['side'] == 'NO')
    assert no['model_probability'] == result['forecast']['win_probability']
    with pytest.raises(ValueError):
        analyze_election_csv(**{**args, 'confirmed': False})
    with pytest.raises(ValueError):
        analyze_election_csv(**{**args, 'cutoff': now+timedelta(days=1)})
    client.close()


def test_link_panel_real_lookup_offline_transport(tmp_path, monkeypatch):
    import src.linked_market_ui as ui
    monkeypatch.setenv('SCANNER_DATABASE', str(tmp_path/'ui.sqlite3'))
    monkeypatch.setattr(ui, 'PolymarketClient', lambda **kwargs: client_for(raw_market(True), clob_market(True, True)))
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py', default_timeout=30).run()
    assert not app.exception
    app.text_input[0].set_value('https://polymarket.com/market/candidate-a-wins')
    app.button[0].click().run()
    assert not app.exception
    assert any('Resultado resolvido na Polymarket: Yes' in s.value for s in app.success)
    assert app.session_state['linked_report']['source'] == 'LIVE_DATA'
    assert not app.session_state['linked_report']['independent_prediction']
    app.text_input[0].set_value('https://evil.example/event/x')
    app.button[0].click().run()
    assert not app.exception
    assert app.error
    assert 'linked_report' not in app.session_state


def test_refresh_uses_market_id_not_slug():
    from src.collectors.market_link import refresh_selected_market
    calls = []
    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=raw_market())
    client = PolymarketClient(transport=httpx.MockTransport(handle), retries=0)
    market = refresh_selected_market(client, '101', 'https://polymarket.com/event/election')
    assert market.id == '101'
    assert calls == ['/markets/101']
    client.close()


def test_refresh_falls_back_to_fresh_event_when_individual_route_fails():
    from src.collectors.market_link import refresh_selected_market
    calls = []
    def handle(request):
        calls.append(request.url.path)
        if request.url.path == '/markets/102':
            return httpx.Response(503)
        return httpx.Response(200, json={'markets': [raw_market(), {**raw_market(), 'id': '102', 'slug': 'changed-slug'}]})
    client = PolymarketClient(transport=httpx.MockTransport(handle), retries=0)
    market = refresh_selected_market(client, '102', 'https://polymarket.com/event/presidential-election-winner-2028')
    assert market.id == '102' and market.slug == 'changed-slug'
    assert calls == ['/markets/102', '/events/slug/presidential-election-winner-2028']
    client.close()


def test_failed_refresh_never_returns_stale_metadata():
    from src.collectors.market_link import refresh_selected_market
    client = PolymarketClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)), retries=0)
    with pytest.raises(httpx.HTTPStatusError):
        refresh_selected_market(client, '101', 'https://polymarket.com/event/election')
    client.close()


@pytest.mark.parametrize('code', [403, 404, 429, 503])
def test_lookup_failure_displays_safe_actionable_status(code):
    from src.collectors.market_link import lookup_error_message
    response = httpx.Response(code, request=httpx.Request('GET', 'https://gamma-api.polymarket.com/markets/101'), text='PRIVATE_BODY')
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        message = lookup_error_message(exc)
    assert str(code) in message
    assert 'PRIVATE_BODY' not in message


def test_multi_market_selection_can_be_changed_and_refreshed(tmp_path, monkeypatch):
    import src.linked_market_ui as ui
    monkeypatch.setenv('SCANNER_DATABASE', str(tmp_path/'multi-ui.sqlite3'))
    monkeypatch.setattr(ui, 'PolymarketClient', lambda **kwargs: client_for())
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py', default_timeout=30).run()
    app.text_input[0].set_value('https://polymarket.com/event/presidential-election-winner-2028')
    app.button[0].click().run()
    assert not app.exception and not app.error
    app.selectbox(key='linked_market_id').set_value('102').run()
    app.button(key='verify_link').click().run()
    assert not app.exception and not app.error
    assert app.session_state['linked_report']['market']['id'] == '102'
    app.selectbox(key='linked_market_id').set_value('101').run()
    app.button(key='verify_link').click().run()
    assert not app.exception and not app.error
    assert app.session_state['linked_report']['market']['id'] == '101'
