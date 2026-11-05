from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import httpx
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.collectors.us_elections import (electoral_units, demo_us_scenario, parse_public_polls,
    parse_county_history, fetch_public_scenario, STATE_INFO, MATCHUPS)
from src.models.us_presidential import forecast_us, allocate_electors, PreElectionConfig


@pytest.fixture
def bundle():
    return demo_us_scenario()


def run(bundle, **kwargs):
    params = dict(polls=bundle['polls'], history=bundle['history'], candidate_a=bundle['candidate_a'],
        candidate_b=bundle['candidate_b'], as_of=bundle['fetched_at'], source=bundle['source'],
        config=PreElectionConfig(draws=1000))
    params.update(kwargs)
    return forecast_us(**params)


def test_allocation_538_and_district_rules():
    units = electoral_units()
    assert len(units) == 56 and units.electoral_votes.sum() == 538
    assert units.loc[units.state == 'ME','electoral_votes'].sum() == 4
    assert units.loc[units.state == 'NE','electoral_votes'].sum() == 5
    shares = np.zeros((2,len(units)))
    shares[0,units.unit == 'ME'] = 1
    shares[1,units.unit == 'ME-2'] = 1
    a,b = allocate_electors(shares)
    assert list(a) == [2,1]
    assert (a+b == 538).all()
    a,b = allocate_electors(np.full((1,len(units)),.5))
    assert a[0] == 0 and b[0] == 0  # no invented tie-breaking winner


def test_electoral_majority_is_not_count_of_states():
    units = electoral_units()
    cheapest = units.sort_values('electoral_votes').index[:30]
    shares = np.zeros((1,len(units)))
    shares[:,cheapest] = 1
    a,b = allocate_electors(shares)
    assert (shares[0] > .5).sum() > len(units)/2
    assert a[0] < 270 and b[0] >= 270


def test_pre_election_determinism_and_probabilities(bundle):
    a,b = run(bundle),run(bundle)
    assert a == b
    assert sum(a['ev_histogram_a']) == 1000
    assert a['probability_a_270']+a['probability_b_270']+a['probability_no_majority'] == pytest.approx(1)
    assert 0 <= a['electoral_votes_a_interval_95'][0] <= a['electoral_votes_a_mean'] <= a['electoral_votes_a_interval_95'][1] <= 538
    assert a['conditional_on_matchup'] and not a['diagnostics']['calibrated']
    assert a['election_day'].startswith('2028-11-07')


def test_future_publication_cannot_affect_forecast(bundle):
    baseline = run(bundle)
    future = bundle['polls'].iloc[:1].copy()
    future['poll_id'] = 'future-leak'
    future['available_at'] = '2027-01-01T00:00:00+00:00'
    future['share_a'],future['share_b'] = .95,.04
    changed = {**bundle,'polls':pd.concat([bundle['polls'],future],ignore_index=True)}
    result = run(changed)
    assert result['probability_a_270'] == baseline['probability_a_270']
    assert result['ev_histogram_a'] == baseline['ev_histogram_a']
    assert result['excluded_poll_count'] == 1
    assert all(p['poll_id'] != 'future-leak' for p in result['polls_used'])


def test_future_history_is_not_used(bundle):
    history = bundle['history'].copy()
    history.loc[history.unit == 'PA','available_at'] = '2027-01-01T00:00:00+00:00'
    with pytest.raises(ValueError,match='50 estados'):
        run(bundle,history=history)


def test_missing_districts_use_uncertain_prior_not_statewide_electors(bundle):
    history = bundle['history'][~bundle['history'].unit.str.contains('-')].copy()
    result = run(bundle,history=history)
    assert len(result['district_prior_units']) == 5
    rows = pd.DataFrame(result['units'])
    assert rows.loc[rows.unit == 'ME','electoral_votes'].item() == 2
    assert rows.loc[rows.unit == 'ME-1','historical_basis'].item() == 'statewide_prior_with_extra_uncertainty'


@pytest.mark.parametrize('column,value', [('share_a',1.1),('share_b',np.nan),('sample_size',-1),('unit','INVALID')])
def test_bad_poll_inputs_rejected(bundle,column,value):
    polls = bundle['polls'].copy()
    polls.loc[0,column] = value
    with pytest.raises(ValueError):
        run(bundle,polls=polls)


def test_cycle_and_candidate_filters(bundle):
    polls = bundle['polls'].copy()
    polls['election_year'] = 2024
    with pytest.raises(ValueError,match='Sem pesquisas'):
        run(bundle,polls=polls)
    with pytest.raises(ValueError,match='Sem pesquisas'):
        run(bundle,candidate_a='Different candidate')
    with pytest.raises(ValueError,match='pré-eleitoral'):
        run(bundle,as_of='2028-11-08T12:00:00+00:00')
    with pytest.raises(ValueError,match='somente'):
        run(bundle,election_year=2032)


def test_duplicate_poll_not_counted_as_independent(bundle):
    polls = pd.concat([bundle['polls'],bundle['polls'].iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError,match='duplicad'):
        run(bundle,polls=polls)


def test_poll_evidence_changes_outcomes():
    b = demo_us_scenario(datetime(2028,11,5,tzinfo=timezone.utc))
    polls = b['polls'].copy()
    polls['unit'] = 'US'
    polls['share_a'],polls['share_b'] = .85,.10
    polls['sample_size'] = 10000
    config = PreElectionConfig(draws=1000,common_election_error_sd=.01,future_drift_sd_per_sqrt_year=.01)
    strong_a = run(b,polls=polls,config=config)
    polls['share_a'],polls['share_b'] = .10,.85
    strong_b = run(b,polls=polls,config=config)
    assert strong_a['probability_a_270'] > .95
    assert strong_b['probability_a_270'] < .05


def test_horizon_uncertainty_shared_national_component(bundle):
    far = run(bundle,config=PreElectionConfig(draws=500,max_poll_age_days=1460))
    near = run(bundle,as_of='2028-11-01T12:00:00+00:00',config=PreElectionConfig(draws=500,max_poll_age_days=1460))
    assert far['diagnostics']['national_future_sd_logit'] > near['diagnostics']['national_future_sd_logit']
    assert far['diagnostics']['joint_draws'] == 500


def public_payload():
    return {'race_id':'pres-2028-vance-newsom','as_of':'2026-11-05','polls':[
        {'pollster':'Test pollster','sample':'1,417 A','end_date':'2026-10-14',
         'vals':{'JD Vance':36,'Gavin Newsom':41},'source_url':'https://pollster.example/report'}]}


def test_public_parser_checks_matchup_sample_and_uses_raw_polls():
    data = public_payload()
    polls,warnings = parse_public_polls(data,'pres-2028-vance-newsom','2026-11-05T12:00:00+00:00')
    assert polls.sample_size.item() == 1417
    assert polls.share_a.item() == .36
    assert polls.unit.item() == 'US'
    assert polls.available_at.item() == '2026-11-05T12:00:00+00:00'
    assert warnings == []
    with pytest.raises(ValueError,match='confronto'):
        parse_public_polls(data,'pres-2028-vance-buttigieg','2026-11-05T12:00:00+00:00')
    data['polls'][0]['sample'] = 'unknown'
    with pytest.raises(ValueError,match='válidas'):
        parse_public_polls(data,'pres-2028-vance-newsom','2026-11-05T12:00:00+00:00')


def county_csv():
    return pd.DataFrame([{'state_name':name,'county_fips':str(i).zfill(5),'votes_gop':100+i,'votes_dem':100}
                         for i,(_,name,_,_) in enumerate(STATE_INFO)]).to_csv(index=False).encode()


def test_historical_county_aggregation_and_validation():
    df = parse_county_history(county_csv(),'2026-11-05T12:00:00+00:00')
    assert len(df) == 51
    assert df.loc[df.unit == 'AL','share_a'].item() == .5
    partial = pd.read_csv(__import__('io').BytesIO(county_csv())).iloc[:-1].to_csv(index=False).encode()
    with pytest.raises(ValueError,match='Histórico sem'):
        parse_county_history(partial,'2026-11-05T12:00:00+00:00')


def test_public_connector_no_keys_and_no_demo_fallback():
    from src.collectors.us_elections import HISTORY_URL
    def handle(request):
        if str(request.url) == HISTORY_URL:
            return httpx.Response(200,content=county_csv())
        return httpx.Response(200,json=public_payload())
    b = fetch_public_scenario('pres-2028-vance-newsom',transport=httpx.MockTransport(handle))
    assert b['source'] == 'PUBLIC_DATA' and len(b['history']) == 51
    assert len(b['history_sha256']) == 64 and len(b['poll_sha256']) == 64
    with pytest.raises(httpx.HTTPStatusError):
        fetch_public_scenario('pres-2028-vance-newsom',transport=httpx.MockTransport(lambda r:httpx.Response(404)))


def test_new_page_demo_and_existing_model_remain_separate(tmp_path,monkeypatch):
    monkeypatch.setenv('SCANNER_DATABASE',str(tmp_path/'us-ui.sqlite3'))
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=40).run()
    app.sidebar.radio[0].set_value('EUA — Pré-eleição').run()
    assert not app.exception
    app.radio(key='us_source').set_value('DEMO simulado').run()
    app.button(key='run_us_forecast').click().run()
    assert not app.exception and not app.error
    result = app.session_state['us_forecast_result']
    assert result['source'] == 'DEMO' and result['conditional_on_matchup']
    app.sidebar.radio[0].set_value('Election Forecast').run()
    assert not app.exception
    app.button[1].click().run()
    assert app.session_state['progress'] == .1
    app.sidebar.radio[0].set_value('EUA — Pré-eleição').run()
    assert not app.exception
    assert app.session_state['us_forecast_result'] == result


def test_new_page_public_failure_does_not_show_demo_as_real(tmp_path,monkeypatch):
    import src.us_forecast_ui as ui
    monkeypatch.setenv('SCANNER_DATABASE',str(tmp_path/'failed-us-ui.sqlite3'))
    def fail(race):
        raise ValueError('Fonte indisponível')
    monkeypatch.setattr(ui,'public_inputs',fail)
    app = AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=40).run()
    app.sidebar.radio[0].set_value('EUA — Pré-eleição').run()
    app.button(key='run_us_forecast').click().run()
    assert not app.exception and app.error
    assert 'us_forecast_result' not in app.session_state


def test_district_history_cannot_silently_mix_cycles(bundle):
    history = bundle['history'].copy()
    history.loc[history.unit == 'ME-1','year'] = 2020
    with pytest.raises(ValueError,match='distrital'):
        run(bundle,history=history)


def test_blank_pollster_is_not_valid_evidence(bundle):
    polls = bundle['polls'].copy()
    polls.loc[0,'pollster'] = ''
    with pytest.raises(ValueError,match='vazios'):
        run(bundle,polls=polls)
