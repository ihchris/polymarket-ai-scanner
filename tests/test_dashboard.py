from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_dashboard_all_pages_offline(tmp_path, monkeypatch):
    import httpx
    def offline(*args, **kwargs):
        raise httpx.ConnectError('offline test')
    monkeypatch.setattr(httpx.Client, 'get', offline)
    monkeypatch.setenv('SCANNER_DATABASE', str(tmp_path / 'dashboard.sqlite3'))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    assert 'Market Scanner' in [x.value for x in app.subheader]
    app.sidebar.radio[0].set_value('Election Forecast').run()
    assert not app.exception
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state['progress'] == .1
    app.sidebar.radio[0].set_value('Backtesting').run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.session_state['backtest']['forecasts']) == 6
    app.sidebar.radio[0].set_value('Settings').run()
    assert not app.exception
    app.selectbox[0].set_value('LIVE_DATA')
    app.button[0].click().run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Market Scanner').run()
    assert not app.exception
    assert any('indisponível' in warning.value for warning in app.warning)
