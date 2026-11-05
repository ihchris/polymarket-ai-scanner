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


def test_dashboard_falls_back_when_configured_database_is_unwritable(tmp_path, monkeypatch):
    import sqlite3
    # A file in place of the parent directory fails even when tests run as root.
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('not a directory')
    monkeypatch.setenv('SCANNER_DATABASE', str(blocked / 'scanner.sqlite3'))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    assert any('temporário' in w.value for w in app.warning)
    fallback = app.session_state['storage_path']
    connection = sqlite3.connect(fallback)
    assert connection.execute('SELECT COUNT(*) FROM markets').fetchone()[0] > 0
    connection.close()
    app.sidebar.radio[0].set_value('Election Forecast').run()
    assert not app.exception
    assert app.session_state['storage_path'] == fallback
    app.sidebar.radio[0].set_value('Backtesting').run()
    app.button[0].click().run()
    assert not app.exception
    connection = sqlite3.connect(fallback)
    assert connection.execute('SELECT COUNT(*) FROM backtests').fetchone()[0] == 1
    connection.close()


def test_dashboard_remains_available_when_all_storage_fails(monkeypatch):
    import sqlite3
    import src.storage.database as storage
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('unable to open database file')
    monkeypatch.setattr(storage, 'Database', fail)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    assert app.session_state['storage_disabled']
    assert any('não serão salvos' in w.value for w in app.warning)
    app.sidebar.radio[0].set_value('Election Forecast').run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Backtesting').run()
    app.button[0].click().run()
    assert not app.exception
    assert 'metrics' in app.session_state['backtest']
