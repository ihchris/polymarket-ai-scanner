"""Append-only UTC observation log, each domain has its own table."""
import json
import sqlite3
from pathlib import Path
from src.utils.validation import utc, utcnow

TABLES = {'markets', 'price_snapshots', 'election_partials', 'forecasts', 'opportunities', 'backtests'}


class Database:
    def __init__(self, path='data/scanner.sqlite3'):
        if str(path) != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        try:
            self.connection.execute('PRAGMA busy_timeout=30000')
            # Rollback journals work on hosted filesystems that cannot support
            # WAL shared-memory/locking. Never require WAL just to open the app.
            self.connection.execute('PRAGMA journal_mode=DELETE')
            for table in sorted(TABLES):
                self.connection.execute(f'CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, model_version TEXT, payload TEXT NOT NULL)')
            self.connection.commit()
        except Exception:
            self.connection.close()
            raise

    def append(self, table, payload, timestamp=None, model_version=None):
        if table not in TABLES:
            raise ValueError('Unknown table')
        timestamp = utc(timestamp or utcnow()).isoformat()
        with self.connection:
            self.connection.execute(f'INSERT INTO {table} (timestamp,model_version,payload) VALUES (?,?,?)',
                (timestamp, model_version, json.dumps(payload, default=str, allow_nan=False)))

    def read(self, table):
        if table not in TABLES:
            raise ValueError('Unknown table')
        return self.connection.execute(f'SELECT timestamp,model_version,payload FROM {table} ORDER BY id').fetchall()

    def close(self):
        self.connection.close()
