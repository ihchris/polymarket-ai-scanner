"""Point-in-time CSV input; no forward filling from future rows."""
import pandas as pd
from src.utils.validation import utc

REQUIRED = {'municipality', 'state', 'previous_share', 'expected_votes', 'electorate',
            'urban', 'counted_votes', 'candidate_a_votes', 'complete', 'timestamp'}


def load_history(path):
    df = pd.read_csv(path, dtype={'municipality': str, 'state': str})
    if not REQUIRED.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(REQUIRED - set(df.columns))}')
    df['timestamp'] = pd.to_datetime(df.timestamp.map(utc), utc=True)
    if df.duplicated(['municipality', 'timestamp']).any():
        raise ValueError('Duplicate municipality timestamps')
    if not df.complete.isin([True, False, 0, 1]).all():
        raise ValueError('complete must be a boolean')
    return df.sort_values('timestamp')


def as_of(history, timestamp):
    cutoff = pd.Timestamp(utc(timestamp))
    eligible = history[ pd.to_datetime(history.timestamp, utc=True) <= cutoff ]
    return eligible.sort_values('timestamp').drop_duplicates('municipality', keep='last').copy()
