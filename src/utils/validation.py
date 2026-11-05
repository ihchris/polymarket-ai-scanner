from datetime import datetime, timezone
import numpy as np


def utcnow():
    return datetime.now(timezone.utc)


def utc(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00')) if not isinstance(value, datetime) else value
    if dt.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return dt.astimezone(timezone.utc)


def probability(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Probability/price must be finite and in [0, 1]')
    return value


def normalize_probabilities(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise ValueError('Invalid probability weights')
    return a / a.sum()
