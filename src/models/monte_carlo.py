"""One row is one JOINT posterior scenario, shared across municipalities."""
import numpy as np
from scipy.special import expit


def simulate_remaining(df, posterior, seed=43, selection_sd=.08, turnout_sd=.04):
    rng = np.random.default_rng(seed)
    draws, n = posterior.logits.shape
    counted = df.counted_votes.to_numpy(int)
    # National turnout shock is shared, local turnout noise is conditional.
    expected = df.expected_votes.to_numpy(float)[None, :] * np.exp(
        rng.normal(-turnout_sd**2 / 2, turnout_sd, (draws, 1)) + rng.normal(-.03**2 / 2, .03, (draws, n)))
    total = np.clip(np.rint(expected), counted, df.electorate.to_numpy(int)).astype(int)
    total[:, df.complete.to_numpy(bool)] = counted[df.complete.to_numpy(bool)]
    remaining = total - counted
    # Unidentified residual reporting bias: shared zero-centered sensitivity,
    # scaled up where more votes remain; explicitly prior-driven uncertainty.
    unreported = remaining / np.maximum(total, 1)
    p = expit(posterior.logits + rng.normal(0, selection_sd, (draws, 1)) * unreported)
    final_a = df.candidate_a_votes.sum() + rng.binomial(remaining, p).sum(axis=1)
    totals = total.sum(axis=1)
    if (totals == 0).any():
        raise ValueError('No projected valid votes')
    return final_a / totals
