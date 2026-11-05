from dataclasses import dataclass, asdict
import numpy as np
from src.models.hierarchical import sample_posterior, MODEL_VERSION
from src.models.monte_carlo import simulate_remaining
from src.utils.validation import utc


@dataclass
class Forecast:
    candidate: str
    estimated_vote_share: float
    credible_interval_95: tuple
    win_probability: float
    counted_votes_percentage: float
    model_version: str
    timestamp: str
    diagnostics: dict
    samples: np.ndarray

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != 'samples'}


def forecast(df, draws=10000, seed=42, backend='laplace', selection_sd=.08):
    posterior = sample_posterior(df, draws, seed, backend)
    shares = simulate_remaining(df, posterior, seed + 1, selection_sd=selection_sd)
    lo, hi = np.quantile(shares, [.025, .975])
    denominator = np.where(df.complete, df.counted_votes,
                           np.maximum(df.expected_votes, df.counted_votes)).sum()
    return Forecast('Candidate A', float(shares.mean()), (float(lo), float(hi)),
                    float((shares > .5).mean() + .5 * (shares == .5).mean()),
                    float(min(100, 100 * df.counted_votes.sum() / denominator)), MODEL_VERSION,
                    max(map(utc, df.timestamp)).isoformat(), posterior.diagnostics, shares)
