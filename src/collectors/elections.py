"""Synthetic data only. Hidden final results never enter model input."""
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy.special import expit, logit

BASE_TIME = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)


class DemoElection:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        n = 100
        state = np.arange(n) % 3
        urban = rng.uniform(-1, 1, n)
        electorate = rng.integers(1000, 15000, n)
        previous = expit(-.1 + np.array([-.25, .2, .05])[state] + .5 * urban + rng.normal(0, .25, n))
        # Turnout and current outcomes are private to the simulator.
        turnout = rng.binomial(electorate, .78)
        current = expit(logit(previous) + .12 + np.array([.08, -.04, .02])[state] + .15 * urban + rng.normal(0, .10, n))
        self._final = pd.DataFrame({'municipality': [f'M{i:03}' for i in range(n)], 'state': state.astype(str),
            'previous_share': previous, 'expected_votes': electorate * .78,
            'electorate': electorate, 'urban': urban, 'latitude': -20 + state * 3 + rng.normal(0, .5, n),
            'longitude': -45 + urban, 'final_votes': turnout, 'final_a': rng.binomial(turnout, current)})
        # Early reporting systematically depends on observed regional and urban features.
        self._start = np.clip(.25 + .22 * urban + .08 * state + rng.uniform(-.07, .07, n), 0, .65)
        self._seed = seed

    def snapshot(self, progress):
        if not 0 <= progress <= 1:
            raise ValueError('progress must be in [0,1]')
        d = self._final
        fraction = np.clip((progress - self._start) / (1 - self._start), 0, 1)
        counted = np.floor(d.final_votes.to_numpy() * fraction).astype(int)
        # Nested random vote order, deterministic and monotonic at all progress points.
        a = []
        for i, row in d.iterrows():
            rng = np.random.default_rng(self._seed * 1000 + i)
            votes = np.concatenate([np.ones(int(row.final_a), dtype=np.int8),
                                    np.zeros(int(row.final_votes - row.final_a), dtype=np.int8)])
            rng.shuffle(votes)
            a.append(int(votes[:counted[i]].sum()))
        public = d.drop(columns=['final_votes', 'final_a']).copy()
        public['counted_votes'] = counted
        public['candidate_a_votes'] = a
        public['complete'] = fraction >= 1
        public['timestamp'] = (BASE_TIME + timedelta(minutes=round(progress * 180))).isoformat()
        return public

    def outcome(self):
        """For scoring only, after a replay has generated all forecasts."""
        share = self._final.final_a.sum() / self._final.final_votes.sum()
        return {'vote_share': float(share), 'winner_a': int(share > .5)}
