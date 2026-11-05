"""Independent hierarchical logistic election model.

Fast default: Laplace posterior of a binomial-logit GLMM with proper normal
priors. Fixed, documented prior scales avoid learning hyperparameters from a
single synthetic election. Optional PyMC NUTS samples the identical model.
Observed geographic covariates and state effects adjust for reporting selection
under MAR conditional on those covariates; residual MNAR is handled only by a
sensitivity term in the remaining-vote simulation, not claimed identifiable.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.linalg import cholesky, solve_triangular

MODEL_VERSION = 'hierarchical-logit-v1'


@dataclass
class Posterior:
    logits: np.ndarray
    diagnostics: dict


def validate_election(df):
    required = ['municipality', 'state', 'previous_share', 'expected_votes', 'electorate',
                'urban', 'counted_votes', 'candidate_a_votes', 'complete', 'timestamp']
    if any(c not in df for c in required) or df.empty:
        raise ValueError('Missing election data')
    if df[required].isna().any().any() or df.municipality.duplicated().any():
        raise ValueError('Missing or duplicate municipalities')
    numeric = df[['previous_share', 'expected_votes', 'electorate', 'urban', 'counted_votes', 'candidate_a_votes']].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite election input')
    if not df.previous_share.between(.000001, .999999).all():
        raise ValueError('Historical vote shares must be strictly inside (0,1)')
    if ((df.counted_votes < 0) | (df.candidate_a_votes < 0) |
        (df.candidate_a_votes > df.counted_votes) | (df.counted_votes > df.electorate) |
        (df.expected_votes <= 0) | (df.expected_votes > df.electorate)).any():
        raise ValueError('Invalid vote counts/turnout')
    if not np.equal(df[['counted_votes', 'candidate_a_votes', 'electorate']],
                    np.floor(df[['counted_votes', 'candidate_a_votes', 'electorate']])).all().all():
        raise ValueError('Vote counts must be integers')
    if not df.complete.isin([True, False, 0, 1]).all():
        raise ValueError('Invalid completion flag')
    from src.utils.validation import utc
    for timestamp in df.timestamp:
        utc(timestamp)


def sample_posterior(df, draws=10000, seed=42, backend='laplace'):
    validate_election(df)
    if draws < 1:
        raise ValueError('draws must be positive')
    states = sorted(df.state.astype(str).unique())
    state_index = np.array([states.index(str(s)) for s in df.state])
    n, k = len(df), len(states)
    # Shared national swing, urban swing, state swing, municipality residual.
    x = np.column_stack([np.ones(n), df.urban, np.eye(k)[state_index], np.eye(n)])
    scales = np.r_[.30, .20, np.full(k, .20), np.full(n, .15)]
    baseline = logit(df.previous_share.to_numpy(float))
    counts, a = df.counted_votes.to_numpy(float), df.candidate_a_votes.to_numpy(float)
    precision = 1 / scales ** 2
    if backend == 'pymc':
        import pymc as pm
        import arviz as az
        with pm.Model():
            theta = pm.Normal('theta', 0, scales, shape=len(scales))
            pm.Binomial('votes', n=counts.astype(int), logit_p=baseline + pm.math.dot(x, theta), observed=a.astype(int))
            trace = pm.sample(draws=max(250, int(np.ceil(draws / 2))), tune=750, chains=2,
                              cores=1, random_seed=seed, progressbar=False, target_accept=.95)
        samples = trace.posterior.theta.values.reshape(-1, len(scales))[:draws]
        summary = az.summary(trace, var_names=['theta'])
        diagnostics = {'backend': 'pymc', 'max_rhat': float(summary.r_hat.max()),
                       'min_ess': float(summary.ess_bulk.min()),
                       'divergences': int(trace.sample_stats.diverging.sum())}
    elif backend == 'laplace':
        def objective(theta):
            z = baseline + x @ theta
            return np.sum(counts * np.logaddexp(0, z) - a * z) + .5 * np.sum(precision * theta ** 2)
        def gradient(theta):
            return x.T @ (counts * expit(baseline + x @ theta) - a) + precision * theta
        def hessian(theta):
            p = expit(baseline + x @ theta)
            return x.T @ ((counts * p * (1 - p))[:, None] * x) + np.diag(precision)
        fit = minimize(objective, np.zeros(x.shape[1]), jac=gradient, hess=hessian,
                       method='trust-exact', options={'gtol': 1e-5, 'maxiter': 150})
        grad_norm = float(np.max(np.abs(gradient(fit.x))))
        if not fit.success and grad_norm > .01:
            raise RuntimeError(f'Posterior optimization failed: {fit.message}')
        l = cholesky(hessian(fit.x), lower=True)
        rng = np.random.default_rng(seed)
        samples = fit.x + solve_triangular(l.T, rng.normal(size=(x.shape[1], draws)), lower=False).T
        diagnostics = {'backend': 'laplace', 'gradient_max': grad_norm,
                       'approximation': 'Gaussian posterior near MAP; validate using NUTS for real analyses'}
    else:
        raise ValueError('Unknown posterior backend')
    return Posterior(baseline[None, :] + samples @ x.T, diagnostics)
