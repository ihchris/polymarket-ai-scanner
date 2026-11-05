"""Pre-election two-party scenario: Gaussian Bayesian polling + joint EC draws.

This forecasts P(270 EV | named candidates contest the general election).
It does NOT estimate nomination probabilities or P(candidate wins any matchup).
Prior/error scales are configurable assumptions, not historically calibrated.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import expit, logit
from scipy.linalg import cholesky, solve_triangular
from src.collectors.us_elections import electoral_units, ALLOCATION_SOURCE
from src.utils.validation import utc

MODEL_VERSION = 'us-pre-election-gaussian-v1'


class PreElectionConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, allow_inf_nan=False)
    draws: int = Field(10000, ge=500, le=100000)
    seed: int = 2028
    poll_half_life_days: float = Field(45, ge=5, le=365)
    max_poll_age_days: int = Field(365, ge=7, le=1460)
    design_effect: float = Field(1.5, ge=1, le=5)
    poll_error_floor_logit: float = Field(.06, ge=.01, le=.5)
    national_prior_sd: float = Field(.40, ge=.05, le=1)
    region_prior_sd: float = Field(.10, ge=.01, le=.5)
    state_prior_sd: float = Field(.15, ge=.01, le=.8)
    pollster_bias_sd: float = Field(.08, ge=.01, le=.5)
    common_election_error_sd: float = Field(.10, ge=.01, le=.5)
    future_drift_sd_per_sqrt_year: float = Field(.20, ge=.01, le=1)


def _validate(polls, history):
    pc = {'poll_id','unit','candidate_a','candidate_b','share_a','share_b','sample_size',
          'pollster','end_date','available_at','election_year','source_url'}
    hc = {'unit','share_a','votes','year','available_at','source_url'}
    if not pc.issubset(polls) or not hc.issubset(history):
        raise ValueError('CSV incompleto: use os modelos de pesquisas e histórico fornecidos na página')
    if polls.empty or history.empty or polls[list(pc)].isna().any().any() or history[list(hc)].isna().any().any():
        raise ValueError('Dados vazios ou campos obrigatórios ausentes')
    pn = polls[['share_a','share_b','sample_size','election_year']].to_numpy(float)
    hn = history[['share_a','votes','year']].to_numpy(float)
    if not np.isfinite(pn).all() or not np.isfinite(hn).all():
        raise ValueError('Valores numéricos não finitos')
    if (not polls.share_a.between(.00001,.99999).all() or
        not polls.share_b.between(.00001,.99999).all() or
        (polls.share_a+polls.share_b > 1.000001).any() or
        not polls.sample_size.between(50,100000).all()):
        raise ValueError('Pesquisas exigem proporções entre 0 e 1 e amostras entre 50 e 100000')
    if not history.share_a.between(.00001,.99999).all() or (history.votes <= 0).any():
        raise ValueError('Histórico exige proporções em (0,1) e votos positivos')
    for df, cols in [(polls,['sample_size','election_year']),(history,['votes','year'])]:
        if not np.equal(df[cols], np.floor(df[cols])).all().all():
            raise ValueError('Tamanhos de amostra, votos e anos devem ser inteiros')
    if polls.poll_id.duplicated().any() or history.unit.duplicated().any():
        raise ValueError('Pesquisas ou unidades históricas duplicadas')
    for df, columns in [(polls,['poll_id','unit','candidate_a','candidate_b','pollster','source_url']),
                        (history,['unit','source_url'])]:
        if any(not df[col].map(lambda v: isinstance(v,str) and bool(v.strip())).all() for col in columns):
            raise ValueError('Identificadores, candidatos, institutos e fontes devem ser textos não vazios')
    units = set(electoral_units().unit)
    if not set(polls.unit).issubset(units | {'US'}) or not set(history.unit).issubset(units):
        raise ValueError('Use siglas dos estados, ME-1/ME-2, NE-1/NE-2/NE-3 e US para pesquisa nacional')


def allocate_electors(unit_shares, units=None):
    units = electoral_units() if units is None else units
    shares = np.asarray(unit_shares, dtype=float)
    if shares.ndim != 2 or shares.shape[1] != len(units) or not np.isfinite(shares).all() or ((shares < 0)|(shares > 1)).any():
        raise ValueError('Cenários eleitorais inválidos')
    # No fractional or invented tie-breaking electors. Exact unit ties are an
    # unresolved allocation, represented by unassigned electors in that draw.
    weights = units.electoral_votes.to_numpy(int)
    a = (shares > .5) @ weights
    b = (shares < .5) @ weights
    return a, b


def forecast_us(polls, history, candidate_a, candidate_b, as_of, election_year=2028,
                config=None, source='USER_CSV'):
    config = config or PreElectionConfig()
    if source not in {'DEMO','PUBLIC_DATA','USER_CSV'}:
        raise ValueError('Origem de dados inválida')
    if election_year not in {2024,2028}:
        raise ValueError('A alocação eleitoral incluída é válida somente para 2024 e 2028')
    if not candidate_a.strip() or not candidate_b.strip() or candidate_a == candidate_b:
        raise ValueError('Informe dois candidatos distintos')
    as_of = utc(as_of)
    # The Tuesday after the first Monday in November.
    first = datetime(election_year,11,1,tzinfo=timezone.utc)
    election_day = first + pd.Timedelta(days=(7-first.weekday()) % 7 + 1)
    if as_of >= election_day + pd.Timedelta(days=1):
        raise ValueError('Este modelo é pré-eleitoral: selecione um instante anterior à eleição')
    polls, history = polls.copy(), history.copy()
    _validate(polls, history)
    polls['available_at'] = polls.available_at.map(utc)
    history['available_at'] = history.available_at.map(utc)
    polls['end_date'] = pd.to_datetime(polls.end_date, utc=True, errors='raise')
    # Exclude future publication, wrong cycles/matchups and old polls before any
    # likelihood computation. Do not fill missing states from future rows.
    mask = ((polls.available_at <= as_of) & (polls.end_date <= as_of) &
            (polls.election_year == election_year) & (polls.candidate_a == candidate_a) &
            (polls.candidate_b == candidate_b))
    eligible = polls.loc[mask].copy()
    eligible['age_days'] = (pd.Timestamp(as_of)-eligible.end_date).dt.total_seconds()/86400
    eligible = eligible[eligible.age_days <= config.max_poll_age_days].copy()
    if eligible.empty:
        raise ValueError('Sem pesquisas disponíveis para esse confronto, ciclo e instante de corte')
    history = history[(history.available_at <= as_of) & (history.year < election_year)].copy()
    units = electoral_units()
    states = list(units[~units.unit.str.contains('-')].unit)
    h = history.set_index('unit')
    if not set(states).issubset(h.index):
        raise ValueError('O histórico precisa dos 50 estados e DC disponíveis no instante selecionado')
    if h.loc[states,'year'].nunique() != 1:
        raise ValueError('Use o mesmo ciclo histórico para os 50 estados e DC')
    if not (history.year == h.loc[states,'year'].iloc[0]).all():
        raise ValueError('O histórico distrital deve usar o mesmo ciclo dos estados')
    if source != 'DEMO':
        for urls in [eligible.source_url, history.source_url]:
            if not urls.map(lambda u: isinstance(u,str) and u.startswith('https://')).all():
                raise ValueError('Dados reais exigem URLs HTTPS para rastreabilidade')
    national_base = float(np.average(h.loc[states,'share_a'], weights=h.loc[states,'votes']))
    national_logit = float(logit(national_base))
    regions = sorted(units.region.unique())
    district_units = list(units[units.unit.str.contains('-')].unit)
    pollsters = sorted(eligible.pollster.unique())
    region_start, state_start = 1, 1+len(regions)
    district_start = state_start+len(states)
    pollster_start = district_start+len(district_units)
    width = pollster_start+len(pollsters)
    x_units = np.zeros((len(units),width))
    offsets, missing_districts = [], []
    for i, row in units.iterrows():
        base = float(h.loc[row.unit,'share_a']) if row.unit in h.index else float(h.loc[row.state,'share_a'])
        if row.unit not in h.index:
            missing_districts.append(row.unit)
        offsets.append(float(logit(base))-national_logit)
        x_units[i,0] = 1
        x_units[i,region_start+regions.index(row.region)] = 1
        x_units[i,state_start+states.index(row.state)] = 1
        if '-' in row.unit:
            x_units[i,district_start+district_units.index(row.unit)] = 1
    offsets = np.array(offsets)
    scales = np.r_[config.national_prior_sd, np.full(len(regions),config.region_prior_sd),
                    np.full(len(states),config.state_prior_sd),
                    [.35 if u in missing_districts else .12 for u in district_units],
                    np.full(len(pollsters),config.pollster_bias_sd)]
    prior_mean = np.zeros(width)
    prior_mean[0] = national_logit
    design, base = [], []
    for _, poll in eligible.iterrows():
        if poll.unit == 'US':
            row = np.zeros(width)
            row[0], intercept = 1, 0.
        else:
            idx = units.index[units.unit == poll.unit][0]
            row, intercept = x_units[idx].copy(), offsets[idx]
        row[pollster_start+pollsters.index(poll.pollster)] = 1
        design.append(row)
        base.append(intercept)
    design = np.array(design)
    support = eligible.share_a+eligible.share_b
    p = (eligible.share_a/support).to_numpy(float)
    effective_n = eligible.sample_size.to_numpy(float)*support.to_numpy(float)/config.design_effect
    variance = 1/(effective_n*p*(1-p)) + (config.poll_error_floor_logit + .1*(1-support.to_numpy(float)))**2
    recency = np.exp(-np.log(2)*eligible.age_days.to_numpy(float)/config.poll_half_life_days)
    precision = recency/variance
    prior_precision = 1/scales**2
    posterior_precision = np.diag(prior_precision)+design.T @ (precision[:,None]*design)
    rhs = prior_precision*prior_mean + design.T @ (precision*(logit(p)-np.array(base)))
    l = cholesky(posterior_precision, lower=True)
    mean = solve_triangular(l.T, solve_triangular(l,rhs,lower=True), lower=False)
    rng = np.random.default_rng(config.seed)
    theta = mean + solve_triangular(l.T,rng.normal(size=(width,config.draws)),lower=False).T
    years = max(0., (election_day-as_of).total_seconds()/86400/365.25)
    # Same national error and horizon drift affect every state in one scenario.
    common = rng.normal(0,np.hypot(config.common_election_error_sd,
                                  config.future_drift_sd_per_sqrt_year*np.sqrt(years)),(config.draws,1))
    region_drift = rng.normal(0,.08*np.sqrt(years),(config.draws,len(regions)))
    state_drift = rng.normal(0,.10*np.sqrt(years),(config.draws,len(states)))
    region_idx = [regions.index(r) for r in units.region]
    state_idx = [states.index(s) for s in units.state]
    logits = offsets[None,:]+theta @ x_units.T + common + region_drift[:,region_idx] + state_drift[:,state_idx]
    shares = expit(logits)
    ev_a, ev_b = allocate_electors(shares, units)
    state_mask = ~units.unit.str.contains('-')
    popular = np.average(shares[:,state_mask.to_numpy()],axis=1,weights=h.loc[states,'votes'].to_numpy(float))
    unit_rows = []
    for i,row in units.iterrows():
        lo,hi = np.quantile(shares[:,i],[.025,.975])
        unit_rows.append({**row.to_dict(), 'projected_share_a': float(shares[:,i].mean()),
                         'lower_95': float(lo), 'upper_95': float(hi),
                         'win_probability_a': float((shares[:,i]>.5).mean()),
                         'poll_count': int((eligible.unit == row.unit).sum()),
                         'historical_basis': 'statewide_prior_with_extra_uncertainty' if row.unit in missing_districts else 'historical_unit'})
    warnings = []
    if missing_districts:
        warnings.append('Sem histórico distrital: ME/NE usam prior estadual com incerteza distrital adicional; não são tratados como winner-take-all.')
    if not (eligible.unit != 'US').any():
        warnings.append('Somente pesquisas nacionais: diferenças estaduais são extrapoladas do histórico, não pesquisas estaduais atuais.')
    if len(eligible.pollster.unique()) < 3:
        warnings.append('Menos de três institutos: forte dependência das hipóteses de erro do modelo.')
    latest = eligible.end_date.max()
    if (pd.Timestamp(as_of)-latest).total_seconds()/86400 > 30:
        warnings.append('A pesquisa mais recente tem mais de 30 dias; cenário depende de informações antigas.')
    warnings.append('Projeção condicional aos dois candidatos selecionados; não inclui indicação partidária, terceiros ou eleição contingente na Câmara.')
    return {'model_version': MODEL_VERSION, 'source': source, 'candidate_a': candidate_a, 'candidate_b': candidate_b,
            'as_of': as_of.isoformat(), 'election_day': election_day.isoformat(),
            'conditional_on_matchup': True, 'probability_a_270': float((ev_a>=270).mean()),
            'probability_b_270': float((ev_b>=270).mean()),
            'probability_no_majority': float(((ev_a<270)&(ev_b<270)).mean()),
            'electoral_votes_a_mean': float(ev_a.mean()),
            'electoral_votes_a_interval_95': [float(x) for x in np.quantile(ev_a,[.025,.975])],
            'popular_share_a_mean': float(popular.mean()),
            'popular_share_a_interval_95': [float(x) for x in np.quantile(popular,[.025,.975])],
            'ev_histogram_a': np.bincount(ev_a,minlength=539).tolist(), 'units': unit_rows,
            'polls_used': eligible.drop(columns=['age_days']).assign(
                end_date=eligible.end_date.map(lambda t:t.isoformat()),
                available_at=eligible.available_at.map(lambda t:t.isoformat())).to_dict('records'),
            'poll_count': len(eligible), 'excluded_poll_count': len(polls)-len(eligible),
            'latest_poll_end': latest.isoformat(), 'district_prior_units': missing_districts,
            'settings': config.model_dump(), 'warnings': warnings, 'allocation_source': ALLOCATION_SOURCE,
            'diagnostics': {'backend': 'conjugate_gaussian_logit_approximation',
                'joint_draws': config.draws, 'horizon_years': years,
                'calibrated': False, 'pollster_effects': len(pollsters),
                'national_future_sd_logit': float(np.hypot(config.common_election_error_sd,config.future_drift_sd_per_sqrt_year*np.sqrt(years)))}}
