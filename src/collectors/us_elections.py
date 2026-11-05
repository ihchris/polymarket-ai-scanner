"""US electoral allocation and auditable, read-only public polling adapters."""
import hashlib
import io
import re
from datetime import datetime, timezone
import httpx
import numpy as np
import pandas as pd
from src.utils.validation import utcnow, utc

ALLOCATION_SOURCE = 'https://www.archives.gov/electoral-college/allocation'
HISTORY_URL = 'https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/master/2024_US_County_Level_Presidential_Results.csv'
POLL_BASE = 'https://electionsusa.org/api/'
POLL_CREDIT = 'ElectionsUSA — https://electionsusa.org/data/ (compilation CC BY 4.0; Wikipedia-derived fields CC BY-SA 4.0)'
MATCHUPS = {'pres-2028-vance-newsom': ('JD Vance', 'Gavin Newsom'),
            'pres-2028-vance-ocasio-cortez': ('JD Vance', 'Alexandria Ocasio-Cortez'),
            'pres-2028-vance-buttigieg': ('JD Vance', 'Pete Buttigieg')}
# 2020 Census allocation, valid for 2024 and 2028; district units handled below.
STATE_INFO = [
 ('AL','Alabama',9,'South'),('AK','Alaska',3,'West'),('AZ','Arizona',11,'West'),
 ('AR','Arkansas',6,'South'),('CA','California',54,'West'),('CO','Colorado',10,'West'),
 ('CT','Connecticut',7,'Northeast'),('DE','Delaware',3,'South'),('DC','District of Columbia',3,'South'),
 ('FL','Florida',30,'South'),('GA','Georgia',16,'South'),('HI','Hawaii',4,'West'),
 ('ID','Idaho',4,'West'),('IL','Illinois',19,'Midwest'),('IN','Indiana',11,'Midwest'),
 ('IA','Iowa',6,'Midwest'),('KS','Kansas',6,'Midwest'),('KY','Kentucky',8,'South'),
 ('LA','Louisiana',8,'South'),('ME','Maine',4,'Northeast'),('MD','Maryland',10,'South'),
 ('MA','Massachusetts',11,'Northeast'),('MI','Michigan',15,'Midwest'),('MN','Minnesota',10,'Midwest'),
 ('MS','Mississippi',6,'South'),('MO','Missouri',10,'Midwest'),('MT','Montana',4,'West'),
 ('NE','Nebraska',5,'Midwest'),('NV','Nevada',6,'West'),('NH','New Hampshire',4,'Northeast'),
 ('NJ','New Jersey',14,'Northeast'),('NM','New Mexico',5,'West'),('NY','New York',28,'Northeast'),
 ('NC','North Carolina',16,'South'),('ND','North Dakota',3,'Midwest'),('OH','Ohio',17,'Midwest'),
 ('OK','Oklahoma',7,'South'),('OR','Oregon',8,'West'),('PA','Pennsylvania',19,'Northeast'),
 ('RI','Rhode Island',4,'Northeast'),('SC','South Carolina',9,'South'),('SD','South Dakota',3,'Midwest'),
 ('TN','Tennessee',11,'South'),('TX','Texas',40,'South'),('UT','Utah',6,'West'),
 ('VT','Vermont',3,'Northeast'),('VA','Virginia',13,'South'),('WA','Washington',12,'West'),
 ('WV','West Virginia',4,'South'),('WI','Wisconsin',10,'Midwest'),('WY','Wyoming',3,'West')]


def electoral_units():
    rows = []
    for state, name, votes, region in STATE_INFO:
        rows.append({'unit': state, 'state': state, 'name': name, 'electoral_votes': 2 if state in {'ME','NE'} else votes, 'region': region})
        if state in {'ME','NE'}:
            for district in range(1, votes-1):
                rows.append({'unit': f'{state}-{district}', 'state': state, 'name': f'{name} CD-{district}', 'electoral_votes': 1, 'region': region})
    return pd.DataFrame(rows)


def _read_public(url, transport=None):
    # Callers provide only constants/allowlisted race identifiers, never raw URLs.
    with httpx.Client(timeout=20, transport=transport, follow_redirects=False,
                      headers={'User-Agent': 'PolymarketAIScanner/2.0 (read-only election research)'}) as client:
        for attempt in range(3):
            try:
                response = client.get(url)
                response.raise_for_status()
                if len(response.content) > 10_000_000:
                    raise ValueError('Fonte excede o limite de 10 MB')
                return response
            except (httpx.NetworkError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {429,500,502,503,504}:
                    raise
                if attempt == 2:
                    raise
                import time
                time.sleep(.5 * 2**attempt)
    raise RuntimeError('Unreachable')


def parse_county_history(content, available_at):
    df = pd.read_csv(io.BytesIO(content), dtype={'county_fips': str})
    required = {'state_name', 'county_fips', 'votes_gop', 'votes_dem'}
    if not required.issubset(df):
        raise ValueError('Arquivo histórico público com esquema inesperado')
    if df.county_fips.isna().any() or df.county_fips.duplicated().any():
        raise ValueError('Condados ausentes/duplicados no histórico')
    counts = df[['votes_gop', 'votes_dem']].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Contagens históricas inválidas')
    grouped = df.groupby('state_name')[['votes_gop','votes_dem']].sum()
    rows = []
    for code, name, _, _ in STATE_INFO:
        if name not in grouped.index:
            raise ValueError(f'Histórico sem {name}')
        r = grouped.loc[name]
        total = float(r.votes_gop+r.votes_dem)
        if total <= 0:
            raise ValueError(f'Histórico sem votos válidos: {name}')
        rows.append({'unit': code, 'share_a': float(r.votes_gop / total), 'votes': int(total),
                     'year': 2024, 'available_at': utc(available_at).isoformat(), 'source_url': HISTORY_URL})
    return pd.DataFrame(rows)


def parse_public_polls(payload, race_id, fetched_at):
    if race_id not in MATCHUPS or payload.get('race_id') != race_id:
        raise ValueError('A fonte retornou um confronto diferente do solicitado')
    a, b = MATCHUPS[race_id]
    rows, warnings = [], []
    for poll in payload.get('polls', []):
        try:
            vals = poll['vals']
            sample = re.match(r'^\s*([0-9][0-9,]*)\b', str(poll['sample']))
            if sample is None or a not in vals or b not in vals:
                raise ValueError('Amostra ou candidatos ausentes')
            sa, sb = float(vals[a])/100, float(vals[b])/100
            source = poll['source_url']
            if not isinstance(source, str) or not source.startswith('https://'):
                raise ValueError('Pesquisa sem link de fonte HTTPS')
            end = pd.Timestamp(poll['end_date'], tz='UTC')
            if end > pd.Timestamp(utc(fetched_at)):
                raise ValueError('Pesquisa com data futura')
            pollster = poll['pollster']
            # Use original-source identity to deduplicate, never aggregator weights.
            key = f'{pollster}|{poll["end_date"]}|{source}'
            rows.append({'poll_id': hashlib.sha256(key.encode()).hexdigest()[:24], 'unit': 'US',
                'candidate_a': a, 'candidate_b': b, 'share_a': sa, 'share_b': sb,
                'sample_size': int(sample.group(1).replace(',','')), 'pollster': pollster,
                'end_date': end.isoformat(), 'available_at': utc(fetched_at).isoformat(),
                'election_year': 2028, 'source_url': source,
                'source_type': 'PUBLIC_AGGREGATOR', 'population': str(poll.get('sample',''))})
        except (ValueError, TypeError, KeyError):
            warnings.append('Uma pesquisa foi omitida por data, amostra, candidatos ou fonte inválidos')
    if not rows:
        raise ValueError('A fonte não retornou pesquisas válidas para o confronto selecionado')
    df = pd.DataFrame(rows).drop_duplicates('poll_id')
    return df, warnings


def fetch_public_scenario(race_id, transport=None):
    if race_id not in MATCHUPS:
        raise ValueError('Confronto sem conector público disponível; use CSV')
    response = _read_public(POLL_BASE+race_id+'.json', transport)
    fetched_at = utcnow()
    polls, warnings = parse_public_polls(response.json(), race_id, fetched_at)
    historical = _read_public(HISTORY_URL, transport)
    historical_at = utcnow()
    history = parse_county_history(historical.content, historical_at)
    return {'polls': polls, 'history': history, 'warnings': warnings, 'source': 'PUBLIC_DATA',
            'candidate_a': MATCHUPS[race_id][0], 'candidate_b': MATCHUPS[race_id][1],
            'fetched_at': historical_at.isoformat(), 'attribution': POLL_CREDIT,
            'poll_api': POLL_BASE+race_id+'.json', 'history_source': HISTORY_URL,
            'poll_sha256': hashlib.sha256(response.content).hexdigest(),
            'history_sha256': hashlib.sha256(historical.content).hexdigest(),
            'publisher_as_of': response.json().get('as_of')}


def demo_us_scenario(as_of=None, seed=2028):
    """Synthetic illustration, not a reconstruction of actual 2024/2028 data."""
    as_of = utc(as_of or datetime(2026,11,5,12,tzinfo=timezone.utc))
    rng = np.random.default_rng(seed)
    units = electoral_units()
    history = []
    for _, row in units.iterrows():
        history.append({'unit': row.unit, 'share_a': float(np.clip(.5+rng.normal(0,.1),.1,.9)),
                        'votes': int(rng.integers(100000,10000000)), 'year': 2024,
                        'available_at': as_of.isoformat(), 'source_url': 'DEMO'})
    polls = []
    for i in range(24):
        unit = 'US' if i < 8 else units.iloc[i].unit
        prior = .51 if unit == 'US' else next(r['share_a'] for r in history if r['unit'] == unit)
        share = float(np.clip(prior+rng.normal(0,.02),.1,.9))
        polls.append({'poll_id': f'demo-{i}', 'unit': unit, 'candidate_a': 'Republicano DEMO',
            'candidate_b': 'Democrata DEMO', 'share_a': share*.9, 'share_b': (1-share)*.9,
            'sample_size': 1000, 'pollster': f'Instituto DEMO {i%6}',
            'end_date': (as_of-pd.Timedelta(days=i*3)).isoformat(), 'available_at': as_of.isoformat(),
            'election_year': 2028, 'source_url': 'DEMO', 'source_type': 'DEMO'})
    return {'polls': pd.DataFrame(polls), 'history': pd.DataFrame(history), 'source': 'DEMO',
            'candidate_a': 'Republicano DEMO', 'candidate_b': 'Democrata DEMO',
            'fetched_at': as_of.isoformat(), 'warnings': [], 'attribution': 'DADOS INTEIRAMENTE SIMULADOS'}
