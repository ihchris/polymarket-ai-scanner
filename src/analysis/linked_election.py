"""Explicit user binding of independent election CSV to a real binary market."""
import hashlib
from src.collectors.historical import load_history, as_of
from src.models.bayesian import forecast
from src.analysis.probability import from_forecast
from src.analysis.opportunities import evaluate
from src.utils.validation import utc, utcnow
from urllib.parse import urlsplit
import io


def analyze_election_csv(market, csv_bytes, cutoff, candidate_a, candidate_b,
                         candidate_a_outcome, source_url, confirmed, settings, now=None):
    now = now or utcnow()
    if not confirmed:
        raise ValueError('Confirme a correspondência entre os dados eleitorais e as regras da disputa')
    if (market.source != 'LIVE_DATA' or market.closed or
            {s.upper() for s in market.outcomes} != {'YES', 'NO'}):
        raise ValueError('Previsão eleitoral disponível apenas para mercados reais binários abertos')
    if not candidate_a.strip() or not candidate_b.strip() or candidate_a.strip() == candidate_b.strip():
        raise ValueError('Informe dois candidatos distintos')
    if candidate_a_outcome not in {'YES', 'NO'}:
        raise ValueError('Escolha o outcome que representa vitória do candidato A')
    source = urlsplit(source_url)
    if source.scheme != 'https' or not source.hostname or source.username or source.password:
        raise ValueError('Informe a URL HTTPS da fonte eleitoral dos dados')
    history = load_history(io.BytesIO(csv_bytes))
    cutoff = utc(cutoff)
    if cutoff > now:
        raise ValueError('Não use resultados eleitorais com instante de corte no futuro')
    snapshot = as_of(history, cutoff)
    if set(snapshot.municipality) != set(history.municipality):
        raise ValueError('Inclua os municípios sem apuração com contagem zero no instante selecionado')
    if snapshot.empty:
        raise ValueError('Nenhuma atualização disponível até o instante selecionado')
    f = forecast(snapshot, draws=settings.simulations)
    f.candidate = candidate_a.strip()
    estimates = {candidate_a_outcome: from_forecast(f),
                 ('NO' if candidate_a_outcome == 'YES' else 'YES'): from_forecast(f, 'NO')}
    # Resolution review is explicitly performed by the user, not inferred from title.
    reviewed_market = market.model_copy(update={'ambiguous_resolution': False})
    opportunities = [evaluate(reviewed_market, side, estimate, settings, now=now)
                     for side, estimate in estimates.items()]
    p = f.win_probability
    probable = candidate_a.strip() if p > .5 else candidate_b.strip() if p < .5 else 'Sem favorito'
    return {'source': 'CSV_USUARIO_NAO_VERIFICADO_AUTOMATICAMENTE', 'market_id': market.id,
            'candidate_a': candidate_a.strip(), 'candidate_b': candidate_b.strip(),
            'candidate_a_outcome': candidate_a_outcome, 'probable_winner': probable,
            'forecast': f.to_dict(), 'opportunities': opportunities,
            'source_url': source_url, 'source_sha256': hashlib.sha256(csv_bytes).hexdigest(),
            'cutoff': cutoff.isoformat(), 'analyzed_at': now.isoformat(),
            'stale': (now-utc(f.timestamp)).total_seconds() > settings.maximum_age_seconds,
            'complete': bool(snapshot.complete.all()),
            'data_validation': 'Estrutura, contagens, timestamps e correspondência confirmada pelo usuário; autenticidade externa não verificada',
            'limitations': 'Modelo de maioria simples com dois candidatos; previsão aproximada, sujeita a viés e incerteza. Não garante vencedor.'}
