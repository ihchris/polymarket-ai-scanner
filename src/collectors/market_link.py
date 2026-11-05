"""Resolve Polymarket links through allowlisted official APIs, never user URLs."""
from dataclasses import dataclass
import json
import re
from urllib.parse import urlsplit, unquote
import httpx
from pydantic import ValidationError
from src.collectors.polymarket import GAMMA, CLOB, parse_market, parse_book
from src.utils.validation import utcnow


@dataclass(frozen=True)
class MarketLink:
    kind: str
    slug: str
    market_slug: str | None = None


def parse_link(url):
    parsed = urlsplit(url.strip())
    if (parsed.scheme != 'https' or parsed.hostname not in {'polymarket.com', 'www.polymarket.com'}
            or parsed.username or parsed.password or parsed.port not in {None, 443}):
        raise ValueError('Use um link HTTPS de polymarket.com/event/... ou /market/...')
    parts = [unquote(p) for p in parsed.path.strip('/').split('/')]
    if (len(parts) not in {2, 3} or parts[0] not in {'event', 'market'}
            or (parts[0] == 'market' and len(parts) != 2)
            or any(not re.fullmatch(r'[A-Za-z0-9-]{1,250}', p) for p in parts[1:])):
        raise ValueError('Link inválido: abra a disputa no Polymarket e copie seu endereço completo')
    return MarketLink(parts[0], parts[1], parts[2] if len(parts) == 3 else None)


def lookup_link(client, url):
    link = parse_link(url)
    if link.kind == 'market':
        raw = client.get(GAMMA, f'/markets/slug/{link.slug}')
        rows = [raw]
    else:
        event = client.get(GAMMA, f'/events/slug/{link.slug}')
        rows = event.get('markets', [])
        if link.market_slug:
            rows = [r for r in rows if r.get('slug') == link.market_slug]
    if not rows:
        raise ValueError('Nenhuma disputa encontrada para este link')
    markets, skipped = [], 0
    for raw in rows:
        try:
            markets.append(parse_market(raw))
        except (ValueError, KeyError, TypeError):
            skipped += 1
    if not markets:
        raise ValueError('A API não retornou mercados com dados válidos para esta disputa')
    return {'url': url.strip(), 'markets': markets, 'skipped': skipped,
            'fetched_at': utcnow().isoformat()}


def refresh_selected_market(client, market_id, context_url):
    """Use a stable Gamma ID, falling back to a freshly fetched original event.

    Never use stale metadata to confirm a resolution if refresh fails.
    """
    market_id = str(market_id)
    if not re.fullmatch(r'[0-9]+', market_id):
        raise ValueError('Identificador de mercado inválido; analise o link novamente')
    try:
        market = parse_market(client.get(GAMMA, f'/markets/{market_id}'))
        if market.id != market_id:
            raise ValueError('Resposta Gamma não corresponde ao mercado selecionado')
        return market
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        # Some deployments/API routes can fail on the individual-market lookup
        # while the event endpoint remains available. Both paths fetch live data.
        context = lookup_link(client, context_url)
        for market in context['markets']:
            if market.id == market_id:
                return market
        raise ValueError('A disputa selecionada não está mais disponível neste evento; analise o link novamente')


def lookup_error_message(exc):
    """Actionable public error without printing headers, bodies or credentials."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return 'A API da Polymarket limitou as consultas (HTTP 429). Aguarde e tente novamente.'
        if code in {401, 403}:
            return f'A API da Polymarket recusou o acesso deste servidor (HTTP {code}). Verifique os logs da hospedagem.'
        if code == 404:
            return 'A API não encontrou a disputa (HTTP 404). Analise o link novamente e selecione o candidato.'
        return f'A API da Polymarket respondeu com erro HTTP {code}. Tente atualizar novamente.'
    if isinstance(exc, httpx.TimeoutException):
        return 'A API da Polymarket demorou para responder. Tente atualizar novamente.'
    if isinstance(exc, httpx.NetworkError):
        return 'Não foi possível conectar à API da Polymarket a partir deste servidor. Tente novamente.'
    if isinstance(exc, ValueError) and not isinstance(exc, ValidationError):
        # Locally generated validation messages are useful; malformed API JSON
        # is represented by the generic schema-validation message below.
        if isinstance(exc, json.JSONDecodeError):
            return 'A API retornou uma resposta inválida. Tente atualizar novamente.'
        return str(exc)
    return 'A API retornou dados incompatíveis com o mercado selecionado. Consulte os logs da hospedagem.'


def inspect_market(client, market, now=None):
    """Winner requires explicit CLOB winner flags AND closed status agreement.

    Gamma prices of 1/0, endDate, favorite, or closed alone never prove a winner.
    API agreement is not independent external validation of the real-world event.
    """
    now = now or utcnow()
    checks, warnings = [], []
    status, winner = 'UNRESOLVED', None
    clob = None
    if market.condition_id and re.fullmatch(r'0x[0-9a-fA-F]{64}', market.condition_id):
        try:
            clob = client.get(CLOB, f'/markets/{market.condition_id}')
        except (httpx.HTTPError, ValueError):
            warnings.append('Não foi possível consultar a resolução no CLOB')
    else:
        warnings.append('Identificador de condição ausente ou inválido')
    if isinstance(clob, dict):
        checks.append({'check': 'Gamma e CLOB concordam sobre encerramento',
                       'passed': isinstance(clob.get('closed'), bool) and market.closed == clob['closed']})
        tokens = clob.get('tokens', [])
        tokens_valid = (isinstance(tokens, list) and len(tokens) == len(market.token_ids)
                        and all(isinstance(t, dict) for t in tokens))
        if tokens_valid:
            tokens_valid = (set(str(t.get('token_id')) for t in tokens) == set(market.token_ids)
                            and all(isinstance(t.get('winner'), bool) for t in tokens)
                            and len({str(t.get('token_id')) for t in tokens}) == len(tokens))
        checks.append({'check': 'Tokens de resultado conferem entre Gamma e CLOB', 'passed': bool(tokens_valid)})
        pending = (market.resolution_status or '').lower() in {'proposed', 'disputed', 'pending'}
        checks.append({'check': 'Sem disputa/proposta de resolução pendente informada', 'passed': not pending})
        if not all(c['passed'] for c in checks):
            status = 'INCONSISTENT'
        elif market.closed and clob.get('closed') is True:
            winning = [t for t in tokens if t['winner'] is True]
            if len(winning) == 1:
                token = str(winning[0]['token_id'])
                winner = market.outcomes[market.token_ids.index(token)]
                status = 'RESOLVED'
            else:
                status = 'AWAITING_RESOLUTION'
        elif any(t.get('winner') is True for t in tokens):
            status = 'INCONSISTENT'
    elif market.closed:
        status = 'AWAITING_RESOLUTION'
    quotes = []
    market.books = {}
    if not market.closed and market.active and market.accepting_orders:
        for outcome, token in zip(market.outcomes, market.token_ids):
            try:
                book = parse_book(client.get(CLOB, '/book', {'token_id': token}), token)
                market.books[outcome.upper()] = book
                quotes.append({'outcome': outcome, 'bid': book.bid, 'ask': book.ask,
                               'ask_depth_contracts': sum(l.size for l in book.asks),
                               'timestamp': book.timestamp.isoformat(),
                               'age_seconds': (now-book.timestamp).total_seconds()})
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                warnings.append(f'Livro indisponível para {outcome}')
    return {'source': 'LIVE_DATA', 'market': market.model_dump(mode='json'),
            'checked_at': now.isoformat(), 'status': status, 'winner': winner,
            'checks': checks, 'warnings': warnings, 'quotes': quotes,
            'independent_prediction': None,
            'sources': {'gamma': f'{GAMMA}/markets/slug/{market.slug}' if market.slug else f'{GAMMA}/markets/{market.id}',
                        'clob': f'{CLOB}/markets/{market.condition_id}' if clob is not None else None}}
