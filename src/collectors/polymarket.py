"""Read-only official Gamma/CLOB clients. No wallet, trading or credentials."""
import json
import time
from datetime import datetime
from typing import Literal
import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.utils.validation import utc, utcnow

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'


class Level(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    price: float = Field(ge=0, le=1)
    size: float = Field(gt=0)


class Book(BaseModel):
    token_id: str
    bids: list[Level] = Field(default_factory=list)
    asks: list[Level] = Field(default_factory=list)
    timestamp: datetime
    fee_metadata: dict | None = None

    @field_validator('timestamp')
    @classmethod
    def timezone_required(cls, v):
        return utc(v)

    @property
    def bid(self):
        return max((v.price for v in self.bids), default=None)

    @property
    def ask(self):
        return min((v.price for v in self.asks), default=None)

    def buy_quote(self, quantity):
        if quantity <= 0:
            raise ValueError('quantity must be positive')
        remaining, cost = quantity, 0.
        for level in sorted(self.asks, key=lambda v: v.price):
            fill = min(remaining, level.size)
            cost += fill * level.price
            remaining -= fill
            if remaining <= 1e-9:
                return cost / quantity
        return None


class Market(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    id: str
    title: str
    slug: str = ''
    condition_id: str | None = None
    resolution_status: str | None = None
    description: str = ''
    resolution_rules: str = ''
    category: str = 'Other'
    outcomes: list[str]
    token_ids: list[str]
    # Indicative prices are NOT executable prices.
    indicative_prices: list[float] = Field(default_factory=list)
    volume: float = Field(ge=0)
    liquidity: float = Field(ge=0)
    end_date: datetime | None = None
    active: bool = True
    closed: bool = False
    accepting_orders: bool = True
    updated_at: datetime
    books: dict[str, Book] = Field(default_factory=dict)
    source: Literal['DEMO', 'LIVE_DATA'] = 'LIVE_DATA'
    ambiguous_resolution: bool = True

    @field_validator('updated_at', 'end_date')
    @classmethod
    def aware(cls, v):
        return utc(v) if v is not None else v

    @field_validator('indicative_prices')
    @classmethod
    def valid_prices(cls, v):
        from src.utils.validation import probability
        return [probability(p) for p in v]

    @model_validator(mode='after')
    def aligned(self):
        if len(self.outcomes) != len(self.token_ids) or not self.outcomes or len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError('Invalid outcome/token mapping')
        if self.indicative_prices and len(self.indicative_prices) != len(self.outcomes):
            raise ValueError('Invalid indicative prices')
        return self

    def eligible(self, minimum_liquidity=0, now=None):
        now = now or utcnow()
        return (self.active and not self.closed and self.accepting_orders and
                self.liquidity >= minimum_liquidity and (self.end_date is None or self.end_date > now))


def array(value):
    return json.loads(value) if isinstance(value, str) else value


def parse_market(raw):
    return Market(id=str(raw['id']), title=raw['question'], slug=raw.get('slug') or '',
        condition_id=raw.get('conditionId'), resolution_status=raw.get('umaResolutionStatus'),
        description=raw.get('description') or '',
        resolution_rules=raw.get('resolutionSource') or '', category=raw.get('category') or 'Other',
        outcomes=array(raw['outcomes']), token_ids=array(raw['clobTokenIds']),
        indicative_prices=array(raw.get('outcomePrices') or []), volume=float(raw.get('volume') or 0),
        liquidity=float(raw.get('liquidity') or 0), end_date=raw.get('endDate'),
        active=raw.get('active', False), closed=raw.get('closed', True),
        accepting_orders=raw.get('acceptingOrders', False), updated_at=raw['updatedAt'])


def parse_book(raw, token_id):
    if str(raw['asset_id']) != token_id:
        raise ValueError('Order book asset mismatch')
    ts = datetime.fromtimestamp(float(raw['timestamp']) / 1000, tz=utcnow().tzinfo)
    return Book(token_id=token_id, bids=raw.get('bids', []), asks=raw.get('asks', []), timestamp=ts)


class PolymarketClient:
    def __init__(self, transport=None, retries=3, sleep=time.sleep):
        self.http = httpx.Client(timeout=12, transport=transport, headers={'User-Agent': 'AnalysisScanner/2.0'})
        self.retries, self.sleep = retries, sleep
        self.warnings = []

    def close(self):
        self.http.close()

    def get(self, base, path, params=None):
        for attempt in range(self.retries + 1):
            try:
                response = self.http.get(base + path, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise
                if attempt == self.retries:
                    raise
                retry = exc.response.headers.get('Retry-After', '')
                self.sleep(min(30, float(retry)) if retry.replace('.', '', 1).isdigit() else min(8, .5 * 2**attempt))
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == self.retries:
                    raise
                self.sleep(min(8, .5 * 2**attempt))
        raise RuntimeError('Unreachable retry state')

    def markets(self, minimum_liquidity=100, max_markets=25):
        result = []
        self.warnings = []
        # Bounded pagination; order books only for eligible binary markets.
        for offset in range(0, 500, 100):
            rows = self.get(GAMMA, '/markets', {'active': 'true', 'closed': 'false', 'limit': 100, 'offset': offset})
            if not isinstance(rows, list):
                raise ValueError('Gamma response is not a list')
            for raw in rows:
                try:
                    market = parse_market(raw)
                    if not market.eligible(minimum_liquidity) or {s.upper() for s in market.outcomes} != {'YES', 'NO'}:
                        continue
                    for outcome, token in zip(market.outcomes, market.token_ids):
                        try:
                            book = parse_book(self.get(CLOB, '/book', {'token_id': token}), token)
                            # Store market fee metadata without inventing a fee formula.
                            try:
                                book.fee_metadata = self.get(CLOB, '/fee-rate', {'token_id': token})
                            except (httpx.HTTPError, ValueError):
                                pass
                            market.books[outcome.upper()] = book
                        except (httpx.HTTPError, ValueError, KeyError) as exc:
                            self.warnings.append(f'{market.id}/{outcome}: {type(exc).__name__}')
                    result.append(market)
                    if len(result) >= max_markets:
                        return result
                except (ValueError, KeyError, TypeError) as exc:
                    self.warnings.append(f'Invalid market skipped: {type(exc).__name__}')
            if len(rows) < 100:
                break
        return result


def demo_markets(progress=0, now=None):
    from datetime import timedelta
    import math
    now = now or utcnow()
    yes = round(.49 + .08 * math.sin(progress * 7), 3)
    markets = []
    for i, title in enumerate(['SIMULADO — Candidate A wins?', 'SIMULADO — Chuva amanhã?']):
        prices = [yes if i == 0 else .4, 1 - yes if i == 0 else .6]
        tokens = [f'demo-{i}-yes', f'demo-{i}-no']
        books = {}
        for side, token, price in zip(['YES', 'NO'], tokens, prices):
            books[side] = Book(token_id=token, bids=[Level(price=price-.015, size=1000)],
                asks=[Level(price=price+.015, size=100), Level(price=price+.025, size=1000)],
                timestamp=now, fee_metadata={'demo_fee_per_share': 0})
        markets.append(Market(id=f'demo-{i}', title=title, category='Eleições' if i == 0 else 'Clima',
            description='Mercado fictício. Sem negociação real.', resolution_rules='Candidate A: maioria dos votos válidos da eleição sintética.' if i == 0 else 'Exemplo sem modelo.',
            outcomes=['YES', 'NO'], token_ids=tokens, indicative_prices=prices, volume=50000,
            liquidity=10000, end_date=now + timedelta(days=1), updated_at=now,
            books=books, source='DEMO', ambiguous_resolution=False))
    return markets
