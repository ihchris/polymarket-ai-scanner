from src.utils.validation import utcnow


def evaluate(market, side, estimate, settings, now=None, fee_per_share=None):
    now = now or utcnow()
    row = {'market': market.title, 'market_id': market.id, 'side': side,
           'ask_price': None, 'execution_price': None, 'model_probability': None,
           'gross_edge': None, 'estimated_costs': None, 'net_edge': None,
           'status': 'INSUFFICIENT_DATA', 'reason': '', 'source': market.source,
           'timestamp': now.isoformat()}
    def stop(status, reason):
        row.update(status=status, reason=reason)
        return row
    if not market.eligible(settings.minimum_liquidity, now):
        return stop('INSUFFICIENT_DATA', 'Mercado encerrado/inativo ou liquidez insuficiente')
    if estimate is None or not estimate.independent:
        return stop('INSUFFICIENT_DATA', 'Sem modelo disponível')
    if not (estimate.lower <= estimate.probability <= estimate.upper):
        return stop('INSUFFICIENT_DATA', 'Intervalo de probabilidade inválido')
    book = market.books.get(side)
    if book is None or book.ask is None:
        return stop('INSUFFICIENT_DATA', 'Sem ask executável')
    timestamps = [book.timestamp, market.updated_at, estimate.timestamp]
    if any((now - t).total_seconds() > settings.maximum_age_seconds or (t-now).total_seconds() > 5 for t in timestamps):
        return stop('STALE', 'Timestamp antigo ou no futuro')
    price = book.buy_quote(settings.order_size)
    row.update(ask_price=book.ask, execution_price=price, model_probability=estimate.probability,
               gross_edge=estimate.probability-book.ask, uncertainty=estimate.upper-estimate.lower,
               spread=None if book.bid is None else book.ask-book.bid)
    if price is None:
        return stop('INSUFFICIENT_DATA', 'Profundidade insuficiente para a quantidade configurada')
    fee = fee_per_share if fee_per_share is not None else settings.estimated_fee_per_share
    if fee is None:
        return stop('INSUFFICIENT_DATA', 'Taxa efetiva desconhecida; configure custo por contrato após revisar documentação')
    if not 0 <= fee <= 1:
        return stop('INSUFFICIENT_DATA', 'Taxa inválida')
    costs = fee + settings.slippage_per_share + price-book.ask
    net = row['gross_edge'] - costs
    row.update(estimated_costs=costs, net_edge=net,
               conservative_edge=estimate.lower-price-fee-settings.slippage_per_share)
    # Spread is reported, not added twice: buy cost already uses the ask.
    if net <= 0:
        return stop('NO_EDGE', 'Vantagem líquida não positiva')
    if market.ambiguous_resolution:
        return stop('UNCERTAIN', 'Regras de resolução requerem revisão independente')
    if row['conservative_edge'] < settings.minimum_edge:
        return stop('UNCERTAIN', 'Vantagem não robusta à faixa de sensibilidade')
    return stop('POSSIBLE_EDGE', 'Possível vantagem em simulação; não é recomendação de aposta')
