"""Chronological replay. Final labels enter scoring only AFTER forecasts exist."""
import numpy as np
import pandas as pd
from src.collectors.historical import as_of
from src.models.bayesian import forecast
from src.analysis.risk import calibration, maximum_drawdown
from src.utils.validation import utc


def replay(history, cutoffs, final_share, draws=1000, seed=42, price_history=None,
           fee_per_share=None, slippage=.002, minimum_edge=.02, quantity=10,
           maximum_age_seconds=300, initial_capital=1000):
    if not 0 <= final_share <= 1:
        raise ValueError('Invalid final outcome')
    if quantity <= 0 or initial_capital <= 0 or slippage < 0 or (fee_per_share is not None and fee_per_share < 0):
        raise ValueError('Invalid trade simulation parameters')
    cutoffs = sorted(set(utc(t) for t in cutoffs))
    if not cutoffs:
        raise ValueError('No replay cutoffs')
    roster = set(history.municipality)
    rows = []
    forecasts = []
    completed = []
    for t in cutoffs:
        snapshot = as_of(history, t)
        if set(snapshot.municipality) != roster:
            raise ValueError('All municipalities need an initial prior row available at replay start')
        f = forecast(snapshot, draws=draws, seed=seed)
        forecasts.append(f)
        completed.append(bool(snapshot.complete.all()))
        rows.append({'timestamp': t.isoformat(), 'probability': f.win_probability,
                     'vote_share': f.estimated_vote_share, 'lower': f.credible_interval_95[0],
                     'upper': f.credible_interval_95[1], 'counted_pct': f.counted_votes_percentage,
                     'ask': None, 'estimated_ev': None})
    # Labels are used only here, after the point-in-time model calls above.
    y = float(final_share > .5) if final_share != .5 else .5
    p = np.array([r['probability'] for r in rows])
    clipped = np.clip(p, 1e-9, 1-1e-9)
    metrics = {'brier_score': float(np.mean((p-y)**2)),
               'log_loss': float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
               'vote_share_mae': float(np.mean(np.abs(np.array([r['vote_share'] for r in rows])-final_share))),
               'calibration': calibration(p, np.full(len(p), y)),
               'net_return': None, 'maximum_drawdown_usd': None, 'profit_usd': None,
               'note': 'Instantes da mesma eleição são correlacionados; calibração descritiva, sem validade multi-eleição.'}
    trades = []
    if price_history is not None and not price_history.empty and fee_per_share is not None:
        required = {'timestamp', 'ask', 'depth', 'source'}
        if not required.issubset(price_history.columns):
            raise ValueError('Historical prices require timestamp, ask, depth, source')
        prices = price_history.copy()
        prices['timestamp'] = pd.to_datetime(prices.timestamp.map(utc), utc=True)
        if not prices.ask.between(0, 1).all() or not np.isfinite(prices[['ask', 'depth']].to_numpy(float)).all() or (prices.depth < 0).any():
            raise ValueError('Invalid historical executable prices')
        # One entry per election, settle at final result. No leverage, no implicit
        # reinvestment of unobserved profits, no mid-election mark-to-market claim.
        for row, f, is_complete in zip(rows, forecasts, completed):
            if is_complete:
                continue
            t = pd.Timestamp(row['timestamp'])
            eligible = prices[prices.timestamp <= t].sort_values('timestamp')
            if eligible.empty:
                continue
            q = eligible.iloc[-1]
            if (t-q.timestamp).total_seconds() > maximum_age_seconds or q.depth < quantity:
                continue
            ask = float(q.ask)
            ev = f.win_probability - ask - fee_per_share - slippage
            row.update(ask=ask, estimated_ev=ev)
            robust_p = float(np.mean(f.samples > .51))
            if not trades and robust_p - ask - fee_per_share - slippage >= minimum_edge:
                cost = quantity * (ask + fee_per_share + slippage)
                if cost > initial_capital:
                    continue
                profit = quantity*y - cost
                trades.append({'timestamp': row['timestamp'], 'side': 'YES', 'quantity': quantity,
                               'ask': ask, 'cost': cost, 'profit': profit, 'source': str(q.source),
                               'settlement': 'final result; drawdown from realized equity only'})
        profits = [t['profit'] for t in trades]
        total_cost = sum(t['cost'] for t in trades)
        metrics.update(profit_usd=float(sum(profits)),
                       net_return=float(sum(profits)/total_cost) if total_cost else 0.,
                       maximum_drawdown_usd=maximum_drawdown(profits, initial_capital))
        metrics['price_source'] = sorted(prices.source.astype(str).unique().tolist())
    return {'forecasts': rows, 'metrics': metrics, 'trades': trades, 'final_share': final_share}
