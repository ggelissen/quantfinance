"""Back-tester for the volatility-adjusted S&P 500 strategy."""

import numpy as np
import pandas as pd


def run_backtest(data: pd.DataFrame, positions: pd.Series,
                 initial_capital: float = 10_000.0,
                 transaction_cost: float = 1.0) -> dict:
    """Simulate daily portfolio value given a position series.

    The position is applied with a one-day lag (signal computed on day t is
    executed at the open of day t+1, i.e. we use the close-to-close return of
    day t+1).

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ``close`` and ``returns``.
    positions : pd.Series
        Position multipliers (e.g. 1.2, 0.5, 0, -1).
    initial_capital : float, optional
        Starting capital in dollars. Default 10 000.
    transaction_cost : float, optional
        Flat fee per trade (both sides), in dollars. Default 1.0.

    Returns
    -------
    dict
        Keys: ``portfolio``, ``total_return``, ``max_drawdown``,
        ``sharpe_ratio``, ``metrics``.
    """
    # Align positions to returns index and shift by 1 day
    shifted = positions.shift(1).reindex(data.index).fillna(0.0).astype(float)

    daily_returns = data["returns"].reindex(shifted.index).fillna(0.0).astype(float)

    # Transaction costs: charged when exposure changes
    trades = shifted.diff().abs().fillna(0.0)
    trade_costs = trades * float(transaction_cost)

    # Build compounded portfolio path
    n = len(daily_returns)
    equity = np.empty(n, dtype=float)
    net_returns = np.empty(n, dtype=float)

    equity[0] = float(initial_capital)
    net_returns[0] = 0.0

    for i in range(1, n):
        gross_multiplier = 1.0 + shifted.iloc[i] * daily_returns.iloc[i]
        next_equity = equity[i - 1] * gross_multiplier - trade_costs.iloc[i]
        equity[i] = max(next_equity, 0.0)
        prev_equity = equity[i - 1]
        net_returns[i] = (equity[i] / prev_equity - 1.0) if prev_equity > 0 else 0.0

    portfolio_value = pd.Series(equity, index=data.index, dtype=float)
    strategy_returns = pd.Series(net_returns, index=data.index, dtype=float)

    portfolio = pd.DataFrame({
        "portfolio_value": portfolio_value,
        "position": shifted,
        "strategy_return": strategy_returns,
    })

    metrics = _calculate_metrics(portfolio_value, initial_capital)

    return {
        "portfolio": portfolio,
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "metrics": metrics,
    }


def _calculate_metrics(portfolio_values: pd.Series, initial_capital: float,
                        trading_days: int = 252) -> dict:
    """Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    portfolio_values : pd.Series
        Daily portfolio value.
    initial_capital : float
        Starting capital.
    trading_days : int, optional
        Number of trading days per year. Default 252.

    Returns
    -------
    dict
        Performance metrics dictionary.
    """
    values = portfolio_values.dropna()
    daily_rets = values.pct_change().dropna()

    total_return = (values.iloc[-1] - initial_capital) / initial_capital * 100.0
    n_years = len(values) / trading_days
    annualized_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    std = daily_rets.std()
    sharpe = (daily_rets.mean() / std) * np.sqrt(trading_days) if std != 0 else 0.0

    downside = daily_rets[daily_rets < 0]
    sortino_std = downside.std()
    sortino = (daily_rets.mean() / sortino_std) * np.sqrt(trading_days) if sortino_std != 0 else 0.0

    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100.0
    max_drawdown = drawdown.min()

    # Win rate and profit factor
    trade_returns = daily_rets[daily_rets != 0]
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = len(wins) / len(trade_returns) * 100.0 if len(trade_returns) > 0 else 0.0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": int((trade_returns != 0).sum()),
    }
