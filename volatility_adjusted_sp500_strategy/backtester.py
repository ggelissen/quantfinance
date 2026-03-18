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
        Integer positions (+1, 0, -1).
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
    shifted = positions.shift(1).reindex(data.index).fillna(0)

    daily_returns = data["returns"].reindex(shifted.index).fillna(0.0)
    strategy_returns = shifted * daily_returns

    # Transaction costs: charged when position changes
    trades = shifted.diff().abs().fillna(0)
    trade_costs = trades * transaction_cost

    # Build portfolio value
    portfolio_value = pd.Series(index=data.index, dtype=float)
    portfolio_value.iloc[0] = initial_capital

    cum_pnl = strategy_returns.cumsum() * initial_capital
    cum_costs = trade_costs.cumsum()
    portfolio_value = initial_capital + cum_pnl - cum_costs

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
