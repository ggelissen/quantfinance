import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_generator import generate_signals


TRANSACTION_COST = 1.0  # flat fee per trade in dollars


def run_backtest(prices, ticker_a, ticker_b, initial_capital=10_000.0,
                 window=20, entry_z=2.0, exit_z=0.0):
    """Simulate pair-trading execution and compute performance metrics.

    For each row the engine checks the signal column:
    - Signal  1 (long spread)  → long *ticker_a*, short *ticker_b*
    - Signal -1 (short spread) → short *ticker_a*, long *ticker_b*
    - Signal  0 (exit)        → close all open positions

    A flat transaction cost of $1 is charged whenever a trade is executed.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns include *ticker_a* and
        *ticker_b*).
    ticker_a : str
        Dependent asset ticker.
    ticker_b : str
        Independent asset ticker.
    initial_capital : float, optional
        Starting portfolio value in dollars. Default 10,000.
    window : int, optional
        Rolling window for spread statistics. Default 20.
    entry_z : float, optional
        Z-score entry threshold. Default 2.0.
    exit_z : float, optional
        Z-score exit threshold. Default 0.0.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'portfolio' : pd.DataFrame with 'portfolio_value' and 'signal' columns
        - 'total_return' : float, percentage return over the period
        - 'max_drawdown' : float, maximum drawdown as a negative percentage
        - 'sharpe_ratio' : float, annualised Sharpe ratio (risk-free rate = 0)
    """
    signal_df = generate_signals(prices, ticker_a, ticker_b,
                                 window=window, entry_z=entry_z, exit_z=exit_z)

    price_a = prices[ticker_a]
    price_b = prices[ticker_b]

    capital = initial_capital
    position = 0        # 1 = long spread, -1 = short spread, 0 = flat
    shares_a = 0.0
    shares_b = 0.0
    entry_price_a = 0.0
    entry_price_b = 0.0

    portfolio_values = []

    for date, row in signal_df.iterrows():
        if date not in price_a.index or date not in price_b.index:
            portfolio_values.append(capital)
            continue

        p_a = price_a.loc[date]
        p_b = price_b.loc[date]

        current_signal = row["signal"]

        if np.isnan(current_signal):
            current_signal_int = None
        else:
            current_signal_int = int(current_signal)

        # Close existing position
        if position != 0 and current_signal_int == 0:
            if position == 1:
                pnl = (p_a - entry_price_a) * shares_a - (p_b - entry_price_b) * shares_b
            else:
                pnl = -(p_a - entry_price_a) * shares_a + (p_b - entry_price_b) * shares_b
            capital += pnl - TRANSACTION_COST
            position = 0
            shares_a = 0.0
            shares_b = 0.0

        # Open new position (only when flat)
        if position == 0 and current_signal_int in (1, -1):
            hedge_ratio = row["hedge_ratio"]
            shares_a = 1.0
            shares_b = hedge_ratio
            entry_price_a = p_a
            entry_price_b = p_b
            capital -= TRANSACTION_COST
            position = current_signal_int

        # Mark-to-market
        if position == 1:
            unrealised = (p_a - entry_price_a) * shares_a - (p_b - entry_price_b) * shares_b
        elif position == -1:
            unrealised = -(p_a - entry_price_a) * shares_a + (p_b - entry_price_b) * shares_b
        else:
            unrealised = 0.0

        portfolio_values.append(capital + unrealised)

    portfolio = pd.DataFrame({
        "portfolio_value": portfolio_values,
        "signal": signal_df["signal"].values,
    }, index=signal_df.index)

    return {
        "portfolio": portfolio,
        "total_return": _total_return(portfolio["portfolio_value"], initial_capital),
        "max_drawdown": _max_drawdown(portfolio["portfolio_value"]),
        "sharpe_ratio": _sharpe_ratio(portfolio["portfolio_value"]),
    }


def _total_return(portfolio_values, initial_capital):
    """Calculate total percentage return."""
    final_value = portfolio_values.dropna().iloc[-1]
    return (final_value - initial_capital) / initial_capital * 100.0


def _max_drawdown(portfolio_values):
    """Calculate maximum drawdown as a percentage (negative number)."""
    values = portfolio_values.dropna()
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100.0
    return drawdown.min()


def _sharpe_ratio(portfolio_values, trading_days=252):
    """Annualized Sharpe ratio assuming a risk-free rate of zero."""
    values = portfolio_values.dropna()
    daily_returns = values.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days)


def plot_results(results, ticker_a, ticker_b):
    """Plot the portfolio equity curve and trade entry points.

    Parameters
    ----------
    results : dict
        Output dictionary from :func:`run_backtest`.
    ticker_a : str
        Label for the dependent asset.
    ticker_b : str
        Label for the independent asset.
    """
    portfolio = results["portfolio"]
    equity = portfolio["portfolio_value"]
    signals = portfolio["signal"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="Portfolio Value", linewidth=1.5)

    long_entries = signals[signals == 1].index
    short_entries = signals[signals == -1].index

    ax.scatter(long_entries, equity.loc[long_entries], marker="^", color="green",
               label="Long Entry", zorder=5)
    ax.scatter(short_entries, equity.loc[short_entries], marker="v", color="red",
               label="Short Entry", zorder=5)

    ax.set_title(f"Equity Curve — {ticker_a} / {ticker_b}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    plt.tight_layout()
    plt.show()
