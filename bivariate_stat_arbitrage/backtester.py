import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_generator import generate_signals_linear, generate_signals_copula
from performance_metrics import calculate_metrics
from dashboard import create_dashboard


def run_backtest(prices: pd.DataFrame, ticker_a: str, ticker_b: str, initial_capital: float = 10_000.0, 
                 window: int = 60, entry_threshold: float = 0.05, exit_threshold: float = 0.5,
                 capital_fraction: float = 0.5, transaction_costs: float = 1.0,
                 copula_generation: bool = True) -> dict:
    """Simulate pair-trading execution and compute performance metrics.

    For each row the engine checks the signal column:
    - Signal  1 (long spread)  → long *ticker_a*, short *ticker_b*
    - Signal -1 (short spread) → short *ticker_a*, long *ticker_b*
    - Signal  0 (exit)        → close all open positions

    A flat transaction cost of $[TRANSACTION_COST] is charged whenever a trade is executed.

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
        Rolling window for spread statistics. Default 60.
    entry_threshold : float, optional
        Entry threshold. Default 0.05.
    exit_threshold : float, optional
        Exit threshold. Default 0.5.
    capital_fraction : float, optional
        Fraction of capital to allocate to each trade. Default 0.5.
    transaction_costs : float, optional
        Flat fee per trade in dollars. Default 1.0.
    copula_generation : bool, optional
        If True, use copula-based signal generation. If False, use linear regression signals. Default True.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'portfolio' : pd.DataFrame with 'portfolio_value' and 'signal' columns
        - 'total_return' : float, percentage return over the period
        - 'max_drawdown' : float, maximum drawdown as a negative percentage
        - 'sharpe_ratio' : float, annualised Sharpe ratio (risk-free rate = 0)
    """
    if copula_generation:
        signal_df = generate_signals_copula(prices, ticker_a, ticker_b,
                                            window=window, entry_prob=entry_threshold, exit_prob=exit_threshold)
    else:
        signal_df = generate_signals_linear(prices, ticker_a, ticker_b,
                                            window=window, entry_z=entry_threshold, exit_z=exit_threshold)

    price_a = prices[ticker_a].reindex(signal_df.index)
    price_b = prices[ticker_b].reindex(signal_df.index)

    position = signal_df['signal']
    trade_starts = (position != 0) & (position != position.shift(1).fillna(0))

    allocated_capital = np.multiply(initial_capital, capital_fraction)
    shares_at_entry = (allocated_capital / price_a).where(trade_starts)
    fixed_shares = shares_at_entry.ffill().fillna(0.0)

    hr_at_entry = signal_df['hedge_ratio'].where(trade_starts)
    fixed_hr = hr_at_entry.ffill().fillna(0.0)

    shifted_position = position.shift(1).fillna(0.0)
    
    # Calculate raw PnL from holding the spread (long A, short B)
    # This represents the PnL if we were LONG the spread with fixed_shares
    diff_a = price_a.diff().fillna(0.0)
    diff_b = price_b.diff().fillna(0.0)
    
    spread_pnl_per_share = diff_a - np.multiply(fixed_hr, diff_b)
    
    # Position sign determines direction: +1 = long spread, -1 = short spread
    pnl = np.multiply(np.multiply(shifted_position, fixed_shares), spread_pnl_per_share)

    trades = position.diff().fillna(position).abs() > 0
    trade_costs = np.multiply(trades.astype(float), transaction_costs)

    net_daily = np.subtract(pnl.cumsum(), trade_costs.cumsum())
    portfolio_value = np.add(initial_capital, net_daily)

    portfolio = pd.DataFrame({'portfolio_value': portfolio_value, 'signal': position.values}, index=signal_df.index)

    # Calculate comprehensive metrics
    metrics = calculate_metrics(portfolio['portfolio_value'], portfolio['signal'], initial_capital)
    
    results = {
        'portfolio': portfolio,
        'total_return': metrics['total_return'],
        'max_drawdown': metrics['max_drawdown'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'metrics': metrics  # Include all metrics for dashboard
    }
    return results

def run_backtest_kelly(prices: pd.DataFrame, ticker_a: str, ticker_b: str, initial_capital: float = 10_000.0, 
                       window: int = 60, entry_threshold: float = 0.05, exit_threshold: float = 0.5, 
                       capital_fraction: float = 0.5, transaction_costs: float = 1.0, copula_generation: bool = True) -> dict:
    """Run backtest using Kelly fraction for position sizing."""
    first_pass = run_backtest(prices, ticker_a, ticker_b, initial_capital=initial_capital, window=window,
                              entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                              capital_fraction=capital_fraction, transaction_costs=transaction_costs,
                              copula_generation=copula_generation)
    
    portfolio_values = first_pass['portfolio']['portfolio_value']
    daily_returns = portfolio_values.pct_change().dropna()
    optimal_fraction = _kelly_fraction(daily_returns)

    final_pass = run_backtest(prices, ticker_a, ticker_b, initial_capital=initial_capital, window=window,
                              entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                              capital_fraction=optimal_fraction, transaction_costs=transaction_costs,
                              copula_generation=copula_generation)
    return final_pass


def _total_return(portfolio_values: pd.Series, initial_capital: float) -> float:
    """Calculate total percentage return."""
    final_value = portfolio_values.dropna().iloc[-1]
    return (final_value - initial_capital) / initial_capital * 100.0


def _max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown as a percentage (negative number)."""
    values = portfolio_values.dropna()
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100.0
    return drawdown.min()


def _sharpe_ratio(portfolio_values: pd.Series, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio assuming a risk-free rate of zero."""
    values = portfolio_values.dropna()
    daily_returns = values.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days)

def _kelly_fraction(portfolio_returns: pd.Series) -> float:
    """Calculate the Kelly fraction based on historical returns."""
    returns = portfolio_returns.dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    kelly_fraction = mean_return / (std_return ** 2)
    return float(np.clip(kelly_fraction, 0.0, 1.0))


def plot_results(results: dict, ticker_a: str, ticker_b: str, use_dashboard: bool = True, 
                 initial_capital: float = 10000, prices: pd.DataFrame = None) -> None:
    """Plot strategy performance.

    Parameters
    ----------
    results : dict
        Results dictionary from run_backtest
    ticker_a : str
        Label for the dependent asset.
    ticker_b : str
        Label for the independent asset.
    use_dashboard : bool, optional
        If True, use comprehensive dashboard. If False, use simple equity plot. Default True.
    initial_capital : float, optional
        Initial capital for metrics calculation. Default 10000.
    prices : pd.DataFrame, optional
        Price DataFrame for copula analysis. If provided, will be used in dashboard.
    """
    if use_dashboard:
        prices_a = prices[ticker_a] if prices is not None else None
        prices_b = prices[ticker_b] if prices is not None else None
        create_dashboard(results, ticker_a, ticker_b, initial_capital, prices_a, prices_b)
    else:
        # Simple equity curve plot
        portfolio = results["portfolio"]
        equity = portfolio["portfolio_value"]
        signals = portfolio["signal"]

        signal_shifted = signals.shift(1).fillna(0)
        trade_starts = signals != signal_shifted

        long_entries = portfolio[trade_starts & (signals == 1)].index
        short_entries = portfolio[trade_starts & (signals == -1)].index

        print(f"  → Plotting {len(long_entries)} long entries and {len(short_entries)} short entries")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity.index, equity.values, label="Portfolio Value", linewidth=1.5, alpha=0.7)

        if len(long_entries) > 0:
            ax.scatter(long_entries, equity.loc[long_entries], marker="^", color="green",
                       label=f"Long Entry ({len(long_entries)})", s=100, zorder=10, edgecolors='darkgreen', linewidths=1.5)
        if len(short_entries) > 0:
            ax.scatter(short_entries, equity.loc[short_entries], marker="v", color="red",
                       label=f"Short Entry ({len(short_entries)})", s=100, zorder=10, edgecolors='darkred', linewidths=1.5)

        ax.set_title(f"Equity Curve — {ticker_a} / {ticker_b}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        plt.tight_layout()
        plt.show()
