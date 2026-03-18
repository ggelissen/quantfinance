import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_generator import generate_signals_basket, generate_signals_basket_vine_copula


def run_backtest(prices: pd.DataFrame, basket_tickers: list[str], initial_capital: float = 10_000.0,
                 window: int = 60, entry_threshold: float = 0.05, exit_threshold: float = 0.5,
                 capital_fraction: float = 0.5, transaction_costs: float = 1.0,
                 copula_generation: bool = True, max_drawdown_stop: float = -999.0,
                 vol_targeting: bool = False, target_annual_vol: float = 0.15,
                 vol_lookback: int = 20, min_exposure_mult: float = 0.25,
                 max_exposure_mult: float = 2.0) -> dict:
    """Simulate basket-trading execution with drawdown protection.
    
    Parameters include max_drawdown_stop: if set (e.g., -20.0), enforces position exit when drawdown reaches threshold.

    For each row the engine checks the signal column:
    - Signal  1 (long basket)  → long weighted basket
    - Signal -1 (short basket) → short weighted basket
    - Signal  0 (exit)        → close all open positions

    A flat transaction cost of $[TRANSACTION_COST] is charged whenever a trade is executed.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices.
    basket_tickers : list[str]
        Tickers included in the traded basket.
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
    basket_prices = prices[basket_tickers].dropna(how="any")
    if basket_prices.empty:
        raise ValueError("No aligned price data available for selected basket tickers.")

    if copula_generation:
        signal_df = generate_signals_basket_vine_copula(
            basket_prices,
            window=window,
            entry_prob=entry_threshold,
            exit_prob=exit_threshold,
            exit_z=exit_threshold,
        )
    else:
        signal_df = generate_signals_basket(
            basket_prices,
            window=window,
            entry_z=entry_threshold,
            exit_z=exit_threshold,
        )

    aligned_prices = basket_prices.reindex(signal_df.index)
    position = signal_df["signal"]

    weight_columns = [f"weight_{ticker}" for ticker in basket_tickers]
    missing_weights = [col for col in weight_columns if col not in signal_df.columns]
    if missing_weights:
        raise ValueError(f"Missing basket weights in signal output: {missing_weights}")

    weights = signal_df[weight_columns].iloc[0].copy()
    weights.index = basket_tickers

    returns = aligned_prices.pct_change().fillna(0.0)
    basket_returns = returns.mul(weights, axis=1).sum(axis=1)

    allocated_capital = initial_capital * capital_fraction

    if vol_targeting:
        rolling_ann_vol = basket_returns.rolling(vol_lookback).std() * np.sqrt(252)
        exposure_multiplier = (target_annual_vol / rolling_ann_vol).replace([np.inf, -np.inf], np.nan)
        exposure_multiplier = exposure_multiplier.shift(1).fillna(1.0)
        exposure_multiplier = exposure_multiplier.clip(lower=min_exposure_mult, upper=max_exposure_mult)
    else:
        exposure_multiplier = pd.Series(1.0, index=basket_returns.index)

    allocated_capital_series = allocated_capital * exposure_multiplier

    # Preliminary equity path used only for drawdown-stop detection
    shifted_position_pre = position.shift(1).fillna(0.0)
    pnl_pre = shifted_position_pre * allocated_capital_series * basket_returns
    trades_pre = position.diff().fillna(position).abs() > 0
    trade_costs_pre = trades_pre.astype(float) * transaction_costs * len(basket_tickers)
    net_daily_pre = pnl_pre - trade_costs_pre

    if max_drawdown_stop > -999.0:
        equity_pre = initial_capital + net_daily_pre.cumsum()
        running_max = equity_pre.cummax()
        running_dd = (equity_pre - running_max) / running_max * 100.0
        dd_triggered = running_dd < max_drawdown_stop
        position = position.where(~dd_triggered, 0.0)

    shifted_position = position.shift(1).fillna(0.0)
    pnl = shifted_position * allocated_capital_series * basket_returns
    trades = position.diff().fillna(position).abs() > 0
    trade_costs = trades.astype(float) * transaction_costs * len(basket_tickers)
    net_daily = pnl - trade_costs
    portfolio_value = initial_capital + net_daily.cumsum()

    portfolio = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "signal": position.values,
            "basket_return": basket_returns.values,
        },
        index=signal_df.index,
    )

    final_return = _total_return(portfolio['portfolio_value'], initial_capital)
    final_dd = _max_drawdown(portfolio['portfolio_value'])
    final_sharpe = _sharpe_ratio(portfolio['portfolio_value'])
    final_ann_vol = _annualized_vol(portfolio['portfolio_value'])

    results = {
        'portfolio': portfolio,
        'total_return': final_return,
        'max_drawdown': final_dd,
        'sharpe_ratio': final_sharpe,
        'annualized_vol': final_ann_vol,
        'target_annual_vol': float(target_annual_vol) if vol_targeting else np.nan,
        'avg_exposure_mult': float(exposure_multiplier.mean()),
        'weights': weights.to_dict(),
        'basket_tickers': basket_tickers,
        'kelly_fraction': float(capital_fraction),
        'raw_kelly_estimate': _kelly_fraction(
            portfolio['portfolio_value'].pct_change().dropna(),
            min_fraction=0.0,
            max_fraction=1.0,
            fractional_scale=1.0,
        ),
        'win_rate': float(np.mean(portfolio['portfolio_value'].diff()[portfolio['basket_return'] != 0] > 0)) if (portfolio['basket_return'] != 0).sum() > 0 else 0.0,
    }
    return results

def run_backtest_kelly(prices: pd.DataFrame, basket_tickers: list[str], initial_capital: float = 10_000.0,
                       window: int = 60, entry_threshold: float = 0.05, exit_threshold: float = 0.5, 
                       capital_fraction: float = 0.5, transaction_costs: float = 1.0, 
                       copula_generation: bool = True, max_drawdown_stop: float = -999.0,
                       vol_targeting: bool = False, target_annual_vol: float = 0.15,
                       vol_lookback: int = 20, min_exposure_mult: float = 0.25,
                       max_exposure_mult: float = 2.0, sizing_method: str = "score",
                       min_position_fraction: float = 0.05, max_position_fraction: float = 0.35,
                       sizing_base_weight: float = 0.30) -> dict:
    """Run backtest with adaptive position sizing and drawdown protection.

    sizing_method:
    - "score" (default): uses Sharpe/win-rate/drawdown/volatility to size exposure.
    - "kelly": uses regularized fractional Kelly.
    - "fixed": uses provided capital_fraction unchanged.
    """
    first_pass = run_backtest(prices, basket_tickers, initial_capital=initial_capital, window=window,
                              entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                              capital_fraction=capital_fraction, transaction_costs=transaction_costs,
                              copula_generation=copula_generation, max_drawdown_stop=max_drawdown_stop,
                              vol_targeting=vol_targeting, target_annual_vol=target_annual_vol,
                              vol_lookback=vol_lookback, min_exposure_mult=min_exposure_mult,
                              max_exposure_mult=max_exposure_mult)
    
    portfolio_values = first_pass['portfolio']['portfolio_value']
    daily_returns = portfolio_values.pct_change().dropna()

    if sizing_method == "kelly":
        optimal_fraction = _kelly_fraction(
            daily_returns,
            min_fraction=min_position_fraction,
            max_fraction=max_position_fraction,
            fractional_scale=0.35,
        )
    elif sizing_method == "fixed":
        optimal_fraction = float(np.clip(capital_fraction, min_position_fraction, max_position_fraction))
    else:
        optimal_fraction = _score_based_fraction(
            sharpe=float(first_pass.get("sharpe_ratio", 0.0)),
            win_rate=float(first_pass.get("win_rate", 0.0)),
            max_drawdown=float(first_pass.get("max_drawdown", 0.0)),
            annualized_vol=float(first_pass.get("annualized_vol", 0.0)),
            target_annual_vol=target_annual_vol,
            base_fraction=capital_fraction,
            min_fraction=min_position_fraction,
            max_fraction=max_position_fraction,
            base_weight=sizing_base_weight,
        )

    final_pass = run_backtest(prices, basket_tickers, initial_capital=initial_capital, window=window,
                              entry_threshold=entry_threshold, exit_threshold=exit_threshold,
                              capital_fraction=optimal_fraction, transaction_costs=transaction_costs,
                              copula_generation=copula_generation, max_drawdown_stop=max_drawdown_stop,
                              vol_targeting=vol_targeting, target_annual_vol=target_annual_vol,
                              vol_lookback=vol_lookback, min_exposure_mult=min_exposure_mult,
                              max_exposure_mult=max_exposure_mult)
    final_pass['kelly_fraction'] = float(optimal_fraction)
    final_pass['position_fraction'] = float(optimal_fraction)
    final_pass['sizing_method'] = sizing_method
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


def _annualized_vol(portfolio_values: pd.Series, trading_days: int = 252) -> float:
    """Annualized volatility of daily portfolio returns."""
    values = portfolio_values.dropna()
    daily_returns = values.pct_change().dropna()
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.std() * np.sqrt(trading_days))

def _kelly_fraction(
    portfolio_returns: pd.Series,
    min_fraction: float = 0.01,
    max_fraction: float = 0.50,
    fractional_scale: float = 0.25,
    pseudo_count: float = 2.0,
    min_observations: int = 30,
) -> float:
    """Estimate a smooth, regularized fractional Kelly allocation.

    Uses win/loss Kelly with Bayesian-style smoothing to avoid 0%/100% saturation,
    then applies a fractional scaling and clamps to risk bounds.
    """
    returns = portfolio_returns.dropna()
    if len(returns) < min_observations:
        return float(min_fraction)

    positive = returns[returns > 0]
    negative = returns[returns < 0]
    n_obs = len(returns)

    wins = len(positive)
    losses = len(negative)

    p_win = (wins + pseudo_count) / (wins + losses + 2.0 * pseudo_count)

    mean_win = float(positive.mean()) if wins > 0 else 0.0
    mean_loss = float(np.abs(negative.mean())) if losses > 0 else 0.0

    avg_magnitude = float(np.abs(returns).mean())
    floor_mag = max(1e-6, avg_magnitude * 0.25)
    mean_win = max(mean_win, floor_mag)
    mean_loss = max(mean_loss, floor_mag)

    payoff_ratio = mean_win / mean_loss
    raw_kelly = p_win - (1.0 - p_win) / payoff_ratio

    confidence = np.sqrt(n_obs / (n_obs + 250.0))
    adjusted_kelly = raw_kelly * confidence

    scaled_kelly = max(0.0, fractional_scale * adjusted_kelly)
    return float(np.clip(scaled_kelly, min_fraction, max_fraction))


def _score_based_fraction(
    sharpe: float,
    win_rate: float,
    max_drawdown: float,
    annualized_vol: float,
    target_annual_vol: float,
    base_fraction: float,
    min_fraction: float,
    max_fraction: float,
    base_weight: float,
) -> float:
    """Risk-aware position fraction from performance score components."""
    sharpe_component = 1.0 / (1.0 + np.exp(-sharpe))
    win_component = float(np.clip((win_rate - 0.45) / 0.20, 0.0, 1.0))
    drawdown_component = float(np.clip(1.0 - abs(max_drawdown) / 25.0, 0.0, 1.0))

    if annualized_vol > 0 and target_annual_vol > 0:
        vol_component = float(np.clip(target_annual_vol / annualized_vol, 0.5, 1.5))
    else:
        vol_component = 1.0

    score = 0.45 * sharpe_component + 0.30 * win_component + 0.25 * drawdown_component
    score_fraction = min_fraction + score * (max_fraction - min_fraction)

    blended_fraction = base_weight * base_fraction + (1.0 - base_weight) * score_fraction
    adjusted_fraction = blended_fraction * vol_component
    return float(np.clip(adjusted_fraction, min_fraction, max_fraction))


def plot_results(results: dict, basket_tickers: list[str]) -> None:
    """Plot the portfolio equity curve and trade entry points.

    Parameters
    ----------
    results : dict
        Output dictionary from :func:`run_backtest`.
    basket_tickers : list[str]
        Labels for basket assets.
    """
    portfolio = results["portfolio"]
    equity = portfolio["portfolio_value"]
    signals = portfolio["signal"]

    signal_shifted = signals.shift(1).fillna(0)
    trade_starts = signals != signal_shifted

    long_entries = portfolio[trade_starts & signals == 1].index
    short_entries = portfolio[trade_starts & signals == -1].index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="Portfolio Value", linewidth=1.5)

    ax.scatter(long_entries, equity.loc[long_entries], marker="^", color="green",
               label="Long Entry", zorder=5)
    ax.scatter(short_entries, equity.loc[short_entries], marker="v", color="red",
               label="Short Entry", zorder=5)

    basket_label = ", ".join(basket_tickers)
    ax.set_title(f"Equity Curve — Basket ({basket_label})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    plt.tight_layout()
    plt.show()
