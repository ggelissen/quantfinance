"""Calculate comprehensive performance metrics for trading strategies."""
import numpy as np
import pandas as pd
from scipy import stats


def calculate_metrics(portfolio_values: pd.Series, signals: pd.Series, initial_capital: float, 
                     trading_days: int = 252) -> dict:
    """Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    portfolio_values : pd.Series
        Time series of portfolio values
    signals : pd.Series
        Time series of position signals (1, -1, or 0)
    initial_capital : float
        Starting capital
    trading_days : int
        Number of trading days per year for annualization
        
    Returns
    -------
    dict
        Dictionary of performance metrics
    """
    values = portfolio_values.dropna()
    returns = values.pct_change().dropna()
    
    # Basic metrics
    total_return = (values.iloc[-1] - initial_capital) / initial_capital * 100
    
    # Drawdown analysis
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Calculate drawdown duration
    is_dd = drawdown < 0
    dd_periods = is_dd.astype(int).groupby((~is_dd).cumsum()).cumsum()
    max_dd_duration = dd_periods.max() if len(dd_periods) > 0 else 0
    
    # Risk-adjusted returns
    if returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_days)
        sortino_ratio = (returns.mean() / returns[returns < 0].std()) * np.sqrt(trading_days) if len(returns[returns < 0]) > 0 else 0
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    
    # Calmar ratio (annualized return / abs(max drawdown))
    annualized_return = (1 + total_return/100) ** (trading_days / len(values)) - 1
    calmar_ratio = annualized_return / abs(max_drawdown/100) if max_drawdown != 0 else 0
    
    # Win rate and profit factor
    winning_days = returns[returns > 0]
    losing_days = returns[returns < 0]
    
    win_rate = len(winning_days) / len(returns) * 100 if len(returns) > 0 else 0
    avg_win = winning_days.mean() if len(winning_days) > 0 else 0
    avg_loss = abs(losing_days.mean()) if len(losing_days) > 0 else 0
    
    profit_factor = (winning_days.sum() / abs(losing_days.sum())) if len(losing_days) > 0 and losing_days.sum() != 0 else 0
    
    # Trade analysis
    signal_changes = signals.diff().fillna(signals)
    num_trades = (signal_changes.abs() > 0).sum()
    
    # Count long and short trades
    long_entries = ((signal_changes == 1) | ((signals == 1) & (signals.shift(1).fillna(0) == 0))).sum()
    short_entries = ((signal_changes == -1) | ((signals == -1) & (signals.shift(1).fillna(0) == 0))).sum()
    
    # Average holding period
    in_position = signals != 0
    position_periods = in_position.astype(int).groupby((~in_position).cumsum()).sum()
    avg_holding_period = position_periods[position_periods > 0].mean() if len(position_periods[position_periods > 0]) > 0 else 0
    
    # Volatility
    annual_volatility = returns.std() * np.sqrt(trading_days) * 100
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5) * 100
    
    # Expected Shortfall (CVaR)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Statistical tests
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Tail ratio (95th percentile / 5th percentile)
    tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
    
    return {
        # Returns
        'total_return': total_return,
        'annualized_return': annualized_return * 100,
        'daily_return_mean': returns.mean() * 100,
        
        # Risk
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_dd_duration,
        'annual_volatility': annual_volatility,
        'var_95': var_95,
        'cvar_95': cvar_95,
        
        # Risk-adjusted
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        
        # Trade statistics
        'num_trades': num_trades,
        'long_trades': long_entries,
        'short_trades': short_entries,
        'avg_holding_period': avg_holding_period,
        'win_rate': win_rate,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100,
        'profit_factor': profit_factor,
        
        # Distribution
        'skewness': skewness,
        'kurtosis': kurtosis,
        'tail_ratio': tail_ratio,
        
        # Series for plotting
        'returns': returns,
        'drawdown': drawdown,
        'cummax': cummax,
    }
