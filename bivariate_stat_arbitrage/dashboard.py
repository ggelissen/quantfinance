"""Professional dashboard for visualizing trading strategy performance."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from scipy import stats
from performance_metrics import calculate_metrics

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.alpha'] = 0.3


def create_dashboard(results: dict, ticker_a: str, ticker_b: str, initial_capital: float = 10000,
                     prices_a: pd.Series = None, prices_b: pd.Series = None) -> None:
    """Create a comprehensive performance dashboard.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_backtest containing 'portfolio' DataFrame
    ticker_a : str
        First ticker symbol
    ticker_b : str
        Second ticker symbol
    initial_capital : float
        Initial portfolio capital
    prices_a : pd.Series, optional
        Price series for ticker_a (for copula analysis). If None, copula plot uses dummy data
    prices_b : pd.Series, optional
        Price series for ticker_b (for copula analysis). If None, copula plot uses dummy data
    """
    portfolio = results['portfolio']
    equity = portfolio['portfolio_value']
    signals = portfolio['signal']
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(equity, signals, initial_capital)
    
    # Set up responsive figure layout
    fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.15, wspace=0.01, hspace=0.015)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.05,
        hspace=0.075,
    )
    
    # Color scheme
    color_long = '#2ecc71'   # Green
    color_short = '#e74c3c'  # Red
    color_equity = '#3498db' # Blue
    color_dd = '#e67e22'     # Orange
    
    # 1. Equity Curve with Entry Points (Top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_equity_curve(ax1, equity, signals, ticker_a, ticker_b, color_equity, color_long, color_short)
    
    # 2. Drawdown Chart (Top right)
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_drawdown(ax2, metrics['drawdown'], color_dd)
    
    # 3. Returns Distribution (Second row, left)
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_returns_distribution(ax3, metrics['returns'])
    
    # 4. Monthly Returns Heatmap (Second row, middle)
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_monthly_returns(ax4, equity)
    
    # 5. Rolling Sharpe Ratio (Second row, right)
    ax5 = fig.add_subplot(gs[1, 2])
    _plot_rolling_sharpe(ax5, equity)
    
    # 6. Trade Analysis (Third row, left)
    ax6 = fig.add_subplot(gs[2, 0])
    _plot_trade_analysis(ax6, signals, equity)
    
    # 7. Copula Analysis (Third row, middle)
    ax7 = fig.add_subplot(gs[2, 1])
    _plot_copula_analysis(ax7, prices_a, prices_b)
    
    # 8. Rolling Win Rate (Third row, right)
    ax8 = fig.add_subplot(gs[2, 2])
    _plot_rolling_win_rate(ax8, signals, equity)
    
    plt.suptitle(
        f'Statistical Arbitrage Dashboard: {ticker_a} / {ticker_b}',
        fontsize=18,
        fontweight='bold',
    )

    # Metrics summary in a separate compact window so plots keep maximum area
    metrics_fig = plt.figure(figsize=(5, 2), constrained_layout=True)
    metrics_ax = metrics_fig.add_subplot(111)
    _plot_metrics_panel(metrics_ax, metrics)
    metrics_fig.suptitle(
        f'Metrics Summary: {ticker_a} / {ticker_b}',
        fontsize=13,
        fontweight='bold',
    )

    plt.show()


def _plot_equity_curve(ax, equity, signals, ticker_a, ticker_b, color_equity, color_long, color_short):
    """Plot equity curve with trade entry markers."""
    ax.plot(equity.index, equity.values, label='Portfolio Value', 
            linewidth=2, color=color_equity, alpha=0.8)
    
    # Mark entry points
    signal_shifted = signals.shift(1).fillna(0)
    trade_starts = signals != signal_shifted
    
    long_entries = equity[trade_starts & (signals == 1)]
    short_entries = equity[trade_starts & (signals == -1)]
    
    if len(long_entries) > 0:
        ax.scatter(long_entries.index, long_entries.values, marker='^', 
                  color=color_long, s=75, label=f'Long Entry ({len(long_entries)})',
                  zorder=10, edgecolors='darkgreen', linewidths=1)
    
    if len(short_entries) > 0:
        ax.scatter(short_entries.index, short_entries.values, marker='v',
                  color=color_short, s=75, label=f'Short Entry ({len(short_entries)})',
                  zorder=10, edgecolors='darkred', linewidths=1)
    
    ax.set_title('Equity Curve & Trade Entries', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    _format_time_axis(ax)


def _plot_drawdown(ax, drawdown, color_dd):
    """Plot drawdown chart."""
    ax.fill_between(drawdown.index, drawdown.values, 0, 
                    color=color_dd, alpha=0.6, label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color=color_dd, linewidth=1.5)
    
    ax.set_title('Drawdown', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Drawdown (%)', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    _format_time_axis(ax)


def _plot_returns_distribution(ax, returns):
    """Plot distribution of non-zero returns."""
    returns = returns[(returns >= 0.001) | (returns <= -0.001)]
    returns_pct = returns * 100
    
    # Calculate extended x-axis limits to capture tails (99.5th percentile)
    p_low = np.percentile(returns_pct, 0.5)
    p_high = np.percentile(returns_pct, 99.5)
    margin = (p_high - p_low) * -0.1
    x_min = p_low - margin
    x_max = p_high + margin
    
    # Histogram with extended range
    counts, bins, patches = ax.hist(returns_pct, bins=100, alpha=0.7, color='#3498db', 
                                     edgecolor='black', density=True, range=(x_min, x_max))
    
    # Fit normal distribution across full range
    mu, sigma = returns_pct.mean(), returns_pct.std()
    x = np.linspace(x_min, x_max, 200)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, normal_curve, 'r-', linewidth=2.5, label='Normal Fit', alpha=0.8)
    
    ax.axvline(returns_pct.mean(), color='green', linestyle='--', linewidth=2, 
              label=f'Mean: {mu:.3f}%', alpha=0.8)
    ax.axvline(returns_pct.median(), color='orange', linestyle='--', linewidth=2, 
              label=f'Median: {returns_pct.median():.3f}%', alpha=0.8)
    
    ax.set_xlim([x_min, x_max])
    ax.set_title('Daily Non-Zero Returns Distribution', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Daily Return (%)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_monthly_returns(ax, equity):
    """Plot quarterly average returns heatmap by day of week."""
    returns = equity.pct_change() * 100  # Convert to percentage
    
    if len(returns) < 30:
        ax.text(0.5, 0.5, 'Insufficient data for returns heatmap', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Returns Calendar Heatmap', fontsize=12, fontweight='bold', pad=10)
        return
    
    # Create DataFrame with temporal groupings
    cal_df = pd.DataFrame({
        'Return': returns.values,
        'Year': returns.index.year,
        'Quarter': returns.index.quarter,
        'DayOfWeek': returns.index.dayofweek  # Monday=0, Sunday=6
    })
    
    # Create Year-Quarter identifier
    cal_df['YearQuarter'] = cal_df['Year'].astype(str) + '-Q' + cal_df['Quarter'].astype(str)
    
    # Calculate average return per day-of-week per quarter
    pivot = cal_df.pivot_table(values='Return', 
                               index='DayOfWeek', 
                               columns='YearQuarter', 
                               aggfunc='mean')
    
    # Determine symmetric color scale with logarithmic normalization
    if len(pivot.columns) > 0:
        data_values = pivot.values.flatten()
        data_values = data_values[~np.isnan(data_values)]
        if len(data_values) > 0:
            vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)))
        else:
            vmax = 1.0
    else:
        vmax = 1.0
    vmin = -vmax
    
    # Use symmetric log normalization for better color differentiation
    # linthresh sets the linear range around zero (prevents log(0) issues)
    linthresh = max(0.01, vmax / 100)  # 1% of max value or 0.01, whichever is larger
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
    
    # Plot heatmap with logarithmic color scaling
    sns.heatmap(pivot, 
                cmap='RdYlGn', 
                center=0,
                norm=norm,
                cbar_kws={'label': 'Avg Return (%) [log scale]', 'shrink': 0.8},
                ax=ax,
                linewidths=2,
                linecolor='white',
                square=True,
                annot=True,
                fmt='.2f',
                annot_kws={'fontsize': 8},
                mask=pivot.isnull(),
                cbar=True)
    
    # Customize labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_yticklabels([day_labels[i] for i in pivot.index], rotation=0, fontsize=9)
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    
    ax.set_title('Avg Returns by Weekday & Quarter', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Day of Week', fontsize=10)
    ax.set_xlabel('Quarter', fontsize=10)


def _plot_rolling_sharpe(ax, equity, window=60):
    """Plot rolling Sharpe ratio."""
    returns = equity.pct_change()
    rolling_sharpe = (returns.rolling(window=window).mean() / 
                     returns.rolling(window=window).std() * np.sqrt(252))
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
           linewidth=2, color='#9b59b6', label=f'{window}-day Rolling Sharpe')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (>1)')
    ax.axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor (<-1)')
    
    ax.set_title(f'Rolling Sharpe Ratio ({window}d)', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Sharpe Ratio', fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    _format_time_axis(ax)


def _plot_trade_analysis(ax, signals, equity):
    """Plot trade analysis: win/loss breakdown."""
    # Identify trade periods
    signal_changes = signals.diff().fillna(signals)
    trade_starts = (signal_changes != 0) & (signals != 0)
    trade_ends = (signal_changes != 0) & (signals.shift(-1) == 0)
    
    # Calculate P&L per trade
    trade_pnls = []
    trade_types = []
    
    start_indices = equity.index[trade_starts]
    end_indices = equity.index[trade_ends]
    
    for i, start_idx in enumerate(start_indices):
        # Find corresponding end
        future_ends = end_indices[end_indices > start_idx]
        if len(future_ends) > 0:
            end_idx = future_ends[0]
            pnl = equity.loc[end_idx] - equity.loc[start_idx]
            trade_pnls.append(pnl)
            trade_types.append('Long' if signals.loc[start_idx] == 1 else 'Short')
    
    if len(trade_pnls) > 0:
        trade_df = pd.DataFrame({'PnL': trade_pnls, 'Type': trade_types})
        
        # Separate wins and losses
        wins = trade_df[trade_df['PnL'] > 0]
        losses = trade_df[trade_df['PnL'] <= 0]
        
        # Bar plot
        x_pos = np.arange(len(trade_pnls))
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        
        ax.bar(x_pos, trade_pnls, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title(f'Trade P&L (Wins: {len(wins)}, Losses: {len(losses)})', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Trade Number', fontsize=10)
        ax.set_ylabel('P&L ($)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No completed trades', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Trade P&L', fontsize=12, fontweight='bold', pad=10)


def _plot_copula_analysis(ax, prices_a=None, prices_b=None, window=60):
    """Plot copula dependence structure: rolling Kendall's tau and tail dependence."""
    if prices_a is None or prices_b is None:
        ax.text(0.5, 0.5, 'Copula analysis requires price data\n(pass prices_a and prices_b)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Copula Dependence Analysis', fontsize=12, fontweight='bold', pad=10)
        return
    
    # Compute returns for correlation-based copula measures
    returns_a = prices_a.pct_change().dropna()
    returns_b = prices_b.pct_change().dropna()
    
    # Align series
    common_idx = returns_a.index.intersection(returns_b.index)
    returns_a = returns_a.loc[common_idx]
    returns_b = returns_b.loc[common_idx]
    
    # Compute rolling Kendall's Tau (copula-based rank correlation)
    from scipy.stats import kendalltau, spearmanr
    
    rolling_tau = []
    rolling_dates = []
    
    for i in range(window, len(returns_a)):
        subset_a = returns_a.iloc[i-window:i].values
        subset_b = returns_b.iloc[i-window:i].values
        tau, _ = kendalltau(subset_a, subset_b)
        rolling_tau.append(tau)
        rolling_dates.append(returns_a.index[i])
    
    if len(rolling_tau) > 0:
        tau_series = pd.Series(rolling_tau, index=rolling_dates)
        
        # Plot Kendall's Tau
        ax.plot(tau_series.index, tau_series.values, linewidth=2.5, color='#3498db', 
               label=f'{window}d Rolling Kendall\'s τ', marker='o', markersize=2)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong (>0.3)')
        ax.axhline(y=-0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Inverse (<-0.3)')
        
        # Shade positive/negative dependence
        ax.fill_between(tau_series.index, tau_series.values, 0,
                       where=(tau_series >= 0), alpha=0.15, color='green', label='Positive Dependence')
        ax.fill_between(tau_series.index, tau_series.values, 0,
                       where=(tau_series < 0), alpha=0.15, color='red', label='Negative Dependence')
        
        ax.set_ylim([-1.0, 1.0])
        ax.set_title(f'Copula Dependence Structure ({window}d)', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Kendall\'s τ', fontsize=10)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        _format_time_axis(ax)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for copula analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Copula Dependence Analysis', fontsize=12, fontweight='bold', pad=10)


def _plot_rolling_win_rate(ax, signals, equity, window=20):
    """Plot rolling win rate (strategy consistency)."""
    # Identify trade periods
    signal_changes = signals.diff().fillna(signals)
    trade_starts = (signal_changes != 0) & (signals != 0)
    trade_ends = (signal_changes != 0) & (signals.shift(-1) == 0)
    
    # Calculate P&L per trade
    trade_pnls = []
    trade_dates = []
    
    start_indices = equity.index[trade_starts]
    end_indices = equity.index[trade_ends]
    
    for start_idx in start_indices:
        # Find corresponding end
        future_ends = end_indices[end_indices > start_idx]
        if len(future_ends) > 0:
            end_idx = future_ends[0]
            pnl = equity.loc[end_idx] - equity.loc[start_idx]
            trade_pnls.append(1 if pnl > 0 else 0)  # 1 for win, 0 for loss
            trade_dates.append(end_idx)
    
    if len(trade_pnls) >= window:
        # Create rolling win rate series
        win_rate_data = pd.Series(trade_pnls, index=trade_dates)
        rolling_win_rate = win_rate_data.rolling(window=window).mean() * 100
        
        ax.plot(rolling_win_rate.index, rolling_win_rate.values, 
               linewidth=2, color='#27ae60', marker='o', markersize=3, label=f'{window}-trade Rolling Win Rate')
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% Breakeven')
        ax.fill_between(rolling_win_rate.index, rolling_win_rate.values, 50, 
                       where=(rolling_win_rate >= 50), alpha=0.3, color='green', label='Above Breakeven')
        ax.fill_between(rolling_win_rate.index, rolling_win_rate.values, 50,
                       where=(rolling_win_rate < 50), alpha=0.3, color='red', label='Below Breakeven')
        
        ax.set_ylim([0, 100])
        ax.set_title(f'Rolling Win Rate ({window}t)', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Win Rate (%)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        _format_time_axis(ax)
    else:
        ax.text(0.5, 0.5, 'Insufficient trades for rolling win rate', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f'Rolling Win Rate ({window}t)', fontsize=12, fontweight='bold', pad=10)


def _plot_metrics_panel(ax, metrics):
    """Plot compact metrics summary panel for responsive layouts."""
    ax.axis('off')

    left_block = (
        f"Returns\n"
        f"  Total Return: {metrics['total_return']:.2f}%\n"
        f"  Annualized: {metrics['annualized_return']:.2f}%\n"
        f"  Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
        f"\nRisk-Adjusted\n"
        f"  Sharpe: {metrics['sharpe_ratio']:.2f}\n"
        f"  Sortino: {metrics['sortino_ratio']:.2f}\n"
        f"  Calmar: {metrics['calmar_ratio']:.2f}"
    )

    mid_block = (
        f"Risk\n"
        f"  Volatility: {metrics['annual_volatility']:.2f}%\n"
        f"  VaR (95%): {metrics['var_95']:.3f}%\n"
        f"  CVaR (95%): {metrics['cvar_95']:.3f}%\n"
        f"\nTrade Stats\n"
        f"  Total Trades: {int(metrics['num_trades'])}\n"
        f"  Long / Short: {int(metrics['long_trades'])} / {int(metrics['short_trades'])}\n"
        f"  Avg Hold: {metrics['avg_holding_period']:.1f} days"
    )

    right_block = (
        f"Win/Loss\n"
        f"  Win Rate: {metrics['win_rate']:.1f}%\n"
        f"  Avg Win: {metrics['avg_win']:.3f}%\n"
        f"  Avg Loss: {metrics['avg_loss']:.3f}%\n"
        f"  Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"\nDistribution\n"
        f"  Skewness: {metrics['skewness']:.2f}\n"
        f"  Kurtosis: {metrics['kurtosis']:.2f}\n"
        f"  Tail Ratio: {metrics['tail_ratio']:.2f}"
    )

    panel_style = dict(facecolor='#ecf0f1', edgecolor='#95a5a6', boxstyle='round,pad=0.35')
    ax.text(0.02, 0.93, left_block, transform=ax.transAxes, va='top', ha='left', fontsize=8.6, bbox=panel_style)
    ax.text(0.35, 0.93, mid_block, transform=ax.transAxes, va='top', ha='left', fontsize=8.6, bbox=panel_style)
    ax.text(0.68, 0.93, right_block, transform=ax.transAxes, va='top', ha='left', fontsize=8.6, bbox=panel_style)

    ax.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=4)


def _format_time_axis(ax):
    """Use concise date ticks to avoid overlap when resizing."""
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelrotation=0)


# Import scipy.stats for normal distribution
from scipy import stats
