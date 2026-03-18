"""Main entry point for the volatility-adjusted S&P 500 strategy.

Usage
-----
    python main.py

Configuration options are set directly in the ``if __name__ == "__main__":``
block at the bottom of this file.

Strategy overview
-----------------
1. Download S&P 500 daily close data via *yfinance*.
2. Compute the composite signal S(t, l) which combines:
   - Rolling lagged auto-covariance (measures serial correlation in returns).
   - A quantile-based volatility floor (baseline risk level).
3. Generate long / flat positions: long when the signal is negative (low /
   mean-reverting vol), flat (or short) when it spikes above a threshold.
4. Run a daily back-test with flat transaction costs.
5. Display:
   - An interactive 3D volatility surface (S(t, l) over time and lag).
   - A dark-mode equity-curve dashboard with entry markers.
   - A rolling volatility regime chart.
"""

import time

import pandas as pd

from data_handler import download_sp500_data
from strategy import build_volatility_surface, compute_signal, generate_positions
from backtester import run_backtest
from visualization import (
    plot_volatility_surface,
    plot_equity_and_signals,
    plot_rolling_volatility,
)
from video_generator import generate_video as make_video


def main(
    start_date: str,
    end_date: str,
    ticker: str = "^GSPC",
    display_name: str = "S&P 500",
    # Signal parameters
    window: int = 60,
    lag: int = 1,
    sub_window: int = None,
    k_windows: int = 10,
    quantile: float = 0.25,
    # Position parameters
    rolling_window: int = 252,
    low_quantile: float = 0.4,
    high_quantile: float = 0.6,
    allow_short: bool = False,
    # Back-test parameters
    initial_capital: float = 10_000.0,
    transaction_cost: float = 1.0,
    # Surface parameters
    lag_range_end: int = 31,
    surface_last_n_days: int = 252,
    # Display
    show_plots: bool = True,
    save_html: bool = False,
    # Video
    generate_video: bool = False,
    video_output_path: str = "volatility_strategy.mp4",
    video_frame_step: int = 5,
    video_show: bool = True,
) -> dict:
    """Run the full volatility-adjusted S&P 500 strategy pipeline.

    Parameters
    ----------
    start_date : str
        Data start date, 'YYYY-MM-DD'.
    end_date : str
        Data end date, 'YYYY-MM-DD'.
    ticker : str, optional
        yfinance ticker. Default '^GSPC'.
    display_name : str, optional
        Human-readable name shown in charts. Default 'S&P 500'.
    window : int, optional
        Auto-covariance rolling window W. Default 60.
    lag : int, optional
        Auto-covariance lag l. Default 1.
    sub_window : int or None, optional
        Realised-variance sub-window.  If ``None`` (default), the improved
        formula uses ``window`` (W) for both terms.
    k_windows : int, optional
        Number K for the volatility floor. Default 10.
    quantile : float, optional
        Volatility-floor quantile level q. Default 0.25.
    rolling_window : int, optional
        Look-back window for rolling quantile position thresholds. Default 252.
    low_quantile : float, optional
        Signal below this rolling quantile triggers long entry. Default 0.4.
    high_quantile : float, optional
        Signal above this rolling quantile triggers short / exit. Default 0.6.
    allow_short : bool, optional
        Allow short positions. Default False.
    initial_capital : float, optional
        Starting capital in dollars. Default 10 000.
    transaction_cost : float, optional
        Flat fee per trade in dollars. Default 1.0.
    lag_range_end : int, optional
        Upper bound (exclusive) for lag range on surface. Default 31.
    surface_last_n_days : int, optional
        Number of recent trading days shown on the 3D surface. Default 252.
    show_plots : bool, optional
        Call ``fig.show()`` for each figure. Default True.
    save_html : bool, optional
        Save each figure as a standalone HTML file. Default False.
    generate_video : bool, optional
        Render and save the cyberpunk strategy video. Default False.
    video_output_path : str, optional
        File path for the output MP4. Default 'volatility_strategy.mp4'.
    video_frame_step : int, optional
        Advance this many trading days per video frame. Default 5.
    video_show : bool, optional
        Attempt to open the video with the system player after rendering.
        Default True.

    Returns
    -------
    dict
        Back-test results dictionary from :func:`backtester.run_backtest`.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print(f"Downloading {display_name} data ({start_date} → {end_date}) …")
    data = download_sp500_data(start_date, end_date, ticker=ticker)
    print(f"  {len(data)} trading days loaded.")

    returns = data["returns"]

    # ------------------------------------------------------------------
    # 2. Signal & positions
    # ------------------------------------------------------------------
    print("Computing composite signal S(t, l) …")
    signal = compute_signal(
        returns,
        window=window,
        lag=lag,
        sub_window=sub_window,
        k_windows=k_windows,
        quantile=quantile,
    )
    positions = generate_positions(signal, rolling_window=rolling_window,
                                   low_quantile=low_quantile,
                                   high_quantile=high_quantile,
                                   allow_short=allow_short)

    # ------------------------------------------------------------------
    # 3. Back-test
    # ------------------------------------------------------------------
    print("Running back-test …")
    results = run_backtest(data, positions, initial_capital=initial_capital,
                           transaction_cost=transaction_cost)

    portfolio = results["portfolio"]
    metrics = results["metrics"]

    print("\n── Performance Summary ─────────────────────────────────────")
    print(f"  Total Return       : {metrics['total_return']:.2f}%")
    print(f"  Annualised Return  : {metrics['annualized_return']:.2f}%")
    print(f"  Max Drawdown       : {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio       : {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio      : {metrics['sortino_ratio']:.2f}")
    print(f"  Win Rate           : {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor      : {metrics['profit_factor']:.2f}")
    print("─" * 60)

    # ------------------------------------------------------------------
    # 4. Volatility surface
    # ------------------------------------------------------------------
    print(f"Building volatility surface over lags 1–{lag_range_end - 1} …")
    surface_df = build_volatility_surface(
        returns,
        window=window,
        lag_range=range(1, lag_range_end),
        sub_window=sub_window,
        k_windows=k_windows,
        quantile=quantile,
    )

    # ------------------------------------------------------------------
    # 5. Visualisations
    # ------------------------------------------------------------------
    fig_surface = plot_volatility_surface(
        surface_df,
        title=f"Volatility Surface — {display_name}  S(t, l)",
        last_n_days=surface_last_n_days,
    )

    fig_equity = plot_equity_and_signals(
        portfolio, data,
        ticker=display_name,
        title=f"Volatility-Adjusted {display_name} Strategy",
    )

    fig_vol = plot_rolling_volatility(
        returns,
        title=f"{display_name} Rolling Volatility Regimes",
    )

    if save_html:
        fig_surface.write_html("volatility_surface.html")
        fig_equity.write_html("equity_curve.html")
        fig_vol.write_html("rolling_volatility.html")
        print("HTML files saved: volatility_surface.html, equity_curve.html, "
              "rolling_volatility.html")

    if show_plots:
        fig_surface.show()
        fig_equity.show()
        fig_vol.show()

    # ------------------------------------------------------------------
    # 6. Optional: cyberpunk video
    # ------------------------------------------------------------------
    if generate_video:
        print("Generating strategy video …")
        make_video(
            surface_df=surface_df,
            portfolio=portfolio,
            data=data,
            output_path=video_output_path,
            frame_step=video_frame_step,
            show_video=video_show,
        )

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    return results


if __name__ == "__main__":
    main(
        start_date="2010-01-01",
        end_date="2026-01-01",
        ticker="^GSPC",
        display_name="S&P 500",
        # Signal
        window=60,
        lag=1,
        k_windows=10,
        quantile=0.25,
        # Positions
        rolling_window=252,
        low_quantile=0.4,
        high_quantile=0.6,
        allow_short=False,
        # Back-test
        initial_capital=10_000.0,
        transaction_cost=1.0,
        # Surface
        lag_range_end=31,
        surface_last_n_days=252,
        # Output
        show_plots=True,
        save_html=False,
        # Video
        generate_video=False,
        video_output_path="volatility_strategy.mp4",
        video_frame_step=5,
        video_show=True,
    )
