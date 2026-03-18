"""Volatility-adjusted signal generation for the S&P 500 strategy.

The signal S(t, l) is defined as:

    S(t, l) = AutoCov(t, l, W) + VolFloor(t, w, K, q)

where

    AutoCov(t, l, W) = (1/W) * sum_{i=0}^{W-1}
                            (r_{t-i} - r_bar_{t,W}) * (r_{t-i-l} - r_bar_{t-l,W})

is the rolling lagged auto-covariance of log-returns at lag ``l`` over window ``W``,
and

    VolFloor(t, w, K, q) = Quantile_q ( { (1/w) * sum_{j=0}^{w-1} r^2_{t-k-j} }_{k=0}^{K} )

is the q-th quantile of a rolling set of K short-horizon realised variances,
each computed over a sub-window of size ``w``.

The position rule is:
    * S(t, l) < -vol_threshold  → Long  (+1)   (low / mean-reverting volatility)
    * S(t, l) >  vol_threshold  → Flat  ( 0) / Short (-1) (volatility spike)
    * otherwise                  → Flat  ( 0)
"""

import numpy as np
import pandas as pd


def _rolling_autocovariance(returns: pd.Series, window: int, lag: int) -> pd.Series:
    """Compute rolling lagged auto-covariance at a fixed lag.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    window : int
        Rolling window length W.
    lag : int
        Auto-covariance lag l (number of periods).

    Returns
    -------
    pd.Series
        Series of rolling auto-covariance values, NaN for the warm-up period.
    """
    n = len(returns)
    values = returns.values
    result = np.full(n, np.nan)

    for t in range(window + lag - 1, n):
        r_window = values[t - window + 1: t + 1]           # r_{t-W+1} … r_t
        r_lagged = values[t - window + 1 - lag: t + 1 - lag]  # same window, shifted by lag

        mean_r = np.mean(r_window)
        mean_r_lag = np.mean(r_lagged)
        result[t] = np.mean((r_window - mean_r) * (r_lagged - mean_r_lag))

    return pd.Series(result, index=returns.index)


def _rolling_volatility_floor(returns: pd.Series, sub_window: int, k_windows: int,
                               quantile: float) -> pd.Series:
    """Compute the quantile-based volatility floor.

    At each time t we collect K rolling realised variances
    ``{ (1/w) * sum_{j=0}^{w-1} r^2_{t-k-j} }_{k=0}^{K-1}``
    and return their ``quantile``-th quantile.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    sub_window : int
        Sub-window size w for each realised variance estimate.
    k_windows : int
        Number of non-overlapping (shifted) sub-windows K.
    quantile : float
        Quantile level q ∈ (0, 1].

    Returns
    -------
    pd.Series
        Series of volatility floor values.
    """
    n = len(returns)
    values = returns.values
    result = np.full(n, np.nan)

    required = sub_window + k_windows - 1

    for t in range(required - 1, n):
        rv_samples = np.array([
            np.mean(values[t - k - sub_window + 1: t - k + 1] ** 2)
            for k in range(k_windows)
        ])
        result[t] = np.quantile(rv_samples, quantile)

    return pd.Series(result, index=returns.index)


def compute_signal(returns: pd.Series, window: int = 60, lag: int = 1,
                   sub_window: int = 21, k_windows: int = 10,
                   quantile: float = 0.25) -> pd.Series:
    """Compute the composite signal S(t, l) = AutoCov + VolFloor.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    window : int, optional
        Auto-covariance window W. Default 60.
    lag : int, optional
        Auto-covariance lag l. Default 1.
    sub_window : int, optional
        Sub-window size w for realised variance. Default 21 (≈ 1 month).
    k_windows : int, optional
        Number of rolling sub-windows K. Default 10.
    quantile : float, optional
        Quantile level q for the volatility floor. Default 0.25.

    Returns
    -------
    pd.Series
        Composite signal values indexed by date.
    """
    autocov = _rolling_autocovariance(returns, window=window, lag=lag)
    vol_floor = _rolling_volatility_floor(returns, sub_window=sub_window,
                                          k_windows=k_windows, quantile=quantile)
    signal = autocov + vol_floor
    signal.name = "signal"
    return signal


def generate_positions(signal: pd.Series, rolling_window: int = 252,
                       low_quantile: float = 0.4, high_quantile: float = 0.6,
                       allow_short: bool = False) -> pd.Series:
    """Convert the composite signal into trade positions using rolling quantile thresholds.

    The signal S(t, l) represents the current volatility regime.  Rather than
    comparing to a fixed threshold, we compare it to rolling quantiles of its
    own history so that the strategy adapts to changing market regimes:

    * signal < rolling low_quantile   → Long  (+1)  — low-vol regime
    * signal > rolling high_quantile  → Short (-1) if allow_short else Flat (0)
    * otherwise                        → Flat  ( 0)

    Parameters
    ----------
    signal : pd.Series
        Output of :func:`compute_signal`.
    rolling_window : int, optional
        Look-back window for computing rolling quantile thresholds. Default 252.
    low_quantile : float, optional
        Signal below this rolling quantile triggers a long. Default 0.4.
    high_quantile : float, optional
        Signal above this rolling quantile triggers a short / exit. Default 0.6.
    allow_short : bool, optional
        Whether to allow short positions. Default False.

    Returns
    -------
    pd.Series
        Integer positions (+1, 0, -1) indexed by date.
    """
    low_thresh = signal.rolling(rolling_window, min_periods=rolling_window // 2).quantile(low_quantile)
    high_thresh = signal.rolling(rolling_window, min_periods=rolling_window // 2).quantile(high_quantile)

    positions = pd.Series(0, index=signal.index, name="position", dtype=int)
    positions[signal < low_thresh] = 1
    if allow_short:
        positions[signal > high_thresh] = -1
    return positions


def build_volatility_surface(returns: pd.Series, window: int = 60,
                              lag_range: range = range(1, 31),
                              sub_window: int = 21, k_windows: int = 10,
                              quantile: float = 0.25) -> pd.DataFrame:
    """Compute S(t, l) for a range of lags to produce a volatility surface.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    window : int, optional
        Auto-covariance window W. Default 60.
    lag_range : range or list of int, optional
        Lags l to compute. Default range(1, 31).
    sub_window : int, optional
        Sub-window size w. Default 21.
    k_windows : int, optional
        Number of sub-windows K. Default 10.
    quantile : float, optional
        Quantile level q. Default 0.25.

    Returns
    -------
    pd.DataFrame
        DataFrame with lags as columns and dates as rows.
    """
    surface = {}
    for lag in lag_range:
        surface[lag] = compute_signal(returns, window=window, lag=lag,
                                      sub_window=sub_window, k_windows=k_windows,
                                      quantile=quantile)
    return pd.DataFrame(surface)
