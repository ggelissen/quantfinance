"""Volatility-adjusted signal generation for the S&P 500 strategy.

The signal S(t, l) is defined as:

    S(t, l) = max( AutoCov(t, l, W), VolFloor(t, W, K, q) )

where

    AutoCov(t, l, W) = (1/(W-l)) * sum_{i=0}^{W-l-1}
                            (r_{t-i} - r_bar_{t,W}) * (r_{t-i-l} - r_bar_{t-l,W})

is the rolling lagged auto-covariance of log-returns at lag ``l`` over window ``W``,
using W-l inner pairs normalised by W-l, while the means r_bar_{t,W} and
r_bar_{t-l,W} are still computed over the full W-window, and

    VolFloor(t, W, K, q) = Quantile_q ( { (1/W) * sum_{j=0}^{W-1} r^2_{t-k-j} }_{k=0}^{K} )

is the q-th quantile of K+1 rolling realised variances, each computed over a
window of size W (the same window as the auto-covariance term).

The position rule is:
    * S(t, l) < rolling low_quantile  → Long  (+1)   (low / mean-reverting volatility)
    * S(t, l) > rolling high_quantile → Flat  ( 0) / Short (-1) (volatility spike)
    * otherwise                        → Flat  ( 0)
"""

import numpy as np
import pandas as pd


def _rolling_autocovariance(returns: pd.Series, window: int, lag: int) -> pd.Series:
    """Compute rolling lagged auto-covariance at a fixed lag.

    Implements the improved estimator

        AutoCov(t, l, W) = (1/(W-l)) * sum_{i=0}^{W-l-1}
                               (r_{t-i} - r_bar_{t,W}) * (r_{t-i-l} - r_bar_{t-l,W})

    where r_bar_{t,W} and r_bar_{t-l,W} are the full-window (W) means of the
    current and lagged return streams respectively, and the product is summed
    over the W-l inner pairs, normalised by W-l.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    window : int
        Rolling window length W.
    lag : int
        Auto-covariance lag l (number of periods).  Must satisfy lag < window.

    Returns
    -------
    pd.Series
        Series of rolling auto-covariance values, NaN for the warm-up period.
    """
    if lag >= window:
        raise ValueError(
            f"lag ({lag}) must satisfy lag < window; got lag={lag}, window={window}."
        )

    n = len(returns)
    values = returns.values
    result = np.full(n, np.nan)

    pairs = window - lag  # number of inner pairs used in the sum

    for t in range(window + lag - 1, n):
        # Full W-length windows for computing the means
        r_window = values[t - window + 1: t + 1]              # r_{t-W+1} … r_t
        r_lagged = values[t - window + 1 - lag: t + 1 - lag]  # r_{t-W+1-l} … r_{t-l}

        mean_r = np.mean(r_window)      # r_bar_{t,W}
        mean_r_lag = np.mean(r_lagged)  # r_bar_{t-l,W}

        # Inner W-l pairs: r_window[lag:] gives r_{t-W+1+l} … r_t  (W-l elements)
        #                   r_lagged[lag:] gives r_{t-W+1}   … r_{t-l} (W-l elements)
        r_inner = r_window[lag:]
        r_inner_lag = r_lagged[lag:]
        result[t] = np.sum((r_inner - mean_r) * (r_inner_lag - mean_r_lag)) / pairs

    return pd.Series(result, index=returns.index)


def _rolling_volatility_floor(returns: pd.Series, sub_window: int, k_windows: int,
                               quantile: float) -> pd.Series:
    """Compute the quantile-based volatility floor.

    At each time t we collect K+1 rolling realised variances
    ``{ (1/W) * sum_{j=0}^{W-1} r^2_{t-k-j} }_{k=0}^{K}``
    and return their ``quantile``-th quantile.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    sub_window : int
        Window size W for each realised variance estimate.
    k_windows : int
        Number of shifted sub-windows K (produces K+1 samples: k = 0 … K).
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

    required = sub_window + k_windows  # need sub_window + K positions (k = 0 … K)

    for t in range(required - 1, n):
        rv_samples = np.array([
            np.mean(values[t - k - sub_window + 1: t - k + 1] ** 2)
            for k in range(k_windows + 1)  # k = 0, 1, …, K  (K+1 samples)
        ])
        result[t] = np.quantile(rv_samples, quantile)

    return pd.Series(result, index=returns.index)


def compute_signal(returns: pd.Series, window: int = 60, lag: int = 1,
                   sub_window: int = None, k_windows: int = 10,
                   quantile: float = 0.25) -> pd.Series:
    """Compute the composite signal S(t, l) = max(AutoCov, VolFloor).

    Implements the improved equation:

        S(t, l) = max(
            (1/(W-l)) * sum_{i=0}^{W-l-1} (r_{t-i}-r_bar_{t,W})(r_{t-i-l}-r_bar_{t-l,W}),
            Quantile_q( { (1/W) * sum_{j=0}^{W-1} r^2_{t-k-j} }_{k=0}^{K} )
        )

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    window : int, optional
        Auto-covariance window W.  Also used as the realised-variance sub-window
        in the volatility floor.  Default 60.
    lag : int, optional
        Auto-covariance lag l.  Must satisfy ``lag < window``.  Default 1.
    sub_window : int or None, optional
        Override for the realised-variance sub-window.  If ``None`` (default),
        the improved formula uses ``window`` (W) for both terms.
    k_windows : int, optional
        Number K for the volatility floor; produces K+1 variance samples. Default 10.
    quantile : float, optional
        Quantile level q for the volatility floor. Default 0.25.

    Returns
    -------
    pd.Series
        Composite signal values indexed by date.
    """
    if sub_window is None:
        sub_window = window

    autocov = _rolling_autocovariance(returns, window=window, lag=lag)
    vol_floor = _rolling_volatility_floor(returns, sub_window=sub_window,
                                          k_windows=k_windows, quantile=quantile)
    signal = pd.Series(
        np.maximum(autocov.values, vol_floor.values),
        index=returns.index,
        name="signal",
    )
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
                              sub_window: int = None, k_windows: int = 10,
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
    sub_window : int or None, optional
        Override for the realised-variance sub-window.  If ``None`` (default),
        uses ``window`` (W) per the improved formula.
    k_windows : int, optional
        Number K for the volatility floor. Default 10.
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
