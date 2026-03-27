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
from typing import Optional


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
                   sub_window: Optional[int] = None, k_windows: int = 10,
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
                       allow_short: bool = True,
                       returns: pd.Series | None = None,
                       prices: pd.Series | None = None,
                       low_vol_leverage: float = 1.2,
                       moderate_exposure: float = 0.5,
                       flat_exposure: float = 0.0,
                       high_vol_bear_exposure: float = -1.0,
                       trend_window: int = 20,
                       target_volatility: float = 0.15,
                       ewma_vol_span: int = 20,
                       trend_sma_window: int = 200,
                       min_target_exposure: float = 0.0,
                       max_target_exposure: float = 2.0) -> pd.Series:
    """Convert signal into regime-aware floating exposures with a fast circuit breaker.

    The signal S(t, l) is mapped to three regimes using rolling quantiles,
    then scaled by target volatility:

        * signal < rolling low_quantile   → Low volatility regime (levered long, 
                                            UNLESS fast trend breaks -> cuts to flat)
        * signal > rolling high_quantile  → High volatility regime (bear / flat,
                                            but buy-the-dip when price is above long-term trend)
        * otherwise                       → Moderate regime (partial or flat)
    """
    # 1. Calculate thresholds
    low_thresh = signal.rolling(
        rolling_window, 
        min_periods=rolling_window // 2
    ).quantile(low_quantile)
    
    high_thresh = signal.rolling(
        rolling_window, 
        min_periods=rolling_window // 2
    ).quantile(high_quantile)

    # 2. Calculate Base Exposure via Risk Parity
    if returns is not None:
        aligned_returns = returns.reindex(signal.index)
        momentum = aligned_returns.rolling(
            trend_window, 
            min_periods=max(5, trend_window // 2)
        ).mean()
        moderate_mask = momentum > 0

        ewma_vol = aligned_returns.ewm(
            span=ewma_vol_span, 
            adjust=False, 
            min_periods=max(5, ewma_vol_span // 2)
        ).std() * np.sqrt(252)
        
        base_exposure = (target_volatility / ewma_vol.replace(0.0, np.nan)).clip(
            lower=min_target_exposure,
            upper=max_target_exposure,
        )
        base_exposure = base_exposure.fillna(float(moderate_exposure)).astype(float)
    else:
        moderate_mask = pd.Series(True, index=signal.index)
        base_exposure = pd.Series(float(moderate_exposure), index=signal.index, dtype=float)

    # 3. Calculate Trend Filters (Macro and Fast Circuit Breaker)
    if prices is not None:
        aligned_prices = prices.reindex(signal.index)
        
        # Long-term macro trend (200 SMA)
        sma_long = aligned_prices.rolling(
            trend_sma_window, 
            min_periods=max(20, trend_sma_window // 2)
        ).mean()
        bull_trend = aligned_prices >= sma_long
        
        # Fast Circuit Breaker (50 EMA) to stop the lag trap
        ema_fast = aligned_prices.ewm(span=50, adjust=False).mean()
        fast_trend_safe = aligned_prices >= ema_fast
    else:
        bull_trend = pd.Series(False, index=signal.index)
        fast_trend_safe = pd.Series(False, index=signal.index)

    # 4. Initialize positions to flat
    positions = pd.Series(float(flat_exposure), index=signal.index, name="position", dtype=float)

    # Define Regimes
    low_regime = signal < low_thresh
    high_regime = signal > high_thresh
    moderate_regime = ~(low_regime | high_regime)

    # --- APPLY LOGIC ROUTING ---

    # REGIME 1: LOW VOLATILITY
    low_safe = low_regime & fast_trend_safe
    low_unsafe = low_regime & ~fast_trend_safe
    
    # Levered long if safe. If fast trend breaks, hit the circuit breaker (go flat).
    positions[low_safe] = (base_exposure[low_safe] * float(low_vol_leverage)).values
    positions[low_unsafe] = float(flat_exposure)

    # REGIME 2: HIGH VOLATILITY
    high_bear_mask = high_regime & ~bull_trend
    high_bull_mask = high_regime & bull_trend
    
    high_exposure = float(high_vol_bear_exposure if allow_short else flat_exposure)
    positions[high_bear_mask] = high_exposure
    positions[high_bull_mask] = (base_exposure[high_bull_mask] * float(low_vol_leverage)).values

    # REGIME 3: MODERATE VOLATILITY
    # Only take moderate exposure if both short-term momentum and fast trend are safe
    mod_safe = moderate_regime & moderate_mask & fast_trend_safe
    mod_unsafe = moderate_regime & ~(moderate_mask & fast_trend_safe)
    
    positions[mod_safe] = (base_exposure[mod_safe] * float(moderate_exposure)).values
    positions[mod_unsafe] = float(flat_exposure)

    return positions


def build_volatility_surface(returns: pd.Series, window: int = 60,
                              lag_range: range = range(1, 31),
                              use_magnitude: bool = True) -> pd.DataFrame:
    """Compute rolling lag-kernel values for a range of lags.

    This surface is built from rolling lagged auto-covariance, not the
    composite signal max(AutoCov, VolFloor), so variation along the lag axis
    is preserved.

    Parameters
    ----------
    returns : pd.Series
        Time series of daily returns.
    window : int, optional
        Auto-covariance window W. Default 60.
    lag_range : range or list of int, optional
        Lags l to compute. Default range(1, 31).
    use_magnitude : bool, optional
        If True, use absolute auto-covariance magnitude. Default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with lags as columns and dates as rows.
    """
    surface = {}
    for lag in lag_range:
        series = _rolling_autocovariance(returns, window=window, lag=lag)
        if use_magnitude:
            series = series.abs()
        surface[lag] = series
    return pd.DataFrame(surface)
