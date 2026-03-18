import importlib

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

try:
    pv = importlib.import_module("pyvinecopulib")
except Exception:
    pv = None


def calculate_johansen_weights(prices_df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.Series:
    """Return normalized Johansen eigenvector weights for a basket."""
    clean_prices = prices_df.dropna(how="any")
    if clean_prices.shape[1] < 3:
        raise ValueError("Johansen basket model requires at least 3 tickers.")
    if len(clean_prices) < 30:
        raise ValueError("Not enough observations for Johansen test.")

    result = coint_johansen(clean_prices.to_numpy(), det_order, k_ar_diff)
    best_eigenvector = result.evec[:, 0]

    if np.isclose(best_eigenvector[0], 0.0):
        non_zero_idx = int(np.argmax(np.abs(best_eigenvector)))
        denom = best_eigenvector[non_zero_idx]
    else:
        denom = best_eigenvector[0]

    weights = best_eigenvector / denom
    return pd.Series(weights, index=clean_prices.columns)


def _basket_spread_components(prices_df: pd.DataFrame, window: int, det_order: int, k_ar_diff: int):
    clean_prices = prices_df.dropna(how="any")
    weights = calculate_johansen_weights(clean_prices, det_order=det_order, k_ar_diff=k_ar_diff)
    spread = clean_prices.mul(weights, axis=1).sum(axis=1)
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std().replace(0.0, np.nan)
    zscore = (spread - rolling_mean) / rolling_std
    returns = clean_prices.pct_change()
    return clean_prices, weights, spread, zscore, returns


def _pseudo_observations(returns_window: pd.DataFrame) -> np.ndarray:
    ranked = returns_window.rank(axis=0, pct=True)
    eps = 1e-4
    return np.clip(ranked.to_numpy(), eps, 1.0 - eps)


def _approx_joint_prob_gaussian(u_window: np.ndarray, random_state: int = 42) -> float:
    d = u_window.shape[1]
    if d < 2:
        return 0.5

    z_window = norm.ppf(u_window)
    z_hist = z_window[:-1, :]
    z_last = z_window[-1, :]
    if z_hist.shape[0] < 10:
        return 0.5

    cov = np.corrcoef(z_hist, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2.0
    cov += np.eye(d) * 1e-6

    rng = np.random.default_rng(random_state)
    sims = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=2000)
    joint_prob = np.mean(np.all(sims <= z_last, axis=1))
    return float(np.clip(joint_prob, 0.0, 1.0))


def _joint_probability_vine(returns_window: pd.DataFrame, random_state: int = 42) -> float:
    u_window = _pseudo_observations(returns_window)
    if u_window.shape[0] < 15:
        return 0.5

    if pv is not None:
        try:
            train_u = u_window[:-1, :]
            last_u = u_window[-1:, :]
            model = pv.Vinecop.from_data(train_u)
            vine_cdf = model.cdf(last_u)
            return float(np.clip(vine_cdf[0], 0.0, 1.0))
        except Exception:
            pass

    return _approx_joint_prob_gaussian(u_window, random_state=random_state)


def generate_signals_basket(prices_df: pd.DataFrame, window: int = 60, entry_z: float = 2.0,
                            exit_z: float = 0.5, det_order: int = 0, k_ar_diff: int = 1) -> pd.DataFrame:
    """Generate multivariate basket signals from Johansen spread z-score."""
    clean_prices, weights, spread, zscore, _ = _basket_spread_components(
        prices_df, window=window, det_order=det_order, k_ar_diff=k_ar_diff
    )

    z_vals = zscore.to_numpy()
    signals = np.full(len(clean_prices), np.nan)
    signals = np.where(z_vals < -entry_z, 1, signals)
    signals = np.where(z_vals > entry_z, -1, signals)
    signals = np.where(np.abs(z_vals) <= exit_z, 0, signals)

    signals_series = pd.Series(signals, index=clean_prices.index).ffill().fillna(0)
    result = pd.DataFrame({"spread": spread, "zscore": zscore, "signal": signals_series})

    for ticker in clean_prices.columns:
        result[f"weight_{ticker}"] = float(weights[ticker])

    return result


def generate_signals_basket_vine_copula(
    prices_df: pd.DataFrame,
    window: int = 60,
    entry_prob: float = 0.05,
    exit_prob: float = 0.5,
    exit_z: float = 0.5,
    exit_tolerance: float = 0.05,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.DataFrame:
    """Generate basket signals using a rolling vine copula + spread filter."""
    clean_prices, weights, spread, zscore, returns = _basket_spread_components(
        prices_df, window=window, det_order=det_order, k_ar_diff=k_ar_diff
    )

    joint_prob = pd.Series(np.nan, index=clean_prices.index, dtype=float)
    min_obs = max(30, clean_prices.shape[1] * 8)

    for i in range(window, len(clean_prices)):
        rolling_returns = returns.iloc[i - window + 1: i + 1].dropna(how="any")
        if len(rolling_returns) < min_obs:
            continue
        joint_prob.iloc[i] = _joint_probability_vine(rolling_returns, random_state=42 + i)

    z_vals = zscore.to_numpy()
    p_vals = joint_prob.to_numpy()
    signals = np.full(len(clean_prices), np.nan)

    long_entries = (p_vals < entry_prob) & (z_vals < 0)
    short_entries = (p_vals > (1.0 - entry_prob)) & (z_vals > 0)
    exits = (np.abs(z_vals) <= exit_z) | (np.abs(p_vals - exit_prob) <= exit_tolerance)

    signals = np.where(long_entries, 1, signals)
    signals = np.where(short_entries, -1, signals)
    signals = np.where(exits, 0, signals)

    signals_series = pd.Series(signals, index=clean_prices.index).ffill().fillna(0)
    result = pd.DataFrame(
        {
            "spread": spread,
            "zscore": zscore,
            "joint_probability": joint_prob,
            "signal": signals_series,
        }
    )

    for ticker in clean_prices.columns:
        result[f"weight_{ticker}"] = float(weights[ticker])

    return result
