import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.stats import norm

def calculate_spread(prices: pd.DataFrame, ticker_a: str, ticker_b: str, window: int = 20) -> pd.DataFrame:
    """Calculate the spread between two assets and its rolling statistics.

    An OLS regression is run to find the hedge ratio between *ticker_a*
    (dependent) and *ticker_b* (independent).  The spread is defined as::

        spread = price_a - hedge_ratio * price_b

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    ticker_a : str
        Dependent asset ticker.
    ticker_b : str
        Independent asset ticker.
    window : int, optional
        Rolling window (in days) for mean and standard deviation. Default 20.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'spread', 'rolling_mean', 'rolling_std',
        'hedge_ratio'.
    """
    series_a = prices[ticker_a].dropna()
    series_b = prices[ticker_b].dropna()
    common_index = series_a.index.intersection(series_b.index)
    series_a = series_a.loc[common_index]
    series_b = series_b.loc[common_index]

    obs_mat = np.vstack([series_b, np.ones(series_b.shape)]).T[:, np.newaxis, :]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 1e-4
    )

    state_means = kf.filter(series_a.values)[0]

    hedge_ratio = pd.Series(state_means[:, 0], index=common_index).shift(1)
    intercept = pd.Series(state_means[:, 1], index=common_index).shift(1)

    adjusted_b = np.add(np.multiply(hedge_ratio, series_b), intercept)
    spread = np.subtract(series_a, adjusted_b)

    result = pd.DataFrame({
        "spread": spread,
        "rolling_mean": 0.0,
        "rolling_std": spread.rolling(window=window).std(),
        "hedge_ratio": hedge_ratio,
    })

    return result


def compute_zscore(spread_df: pd.DataFrame) -> pd.Series:
    """Compute the z-score of the spread.

    Parameters
    ----------
    spread_df : pd.DataFrame
        Output from :func:`calculate_spread`.

    Returns
    -------
    pd.Series
        Z-score series aligned with *spread_df*.
    """
    spread = spread_df["spread"].to_numpy()

    rolling_mean = spread_df["rolling_mean"].to_numpy()
    rolling_std = spread_df["rolling_std"].to_numpy()

    zscore = (spread - rolling_mean) / rolling_std

    return pd.Series(zscore, index=spread_df.index, name="zscore")


def generate_signals_linear(prices: pd.DataFrame, ticker_a: str, ticker_b: str, window: int = 20, 
                            entry_z: float = 2.0, exit_z: float = 0.0) -> pd.DataFrame:
    """Generate long, short, and exit trading signals based on z-score thresholds.

    Signals
    -------
    - **1**  (long)  : z-score < -entry_z  → buy spread (long *ticker_a*, short *ticker_b*)
    - **-1** (short) : z-score >  entry_z  → sell spread (short *ticker_a*, long *ticker_b*)
    - **0**  (exit)  : z-score crosses *exit_z* (passes through zero)

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    ticker_a : str
        Dependent asset ticker.
    ticker_b : str
        Independent asset ticker.
    window : int, optional
        Rolling window for spread statistics. Default 20.
    entry_z : float, optional
        Z-score threshold that triggers entry. Default 2.0.
    exit_z : float, optional
        Z-score level at which positions are closed. Default 0.0.

    Returns
    -------
    pd.DataFrame
        Original spread DataFrame with additional columns: 'zscore' and
        'signal' (1 = long, -1 = short, 0 = exit / no position).
    """
    spread_df = calculate_spread(prices, ticker_a, ticker_b, window=window)

    zscore = compute_zscore(spread_df)
    z_vals = zscore.to_numpy()

    signals = np.select([z_vals < np.negative(entry_z), z_vals > entry_z, np.abs(z_vals) <= exit_z],
                        [1, -1, 0],
                        default=np.nan,
    )

    spread_df["zscore"] = zscore
    raw_signals = pd.Series(signals, index=spread_df.index)
    spread_df["signal"] = raw_signals.ffill().fillna(0)

    return spread_df



def generate_signals_copula(prices: pd.DataFrame, ticker_a: str, ticker_b: str, window: int = 60, 
                            entry_prob: float = 0.05, exit_prob: float = 0.5) -> pd.DataFrame:
    """Generate long, short, and exit trading signals based on z-score thresholds.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    ticker_a : str
        Dependent asset ticker.
    ticker_b : str
        Independent asset ticker.
    window : int, optional
        Rolling window for spread statistics. Default 60.
    entry_prob : float, optional
        Probability threshold that triggers entry. Default 0.05.
    exit_prob : float, optional
        Probability level at which positions are closed. Default 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'signal' (1 = long, -1 = short, 0 = exit) and
        'hedge_ratio' for the spread.

    """
    returns_a = prices[ticker_a].pct_change().dropna()
    returns_b = prices[ticker_b].pct_change().dropna()
    common_index = returns_a.index.intersection(returns_b.index)
    ret_a = returns_a.loc[common_index]
    ret_b = returns_b.loc[common_index]

    u = ret_a.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    v = ret_b.rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    u = np.clip(u, 0.001, 0.999)
    v = np.clip(v, 0.001, 0.999)

    rho = ret_a.rolling(window=window).corr(ret_b)

    x = norm.ppf(u)
    y = norm.ppf(v)

    cond_prob_series = norm.cdf((x - rho * y) / np.sqrt(1 - rho**2))

    signals = np.full(len(common_index), np.nan)
    signals = np.where(cond_prob_series < entry_prob, 1, signals)
    signals = np.where(cond_prob_series > (1 - entry_prob), -1, signals)

    signals_series = pd.Series(signals, index=common_index).ffill().fillna(0)

    hr = np.divide(prices[ticker_a].loc[common_index], prices[ticker_b].loc[common_index])
    result = pd.DataFrame({'signal': signals_series, 'hedge_ratio': hr}, index=common_index)

    return result