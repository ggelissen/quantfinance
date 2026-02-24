import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_spread(prices, ticker_a, ticker_b, window=20):
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

    model = sm.OLS(series_a, sm.add_constant(series_b)).fit()
    hedge_ratio = model.params[ticker_b]

    spread = series_a - hedge_ratio * series_b

    result = pd.DataFrame({
        "spread": spread,
        "rolling_mean": spread.rolling(window=window).mean(),
        "rolling_std": spread.rolling(window=window).std(),
        "hedge_ratio": hedge_ratio,
    })
    return result


def compute_zscore(spread_df):
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
    zscore = (spread_df["spread"] - spread_df["rolling_mean"]) / spread_df["rolling_std"]
    return zscore


def generate_signals(prices, ticker_a, ticker_b, window=20, entry_z=2.0, exit_z=0.0):
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
    spread_df["zscore"] = compute_zscore(spread_df)

    signals = pd.Series(np.nan, index=spread_df.index)
    signals[spread_df["zscore"] < -entry_z] = 1
    signals[spread_df["zscore"] > entry_z] = -1
    signals[spread_df["zscore"].abs() <= exit_z] = 0

    spread_df["signal"] = signals
    return spread_df
