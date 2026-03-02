from itertools import combinations
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def find_cointegrated_pairs(prices: pd.DataFrame, significance: float = 0.05) -> list:
    """Find cointegrated pairs in a price DataFrame using the Engle-Granger test.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    significance : float, optional
        P-value threshold for cointegration. Default is 0.05.

    Returns
    -------
    list of tuple
        Each tuple contains (ticker_a, ticker_b, p_value) for pairs whose
        Engle-Granger p-value is below the significance threshold.
    """
    cointegrated_pairs = []
    tickers = list(prices.columns)

    for ticker_a, ticker_b in combinations(tickers, 2):
        series_a = prices[ticker_a].dropna()
        series_b = prices[ticker_b].dropna()
        common_index = series_a.index.intersection(series_b.index)
        if len(common_index) < 30:
            continue
        _, p_value, _ = coint(series_a.loc[common_index], series_b.loc[common_index])
        if p_value < significance:
            cointegrated_pairs.append((ticker_a, ticker_b, p_value))

    return cointegrated_pairs


def calculate_johansen_weights(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.Series:
    """Calculate hedge ratio using Johansen's cointegration method.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    det_order : int, optional
        Deterministic trend order for the Johansen test. Default is 0 (no deterministic trend).
    k_ar_diff : int, optional
        Number of lagged differences to include in the Johansen test. Default is 1.

    Returns
    -------
    pd.Series
        Hedge ratio series for the spread.
    """
    prices_array = prices.dropna().to_numpy()
    result = coint_johansen(prices_array, det_order, k_ar_diff)
    eigenvectors = result.evec
    best_eigenvector = eigenvectors[:, 0]
    weights = best_eigenvector / best_eigenvector[0]
    weights_series = pd.Series(weights, index=prices.columns)
    return weights_series