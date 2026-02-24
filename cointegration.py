from itertools import combinations

import pandas as pd
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(prices, significance=0.05):
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
