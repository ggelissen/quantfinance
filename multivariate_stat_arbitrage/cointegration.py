from itertools import combinations
from typing import Any

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def find_cointegrated_baskets(
    prices: pd.DataFrame,
    min_size: int = 3,
    max_size: int = 5,
    det_order: int = 0,
    k_ar_diff: int = 1,
    max_baskets: int = 5,
    min_rank: int = 1,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Find top cointegrated baskets using Johansen trace statistics.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data for basket discovery.
    min_size : int
        Minimum basket size.
    max_size : int
        Maximum basket size.
    det_order : int
        Johansen deterministic order.
    k_ar_diff : int
        Johansen lag order.
    max_baskets : int
        Maximum baskets to return.
    min_rank : int
        Minimum Johansen rank (cointegration pairs) to include. Default 1.
    min_score : float
        Minimum cointegration score (trace statistic - critical value) to include. Default 0.0.
    """
    clean_prices = prices.dropna(how="all").dropna(axis=1, how="all")
    tickers = list(clean_prices.columns)
    if len(tickers) < min_size:
        return []

    max_size = min(max_size, len(tickers))
    results: list[dict[str, Any]] = []

    for basket_size in range(min_size, max_size + 1):
        for basket in combinations(tickers, basket_size):
            basket_prices = clean_prices[list(basket)].dropna(how="any")
            min_obs = max(60, basket_size * 20)
            if len(basket_prices) < min_obs:
                continue

            try:
                johansen = coint_johansen(basket_prices.to_numpy(), det_order, k_ar_diff)
            except Exception:
                continue

            rank = int((johansen.lr1 > johansen.cvt[:, 1]).sum())
            if rank < min_rank:
                continue

            score = float(johansen.lr1[0] - johansen.cvt[0, 1])
            if score < min_score:
                continue

            results.append(
                {
                    "tickers": list(basket),
                    "rank": rank,
                    "score": score,
                    "observations": int(len(basket_prices)),
                }
            )

    results.sort(key=lambda x: (x["score"], x["rank"], x["observations"]), reverse=True)
    return results[:max_baskets]