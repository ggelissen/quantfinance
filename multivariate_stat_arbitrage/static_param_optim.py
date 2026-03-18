import pandas as pd
import numpy as np
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real, Integer

from backtester import run_backtest


def optimize_params(prices: pd.DataFrame, basket_tickers: list[str], windows: list = [30, 60, 90],
                    entry_thresholds: list = [0.02, 0.05, 0.08], exit_thresholds: list = [0.3, 0.5, 0.7],
                    copula_generation: bool = True, vol_targeting: bool = False,
                    target_annual_vol: float = 0.15, vol_lookback: int = 20,
                    min_exposure_mult: float = 0.25, max_exposure_mult: float = 2.0) -> dict:
    """Run basket backtests for different parameter combinations and find the best one.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    basket_tickers : list[str]
        Tickers included in the basket.
    windows : list of int, optional
        List of rolling window sizes to test. Default is [10, 20, 30].
    entry_thresholds : list of float, optional
        List of z-score entry thresholds to test. Default is [1.0, 1.5, 2.0].
    exit_thresholds : list of float, optional
        List of z-score exit thresholds to test. Default is [0.5, 1.0, 1.5].
    copula_generation : bool, optional
        If True, use copula-based signal generation. If False, use linear regression signals. Default True.

    Returns
    -------
    dict
        Dictionary with keys 'window', 'entry_threshold', and 'exit_threshold' 
        for the best parameters.
    """
    best_sharpe = -np.inf
    best_params = {"window": None, "entry_threshold": None, "exit_threshold": None}

    for window in windows:
        for entry in entry_thresholds:
            for exit in exit_thresholds:
                results = run_backtest(prices, basket_tickers, copula_generation=copula_generation,
                                       window=window, entry_threshold=entry, exit_threshold=exit,
                                       vol_targeting=vol_targeting, target_annual_vol=target_annual_vol,
                                       vol_lookback=vol_lookback, min_exposure_mult=min_exposure_mult,
                                       max_exposure_mult=max_exposure_mult)
                if results["sharpe_ratio"] > best_sharpe:
                    best_sharpe = results["sharpe_ratio"]
                    best_params["window"] = window
                    best_params["entry_threshold"] = entry
                    best_params["exit_threshold"] = exit

    return best_params



def optimize_params_bayes(prices: pd.DataFrame, basket_tickers: list[str],
                          copula_generation: bool = True, vol_targeting: bool = False,
                          target_annual_vol: float = 0.15, vol_lookback: int = 20,
                          min_exposure_mult: float = 0.25, max_exposure_mult: float = 2.0) -> dict:
    """Bayesian optimization of backtest parameters.

    This function implements a sophisticated optimization approach 
    using Bayesian methods to maximize the Sharpe ratio.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (columns = tickers).
    basket_tickers : list[str]
        Tickers included in the basket.
    copula_generation : bool, optional  
        If True, use copula-based signal generation. If False, use linear regression signals. Default True.
    Returns
    -------
    dict
        Dictionary with keys 'window', 'entry_threshold', and 
        'exit_threshold' for the best parameters.
    """

    basket_prices = prices[basket_tickers].dropna(how="any")
    max_window = int(len(basket_prices) / 4)
    if copula_generation:
        coarse_space = [Integer(20, max_window, name='window'),
                        Real(0.01, 0.1, name='entry_threshold'),
                        Real(0.4, 0.6, name='exit_threshold')
        ]
    else:  
        coarse_space = [Integer(10, max_window, name='window'),
                        Real(1.0, 3.0, name='entry_threshold'),
                        Real(0.0, 2.0, name='exit_threshold')
        ]

    def objective(params):
        window, entry_threshold, exit_threshold = params
        results = run_backtest(prices, basket_tickers, copula_generation=copula_generation,
                               window=window, entry_threshold=entry_threshold, 
                               exit_threshold=exit_threshold, vol_targeting=vol_targeting,
                               target_annual_vol=target_annual_vol, vol_lookback=vol_lookback,
                               min_exposure_mult=min_exposure_mult, max_exposure_mult=max_exposure_mult)
        sharpe = results["sharpe_ratio"]
        
        if np.isnan(sharpe):
            return 0.0
        return np.negative(sharpe)
    
    coarse_result = dummy_minimize(func=objective, dimensions=coarse_space, n_calls=10, random_state=42)

    best_w = int(coarse_result.x[0])
    best_entry = float(coarse_result.x[1])
    best_exit = float(coarse_result.x[2])

    if copula_generation:
        fine_space = [Integer(max(20, best_w - 10), min(max_window, best_w + 10), name='window'),
                      Real(max(0.01, best_entry - 0.02), min(0.1, best_entry + 0.02), name='entry_threshold'),
                      Real(max(0.4, best_exit - 0.05), min(0.6, best_exit + 0.05), name='exit_threshold')
        ]
    else:
        fine_space = [Integer(max(10, best_w - 10), min(max_window, best_w + 10), name='window'),
                    Real(max(1.0, best_entry - 0.5), min(3.0, best_entry + 0.5), name='entry_threshold'),
                    Real(max(0.0, best_exit - 0.5), min(2.0, best_exit + 0.5), name='exit_threshold')
        ]
    
    fine_result = gp_minimize(func=objective, dimensions=fine_space,
                         n_calls=20, random_state=42)
    
    best_params = {"window": int(fine_result.x[0]), 
                   "entry_threshold": float(fine_result.x[1]), 
                   "exit_threshold": float(fine_result.x[2])
    }

    return best_params