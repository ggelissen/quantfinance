import time
import pandas as pd
import matplotlib.pyplot as plt

from data_handler import download_data
from cointegration import find_cointegrated_pairs
from backtester import run_backtest_kelly, plot_results
from static_param_optim import optimize_params, optimize_params_bayes


def main(prices: pd.DataFrame, pairs: list, initial_capital: float, capital_fraction: float = 0.5,
         transaction_costs: float = 1.0, copula_generation: bool = True) -> None:
    """Main function to run the statistical arbitrage strategy."""

    if not pairs:
        print("No cointegrated pairs found.")
        return None
    else:
        for ticker_a, ticker_b, p_value in pairs:
            pair_start = time.perf_counter()
            print(f"Backtesting pair: {ticker_a} & {ticker_b} (p-value: {p_value:.4f})")

            if BAYESIAN_OPTIMIZATION:
                optimized_params = optimize_params_bayes(prices, ticker_a, ticker_b, 
                                                         copula_generation=copula_generation)
            else:
                optimized_params = optimize_params(prices, ticker_a, ticker_b,
                                                   copula_generation=copula_generation) 
            print(f"Optimized Parameters: Window={optimized_params['window']}, "
                  f"Entry Threshold={round(optimized_params['entry_threshold'], 2)}, "
                  f"Exit Threshold={round(optimized_params['exit_threshold'], 2)}")

            results = run_backtest_kelly(prices, ticker_a, ticker_b, initial_capital=initial_capital,
                                   entry_threshold=optimized_params['entry_threshold'],
                                   exit_threshold=optimized_params['exit_threshold'],
                                   capital_fraction=capital_fraction,
                                   transaction_costs=transaction_costs, 
                                   copula_generation=copula_generation
            )
            
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            plot_results(results, ticker_a, ticker_b)
            pair_elapsed = time.perf_counter() - pair_start
            print(f"Pair runtime: {pair_elapsed:.2f} seconds\n")

    return None


if __name__ == "__main__":
    start_time = time.perf_counter()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "DIS"]
    start_date = "2024-01-01"
    end_date = "2026-01-01"
    initial_capital = 10_000.0 
    capital_fraction = 0.50

    BAYESIAN_OPTIMIZATION = True  # Set to False to use grid search instead
    TRANSACTION_COST = 1.0  # flat fee per trade in dollars
    COPULA_GENERATION = True  # Set to False to use linear regression signals instead

    plt.close("all")

    prices = download_data(tickers, start_date, end_date)
    pairs = find_cointegrated_pairs(prices)

    main(prices=prices, pairs=pairs, initial_capital=initial_capital, capital_fraction=capital_fraction,
         transaction_costs=TRANSACTION_COST, copula_generation=COPULA_GENERATION)

    elapsed = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed:.2f} seconds")