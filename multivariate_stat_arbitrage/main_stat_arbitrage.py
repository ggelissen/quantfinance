import time
import pandas as pd
import matplotlib.pyplot as plt

from data_handler import download_data
from cointegration import find_cointegrated_baskets
from backtester import run_backtest_kelly, plot_results
from static_param_optim import optimize_params, optimize_params_bayes


def main(prices: pd.DataFrame, baskets: list, initial_capital: float, capital_fraction: float = 0.5,
         transaction_costs: float = 1.0, copula_generation: bool = True, max_drawdown_stop: float = -999.0,
         vol_targeting: bool = False, target_annual_vol: float = 0.15,
         vol_lookback: int = 20, min_exposure_mult: float = 0.25,
         max_exposure_mult: float = 2.0, sizing_method: str = "score",
         min_position_fraction: float = 0.05, max_position_fraction: float = 0.35,
         sizing_base_weight: float = 0.30) -> list:
    """Main function to run multivariate statistical arbitrage on baskets.
    
    Returns list of result dicts with performance metrics for each basket.
    """
    results_summary = []

    if not baskets:
        print("No cointegrated baskets found.")
        return results_summary
    else:
        for basket_info in baskets:
            basket_start = time.perf_counter()
            basket_tickers = basket_info["tickers"]
            basket_label = ", ".join(basket_tickers)
            print(
                f"Backtesting basket: [{basket_label}] "
                f"(rank={basket_info['rank']}, score={basket_info['score']:.2f})"
            )

            if BAYESIAN_OPTIMIZATION:
                optimized_params = optimize_params_bayes(prices, basket_tickers,
                                                         copula_generation=copula_generation,
                                                         vol_targeting=vol_targeting,
                                                         target_annual_vol=target_annual_vol,
                                                         vol_lookback=vol_lookback,
                                                         min_exposure_mult=min_exposure_mult,
                                                         max_exposure_mult=max_exposure_mult)
            else:
                optimized_params = optimize_params(prices, basket_tickers,
                                                   copula_generation=copula_generation,
                                                   vol_targeting=vol_targeting,
                                                   target_annual_vol=target_annual_vol,
                                                   vol_lookback=vol_lookback,
                                                   min_exposure_mult=min_exposure_mult,
                                                   max_exposure_mult=max_exposure_mult) 
            print(f"Optimized Parameters: Window={optimized_params['window']}, "
                  f"Entry Threshold={round(optimized_params['entry_threshold'], 2)}, "
                  f"Exit Threshold={round(optimized_params['exit_threshold'], 2)}")

            results = run_backtest_kelly(prices, basket_tickers, initial_capital=initial_capital,
                                   entry_threshold=optimized_params['entry_threshold'],
                                   exit_threshold=optimized_params['exit_threshold'],
                                   capital_fraction=capital_fraction,
                                   transaction_costs=transaction_costs, 
                                   copula_generation=copula_generation,
                                   max_drawdown_stop=max_drawdown_stop,
                                   vol_targeting=vol_targeting,
                                   target_annual_vol=target_annual_vol,
                                   vol_lookback=vol_lookback,
                                   min_exposure_mult=min_exposure_mult,
                                   max_exposure_mult=max_exposure_mult,
                                   sizing_method=sizing_method,
                                   min_position_fraction=min_position_fraction,
                                   max_position_fraction=max_position_fraction,
                                   sizing_base_weight=sizing_base_weight,
            )
            
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Annualized Vol: {results.get('annualized_vol', 0.0) * 100:.2f}%")
            print(f"Win Rate: {results.get('win_rate', 0.0) * 100:.1f}%")
            print(f"Position Fraction ({results.get('sizing_method', sizing_method)}): {results.get('position_fraction', results.get('kelly_fraction', 0.0)):.2%}")
            if vol_targeting:
                print(f"Avg Exposure Multiplier: {results.get('avg_exposure_mult', 1.0):.2f}x")
            plot_results(results, basket_tickers)
            basket_elapsed = time.perf_counter() - basket_start
            print(f"Basket runtime: {basket_elapsed:.2f} seconds\n")
            
            results_summary.append({
                'basket': basket_label,
                'tickers': basket_tickers,
                'rank': basket_info['rank'],
                'score': basket_info['score'],
                'return': results['total_return'],
                'drawdown': results['max_drawdown'],
                'sharpe': results['sharpe_ratio'],
                'ann_vol': results.get('annualized_vol', 0.0),
                'win_rate': results.get('win_rate', 0.0),
                'kelly': results.get('position_fraction', results.get('kelly_fraction', 0.0)),
                'avg_exposure': results.get('avg_exposure_mult', 1.0),
                'sizing_method': results.get('sizing_method', sizing_method),
            })

    return results_summary


if __name__ == "__main__":
    start_time = time.perf_counter()

    # ── Tickers & date range ──────────────────────────────────────────────────
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "DIS"]
    START_DATE = "2024-01-01"
    END_DATE = "2026-01-01"

    # ── Capital ───────────────────────────────────────────────────────────────
    INITIAL_CAPITAL = 10_000.0          # Starting portfolio value ($)
    CAPITAL_FRACTION = 0.40             # Base fraction of capital deployed per trade

    # ── Strategy mode ─────────────────────────────────────────────────────────
    BAYESIAN_OPTIMIZATION = True        # False → grid search
    COPULA_GENERATION = True            # False → z-score basket signals
    TRANSACTION_COST = 1.0              # Flat fee per trade per leg ($)

    # ── Basket selection ──────────────────────────────────────────────────────
    BASKET_MIN_SIZE = 3                 # Minimum tickers per basket
    BASKET_MAX_SIZE = 5                 # Maximum tickers per basket
    MAX_BASKETS = 5                     # Number of top baskets to backtest

    # ── Pre-screening filters ─────────────────────────────────────────────────
    MIN_RANK = 1                        # Skip baskets with Johansen rank < this
    MIN_COINTEGRATION_SCORE = 5.0       # Skip baskets with score < this (trace - crit value)

    # ── Risk management ──────────────────────────────────────────────────────
    MAX_DRAWDOWN_STOP = -15.0           # Force exit if drawdown hits this % (e.g., -15.0 = -15%)

    # ── Volatility targeting ────────────────────────────────────────────────
    VOL_TARGETING = True                # Dynamically scale exposure to target annualized vol
    TARGET_ANNUAL_VOL = 0.18            # 18% annualized volatility target
    VOL_LOOKBACK = 20                   # Rolling window for realized vol estimation (days)
    MIN_EXPOSURE_MULT = 0.50            # Min exposure scaling factor
    MAX_EXPOSURE_MULT = 3.00            # Max exposure scaling factor

    # ── Position sizing (replaces pure Kelly) ───────────────────────────────
    SIZING_METHOD = "score"             # "score", "kelly", or "fixed"
    MIN_POSITION_FRACTION = 0.10        # Lower bound on deployed capital fraction
    MAX_POSITION_FRACTION = 0.80        # Upper bound on deployed capital fraction
    SIZING_BASE_WEIGHT = 0.70           # Blend weight for base CAPITAL_FRACTION in score sizing

    # ── Johansen model ────────────────────────────────────────────────────────
    DET_ORDER = 0                       # 0 = no trend; 1 = constant; 2 = linear trend
    K_AR_DIFF = 1                       # Lag order for Johansen test

    plt.close("all")

    prices = download_data(TICKERS, START_DATE, END_DATE)
    baskets = find_cointegrated_baskets(
        prices,
        min_size=BASKET_MIN_SIZE,
        max_size=min(BASKET_MAX_SIZE, len(TICKERS)),
        det_order=DET_ORDER,
        k_ar_diff=K_AR_DIFF,
        min_rank=MIN_RANK,
        min_score=MIN_COINTEGRATION_SCORE,
        max_baskets=MAX_BASKETS,
    )

    results_summary = main(prices=prices, baskets=baskets, initial_capital=INITIAL_CAPITAL,
                           capital_fraction=CAPITAL_FRACTION, transaction_costs=TRANSACTION_COST,
                           copula_generation=COPULA_GENERATION, max_drawdown_stop=MAX_DRAWDOWN_STOP,
                           vol_targeting=VOL_TARGETING, target_annual_vol=TARGET_ANNUAL_VOL,
                           vol_lookback=VOL_LOOKBACK, min_exposure_mult=MIN_EXPOSURE_MULT,
                           max_exposure_mult=MAX_EXPOSURE_MULT, sizing_method=SIZING_METHOD,
                           min_position_fraction=MIN_POSITION_FRACTION,
                           max_position_fraction=MAX_POSITION_FRACTION,
                           sizing_base_weight=SIZING_BASE_WEIGHT)

    elapsed = time.perf_counter() - start_time

    # ── Results Summary ───────────────────────────────────────────────────────
    print("\n" + "="*120)
    print("BACKTEST SUMMARY")
    print("="*120)
    
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df[['basket', 'rank', 'score', 'return', 'drawdown', 'sharpe', 'ann_vol', 'win_rate', 'kelly', 'avg_exposure']]
        summary_df['return'] = summary_df['return'].apply(lambda x: f"{x:.2f}%")
        summary_df['drawdown'] = summary_df['drawdown'].apply(lambda x: f"{x:.2f}%")
        summary_df['sharpe'] = summary_df['sharpe'].apply(lambda x: f"{x:.2f}")
        summary_df['ann_vol'] = summary_df['ann_vol'].apply(lambda x: f"{x*100:.2f}%")
        summary_df['win_rate'] = summary_df['win_rate'].apply(lambda x: f"{x*100:.1f}%")
        summary_df['kelly'] = summary_df['kelly'].apply(lambda x: f"{x:.2%}")
        summary_df['avg_exposure'] = summary_df['avg_exposure'].apply(lambda x: f"{x:.2f}x")
        summary_df['score'] = summary_df['score'].apply(lambda x: f"{x:.2f}")
        
        print(summary_df.to_string(index=False))
        
        winners = [r for r in results_summary if r['return'] > 0]
        losers = [r for r in results_summary if r['return'] <= 0]
        print(f"\nWinners: {len(winners)} | Losers: {len(losers)}")
        if winners:
            avg_winner_return = sum(r['return'] for r in winners) / len(winners)
            avg_winner_dd = sum(r['drawdown'] for r in winners) / len(winners)
            print(f"  Avg Winner: {avg_winner_return:.2f}% return, {avg_winner_dd:.2f}% drawdown")
        if losers:
            avg_loser_return = sum(r['return'] for r in losers) / len(losers)
            print(f"  Avg Loser: {avg_loser_return:.2f}% return")
    else:
        print("No baskets survived pre-screening or backtesting.")

    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    print("="*120)