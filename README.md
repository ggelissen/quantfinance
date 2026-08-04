# Quantitative Finance Lab

Python research prototypes for statistical arbitrage, dependence-aware signals, volatility targeting, backtesting, and portfolio risk analysis.

The repository is a laboratory rather than a production trading system. Its purpose is to make modelling assumptions explicit, test ideas on historical data, and develop reusable research components.

> Historical backtests are educational research, not evidence of future profitability or investment advice. Results should be interpreted only after checking data timing, costs, parameter selection, and out-of-sample design.

## Projects

### Bivariate statistical arbitrage

- Pair discovery using cointegration tests
- Rolling spread estimation and z-score signals
- Kalman-filter and copula-aware signal components
- Position sizing, backtesting, and performance diagnostics
- Static-grid and Bayesian parameter-search utilities
- Multi-panel research dashboard

Entry point: `bivariate_stat_arbitrage/main_stat_arbitrage.py`

### Multivariate statistical arbitrage

- Basket discovery with the Johansen cointegration procedure
- Cointegration-weighted spread construction
- Gaussian and vine-copula joint-probability features
- Basket backtesting and risk-based sizing
- Static-grid and Bayesian parameter search

Entry point: `multivariate_stat_arbitrage/main_stat_arbitrage.py`

### Volatility-adjusted S&P 500 strategy

- Rolling lagged autocovariance and realised-volatility features
- EWMA target-volatility exposure
- Trend-conditioned volatility regimes
- Transaction-cost-aware backtesting
- Interactive Plotly and animated Matplotlib visualisations

See the [project README](volatility_adjusted_sp500_strategy/README.md) for the model definition and usage.

## Repository map

```text
bivariate_stat_arbitrage/            pair-trading research pipeline
multivariate_stat_arbitrage/         basket-trading research pipeline
volatility_adjusted_sp500_strategy/  volatility-regime strategy and visualisations
volatility_strategy.mp4              short strategy visualisation
```

## Installation

The current project does not yet include a locked dependency manifest. The modules use packages including:

```bash
python -m pip install numpy pandas scipy matplotlib seaborn plotly \
  statsmodels scikit-optimize pykalman pyvinecopulib yfinance opencv-python
```

Create and commit a versioned `pyproject.toml` or requirements lock before treating runs as reproducible.

## Research standards still to add

- Unit tests for return alignment, costs, drawdown, and position timing
- Train/validation/test separation for every parameter search
- A shared backtest interface across all projects
- Explicit survivorship-bias and data-source notes
- Reproducible example commands with frozen dates and configurations
- Genuine result tables generated from saved run metadata

## Licence

Distributed under the [MIT License](LICENSE).
