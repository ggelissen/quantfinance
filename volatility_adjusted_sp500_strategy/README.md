# Volatility-Adjusted S&P 500 Strategy

A Python project that implements and visualises a volatility-regime-based trading strategy on the S&P 500.

---

## Overview

### The Model

The composite signal is defined as:

$$
S(t,l) = \max \left( \frac{1}{W-l} \sum_{i=0}^{W-l-1} (r_{t-i}-\bar{r}_{t,W})(r_{t-i-l}-\bar{r}_{t-l,W}), \text{Quantile}_q \left( \left\lbrace \frac{1}{W} \sum_{j=0}^{W-1} r^2_{t-k-j} \right\rbrace_{k=0}^{K} \right) \right)
$$

**Part 1 – Lagged Auto-Covariance:** measures whether past returns predict current returns (serial correlation) over a rolling window *W* at lag *l*.

**Part 2 – Volatility Floor:** a quantile of *K* rolling realised variances (each computed over a sub-window *w*) that prevents the signal from dropping below a minimum baseline risk level.

### Position Rule

| Signal S(t, l)    | Position |
|-------------------|----------|
| < -`vol_threshold` | Long (+1) |
| > +`vol_threshold` | Short (−1) if `allow_short=True`, else Flat (0) |
| otherwise          | Flat (0) |

> Low / mean-reverting volatility → Long.  
> Volatility spike → exit or short.

### Visualisations (Dark Mode)

1. **3D Interactive Volatility Surface** — `S(t, l)` plotted over time (x), lag (y), and signal magnitude (z) using Plotly with a *Plasma* colour scale.
2. **Equity Curve Dashboard** — three-panel dark-mode chart: S&P 500 price, portfolio value with entry markers, and position bars.
3. **Rolling Volatility Regimes** — annualised 21-day, 63-day, and 126-day rolling volatility.

---

## Tech Stack

| Category | Library |
|----------|---------|
| Data download | `yfinance` |
| Numerical computation | `numpy`, `pandas` |
| Interactive 3D charts | `plotly` |
| 2D / animated charts | `matplotlib` |
| (Optional) risk models | `scikit-learn` |

All libraries are free and open-source.

---

## Installation

```bash
pip install yfinance numpy pandas plotly matplotlib scikit-learn
```

---

## Usage

```bash
cd volatility_adjusted_sp500_strategy
python main.py
```

All configuration (dates, signal parameters, capital) is set inside `main.py`:

```python
main(
    start_date="2010-01-01",
    end_date="2026-01-01",
    ticker="^GSPC",        # S&P 500
    window=60,             # auto-covariance window W
    lag=1,                 # lag l
    sub_window=21,         # realised-variance sub-window w
    k_windows=10,          # number of sub-windows K
    quantile=0.25,         # volatility-floor quantile q
    vol_threshold=0.0,     # signal entry threshold
    allow_short=False,     # allow short positions
    initial_capital=10_000.0,
    transaction_cost=1.0,
    show_plots=True,
    save_html=False,       # set True to save HTML files
)
```

### Save charts to HTML

```python
main(..., save_html=True)
```

This creates three standalone HTML files: `volatility_surface.html`, `equity_curve.html`, `rolling_volatility.html`.

---

## File Structure

```
volatility_adjusted_sp500_strategy/
├── data_handler.py      # Download S&P 500 data via yfinance
├── strategy.py          # Compute S(t,l) signal and positions
├── backtester.py        # Daily back-test engine & performance metrics
├── visualization.py     # Plotly dark-mode charts (3D surface + equity dashboard)
├── main.py              # Entry point — ties all modules together
└── README.md            # This file
```

---

## Performance Metrics

The back-tester reports:

- Total Return (%)
- Annualised Return (%)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown (%)
- Win Rate (%)
- Profit Factor
