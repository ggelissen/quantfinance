"""Visualization utilities for the volatility-adjusted S&P 500 strategy.

All charts use a dark-mode aesthetic (black background, neon accent colours)
that matches the "Quant Research Decoded" style described in the brief.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - optional dependency fallback
    gaussian_filter = None


# ---------------------------------------------------------------------------
# Dark-mode colour palette
# ---------------------------------------------------------------------------
BG_COLOR = "#000000"
GRID_COLOR = "#1a1a2e"
ACCENT_CYAN = "#00ffff"
ACCENT_MAGENTA = "#ff00ff"
ACCENT_GREEN = "#39ff14"
ACCENT_RED = "#ff3131"
ACCENT_YELLOW = "#ffff00"
TEXT_COLOR = "#e0e0e0"

_DARK_LAYOUT = dict(
    plot_bgcolor=BG_COLOR,
    paper_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, family="Courier New, monospace"),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showline=True,
               linecolor=ACCENT_CYAN),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showline=True,
               linecolor=ACCENT_CYAN),
)


def plot_volatility_surface(surface_df: pd.DataFrame,
                             title: str = "Volatility Surface — S(t, l)",
                             colorscale: str = "Plasma",
                             last_n_days: int = 252,
                             smoothing_sigma: float = 1.2,
                             color_low_percentile: float = 5.0,
                             color_high_percentile: float = 95.0) -> go.Figure:
    """Render an interactive 3D volatility surface.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of :func:`strategy.build_volatility_surface`. Rows = dates,
        columns = lags (integers).
    title : str, optional
        Figure title. Default 'Volatility Surface — S(t, l)'.
    colorscale : str, optional
        Plotly colour scale name. Default 'Plasma'.
    last_n_days : int, optional
        How many of the most-recent trading days to show. Default 252 (1 year).
    smoothing_sigma : float, optional
        Gaussian smoothing strength applied to the Z matrix before plotting.
        Set to 0 to disable. Default 1.2.
    color_low_percentile : float, optional
        Lower percentile for dynamic color-scale clamping. Default 5.
    color_high_percentile : float, optional
        Upper percentile for dynamic color-scale clamping. Default 95.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D surface figure.
    """
    df = surface_df.dropna().iloc[-last_n_days:]

    lags = np.array(df.columns, dtype=float)
    dates = df.index
    z_matrix = df.values.T  # shape: (n_lags, n_dates)

    if smoothing_sigma > 0 and gaussian_filter is not None:
        z_plot = gaussian_filter(z_matrix, sigma=float(smoothing_sigma))
    else:
        z_plot = z_matrix

    finite = z_plot[np.isfinite(z_plot)]
    if finite.size > 0:
        cmin = float(np.percentile(finite, color_low_percentile))
        cmax = float(np.percentile(finite, color_high_percentile))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            cmin = float(np.min(finite))
            cmax = float(np.max(finite))
    else:
        cmin, cmax = 0.0, 1.0

    # Map dates to numeric indices for the 3D axis
    x_indices = np.arange(len(dates))
    date_labels = [str(d)[:10] for d in dates]

    fig = go.Figure(data=[
        go.Surface(
            x=x_indices,
            y=lags,
            z=z_plot,
            colorscale=colorscale,
            opacity=0.92,
            showscale=True,
            cmin=cmin,
            cmax=cmax,
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False),
            ),
            lighting=dict(
                ambient=0.6,
                diffuse=0.5,
                roughness=0.5,
                specular=0.6,
                fresnel=0.2,
            ),
            lightposition=dict(x=120, y=180, z=80),
            colorbar=dict(
                title=dict(text="S(t,l)", font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TEXT_COLOR),
                bgcolor=BG_COLOR,
                bordercolor=ACCENT_CYAN,
            ),
        )
    ])

    # Sample tick positions for date axis
    n_ticks = min(8, len(dates))
    tick_step = max(1, len(dates) // n_ticks)
    tick_vals = list(range(0, len(dates), tick_step))
    tick_text = [date_labels[i] for i in tick_vals]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=ACCENT_CYAN)),
        scene=dict(
            bgcolor=BG_COLOR,
            xaxis=dict(
                title="Date",
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=9, color=TEXT_COLOR),
                gridcolor=GRID_COLOR,
                backgroundcolor=BG_COLOR,
                showbackground=True,
            ),
            yaxis=dict(
                title="Lag (days)",
                tickfont=dict(color=TEXT_COLOR),
                gridcolor=GRID_COLOR,
                backgroundcolor=BG_COLOR,
                showbackground=True,
            ),
            zaxis=dict(
                title="S(t, l)",
                tickfont=dict(color=TEXT_COLOR),
                gridcolor=GRID_COLOR,
                backgroundcolor=BG_COLOR,
                showbackground=True,
            ),
        ),
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Courier New, monospace"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=650,
    )
    return fig


def plot_equity_and_signals(portfolio: pd.DataFrame, data: pd.DataFrame,
                             ticker: str = "S&P 500",
                             title: str = "Volatility-Adjusted S&P 500 Strategy") -> go.Figure:
    """Render a dark-mode dashboard with equity curve, position, and index.

    Parameters
    ----------
    portfolio : pd.DataFrame
        Output ``portfolio`` key from :func:`backtester.run_backtest`. Must
        contain columns ``portfolio_value`` and ``position``.
    data : pd.DataFrame
        Raw OHLCV data with column ``close``.
    ticker : str, optional
        Display name for the index. Default 'S&P 500'.
    title : str, optional
        Overall figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Multi-panel interactive figure.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"{ticker} Close Price",
            "Strategy Equity Curve ($)",
            "Position (+1 Long / -1 Short / 0 Flat)",
        ),
        row_heights=[0.35, 0.45, 0.20],
    )

    # ---- Row 1: index close price ----------------------------------------
    close = data["close"].reindex(portfolio.index)
    fig.add_trace(
        go.Scatter(
            x=close.index, y=close.values,
            mode="lines",
            name=ticker,
            line=dict(color=ACCENT_CYAN, width=1.5),
        ),
        row=1, col=1,
    )

    # ---- Row 2: equity curve with long/short/flat shading ----------------
    equity = portfolio["portfolio_value"]
    position = portfolio["position"]

    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color=ACCENT_MAGENTA, width=2),
            fill="tozeroy",
            fillcolor="rgba(255,0,255,0.07)",
        ),
        row=2, col=1,
    )

    # Mark long entry points
    long_mask = (position > 0) & (position.shift(1).fillna(0.0) <= 0)
    long_entries = equity[long_mask]
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries.index, y=long_entries.values,
                mode="markers",
                name="Long Entry",
                marker=dict(symbol="triangle-up", color=ACCENT_GREEN, size=8,
                            line=dict(color="darkgreen", width=1)),
            ),
            row=2, col=1,
        )

    # Mark short entry points
    short_mask = (position < 0) & (position.shift(1).fillna(0.0) >= 0)
    short_entries = equity[short_mask]
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries.index, y=short_entries.values,
                mode="markers",
                name="Short Entry",
                marker=dict(symbol="triangle-down", color=ACCENT_RED, size=8,
                            line=dict(color="darkred", width=1)),
            ),
            row=2, col=1,
        )

    # ---- Row 3: position bar chart ---------------------------------------
    pos_colors = [
        ACCENT_GREEN if p == 1 else (ACCENT_RED if p == -1 else ACCENT_YELLOW)
        for p in position.values
    ]
    fig.add_trace(
        go.Bar(
            x=position.index, y=position.values,
            name="Position",
            marker_color=pos_colors,
            showlegend=False,
        ),
        row=3, col=1,
    )

    # ---- Global layout ---------------------------------------------------
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=ACCENT_CYAN)),
        height=850,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Courier New, monospace"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor=ACCENT_CYAN,
            borderwidth=1,
            font=dict(color=TEXT_COLOR),
        ),
        hovermode="x unified",
    )

    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            showline=True, linecolor=ACCENT_CYAN, row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            showline=True, linecolor=ACCENT_CYAN, row=i, col=1,
        )

    # Subtitle annotations for panels
    for annotation in fig.layout.annotations:
        annotation.font.color = ACCENT_CYAN
        annotation.font.size = 12

    return fig


def plot_rolling_volatility(returns: pd.Series, windows: list = None,
                             title: str = "Rolling Volatility Regimes") -> go.Figure:
    """Plot multiple rolling volatility bands on a dark-mode chart.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    windows : list of int, optional
        Rolling window sizes (in days). Default [21, 63, 126].
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if windows is None:
        windows = [21, 63, 126]

    colours = [ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_YELLOW]
    annualized = np.sqrt(252)

    fig = go.Figure()

    for window, colour in zip(windows, colours):
        vol = returns.rolling(window).std() * annualized * 100
        fig.add_trace(
            go.Scatter(
                x=vol.index, y=vol.values,
                mode="lines",
                name=f"{window}d Vol (%)",
                line=dict(color=colour, width=1.5),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=ACCENT_CYAN)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   showline=True, linecolor=ACCENT_CYAN),
        yaxis=dict(title="Annualised Vol (%)", gridcolor=GRID_COLOR,
                   zerolinecolor=GRID_COLOR, showline=True, linecolor=ACCENT_CYAN),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Courier New, monospace"),
        legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor=ACCENT_CYAN,
                    borderwidth=1, font=dict(color=TEXT_COLOR)),
        height=450,
    )
    return fig
