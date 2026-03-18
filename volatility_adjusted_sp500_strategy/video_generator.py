"""Cyberpunk strategy video generator for the volatility-adjusted S&P 500 strategy.

Renders a 9:16 (vertical / mobile) MP4 animation that combines:

* **Upper pane** – 3D volatility surface (matplotlib Axes3D, Plasma colormap).
* **Mid section** – Dynamic data readout: date, strategy value, current position.
* **Lower pane** – Portfolio backtest timeline: strategy equity curve (neon green)
  and S&P 500 buy-and-hold (neon orange), with shaded high-volatility regime bars.

Frames are assembled with OpenCV (``cv2``).

Usage
-----
    from video_generator import generate_video
    generate_video(surface_df, portfolio, data)
"""

import io
import os
import subprocess
import sys

import cv2
import matplotlib
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Colour palette  (Neon/Cyberpunk)
# ---------------------------------------------------------------------------
_BG = "#000000"
_GREEN = "#39FF14"   # Neon Green  – strategy line
_ORANGE = "#FF5F1F"  # Neon Orange – S&P 500 line / regime bars
_MAGENTA = "#FF00FF"  # Accent Magenta – floating marker
_CYAN = "#00FFFF"    # Accent Cyan – axis spines
_RED = "#FF3131"     # Accent Red  – BEAR label
_TEXT = "#E0E0E0"    # Light grey text

# Video defaults
_FPS = 24
_FRAME_STEP = 5          # trading days between consecutive frames
_SURFACE_WINDOW = 120    # rolling days shown on the 3D surface
_W_PX, _H_PX = 720, 1280  # 9:16 frame size in pixels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_video(
    surface_df: pd.DataFrame,
    portfolio: pd.DataFrame,
    data: pd.DataFrame,
    output_path: str = "volatility_strategy.mp4",
    fps: int = _FPS,
    frame_step: int = _FRAME_STEP,
    surface_window: int = _SURFACE_WINDOW,
    show_video: bool = True,
) -> str:
    """Render and save the cyberpunk strategy animation.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Volatility surface from :func:`strategy.build_volatility_surface`.
        Rows = dates, columns = integer lags.
    portfolio : pd.DataFrame
        Back-test portfolio with columns ``portfolio_value`` and ``position``.
    data : pd.DataFrame
        Raw OHLCV data with column ``close``.
    output_path : str, optional
        Destination file path for the MP4.  Default ``'volatility_strategy.mp4'``.
    fps : int, optional
        Frames per second.  Default 24.
    frame_step : int, optional
        Number of trading days to advance per frame.  Default 5 (~1 week).
        Increase (e.g. 20) for a faster preview render.
    surface_window : int, optional
        Number of most-recent trading days shown on the rolling 3D surface.
        Default 120.
    show_video : bool, optional
        Attempt to open the saved video with the system default media player
        after rendering.  Default True.

    Returns
    -------
    str
        Absolute path to the saved MP4 file.
    """
    # ------------------------------------------------------------------
    # Align all series to the valid (non-NaN) surface dates
    # ------------------------------------------------------------------
    common = surface_df.dropna().index.intersection(portfolio.index)
    surf = surface_df.loc[common]
    port = portfolio.reindex(common)
    close = data["close"].reindex(common)

    lags = np.array(surf.columns, dtype=float)
    dates = surf.index
    n = len(dates)

    if n == 0:
        raise ValueError("No overlapping valid dates between surface_df and portfolio.")

    # Buy-and-hold reference normalised to the same starting capital
    initial_capital = float(port["portfolio_value"].iloc[0])
    bh_values = close / float(close.iloc[0]) * initial_capital

    # Global Z bounds for a consistent colour scale across all frames
    z_all = surf.values
    z_min = float(np.nanmin(z_all))
    z_max = float(np.nanmax(z_all))
    if z_min == z_max:
        z_max = z_min + 1e-6

    # Frame indices: start after the surface warm-up window
    frame_indices = list(range(surface_window, n, frame_step))
    if not frame_indices:
        frame_indices = [n - 1]

    # ------------------------------------------------------------------
    # OpenCV video writer
    # ------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (_W_PX, _H_PX))

    print(
        f"Rendering {len(frame_indices)} frames "
        f"({frame_step} day/frame, {fps} fps) → {output_path}"
    )

    for frame_num, t_idx in enumerate(frame_indices):
        if frame_num % 50 == 0:
            print(f"  Frame {frame_num}/{len(frame_indices)} …")

        img = _render_frame(
            surf=surf,
            port=port,
            bh_values=bh_values,
            dates=dates,
            lags=lags,
            t_idx=t_idx,
            surface_window=surface_window,
            initial_capital=initial_capital,
            z_min=z_min,
            z_max=z_max,
        )
        # Ensure exact pixel dimensions
        img = cv2.resize(img, (_W_PX, _H_PX))
        writer.write(img)

    writer.release()
    abs_path = os.path.abspath(output_path)
    print(f"Video saved: {abs_path}")

    if show_video:
        _try_open_video(abs_path)

    return abs_path


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def _render_frame(
    surf: pd.DataFrame,
    port: pd.DataFrame,
    bh_values: pd.Series,
    dates,
    lags: np.ndarray,
    t_idx: int,
    surface_window: int,
    initial_capital: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Render a single animation frame and return a BGR numpy array."""
    # ---------------------------------------------------------------
    # Figure: 9:16 ratio at 100 DPI → 720 × 1280 pixels
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 12.8), dpi=100, facecolor=_BG,
                     constrained_layout=True)
    fig.patch.set_facecolor(_BG)

    # GridSpec: [3D surface | readout | equity chart]
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[5, 1.2, 4],
        hspace=0.06,
    )

    # ---------------------------------------------------------------
    # Upper pane – 3D volatility surface
    # ---------------------------------------------------------------
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.set_facecolor(_BG)
    ax3d.patch.set_facecolor(_BG)

    t_start = max(0, t_idx - surface_window)
    surf_win = surf.iloc[t_start: t_idx + 1]
    x_idx = np.arange(len(surf_win))
    X, Y = np.meshgrid(x_idx, lags)
    Z = surf_win.values.T  # (n_lags, n_days)

    surf_plot = ax3d.plot_surface(
        X, Y, Z,
        cmap="plasma",
        vmin=z_min, vmax=z_max,
        alpha=0.88,
        linewidth=0,
        antialiased=True,
    )

    # Floating magenta diamond at the most-recent day, smallest lag
    latest_z = float(surf_win.iloc[-1, 0])
    ax3d.scatter(
        [len(surf_win) - 1], [lags[0]], [latest_z],
        color=_MAGENTA, s=140, marker="D", zorder=10, depthshade=False,
    )

    # Axis labels and style
    ax3d.set_xlabel("Tau", color=_TEXT, fontsize=8, labelpad=4)
    ax3d.set_ylabel("Kernel Offset", color=_TEXT, fontsize=8, labelpad=4)
    ax3d.set_zlabel("Vol", color=_TEXT, fontsize=8, labelpad=4)
    ax3d.tick_params(colors=_TEXT, labelsize=6)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1e1e1e")
    ax3d.xaxis.line.set_color(_TEXT)
    ax3d.yaxis.line.set_color(_TEXT)
    ax3d.zaxis.line.set_color(_TEXT)
    ax3d.grid(True, linestyle="--", linewidth=0.25, color="#2a2a2a")
    ax3d.view_init(elev=25, azim=225)

    # Colour-bar (right side)
    norm = Normalize(vmin=z_min, vmax=z_max)
    sm = ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, pad=0.02, shrink=0.45, aspect=14)
    cbar.set_label("Vol", color=_TEXT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=_TEXT, labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT)
    cbar.outline.set_edgecolor(_TEXT)

    # Top-left overlay: drop-down style UI labels
    ax3d.text2D(
        0.02, 0.97, "Order Flow", transform=ax3d.transAxes,
        color=_GREEN, fontsize=8, fontfamily="monospace", va="top",
    )
    ax3d.text2D(
        0.02, 0.90, "Region: S&P 500", transform=ax3d.transAxes,
        color=_GREEN, fontsize=8, fontfamily="monospace", va="top",
    )

    # ---------------------------------------------------------------
    # Mid section – dynamic data readout
    # ---------------------------------------------------------------
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_facecolor(_BG)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis("off")

    current_date = str(dates[t_idx])[:10]
    strat_val = float(port["portfolio_value"].iloc[t_idx])
    position = int(port["position"].iloc[t_idx])

    if position == 1:
        pos_label = "100% LONG"
        pos_color = _GREEN
    elif position == -1:
        pos_label = "100% BEAR"
        pos_color = _RED
    else:
        pos_label = "100% FLAT"
        pos_color = _ORANGE

    readout_items = [
        (0.01, f"Date:     {current_date}", _TEXT),
        (0.35, f"Strategy: ${strat_val:,.0f}", _GREEN),
        (0.68, f"Position: {pos_label}", pos_color),
    ]
    for x_frac, txt, col in readout_items:
        ax_mid.text(
            x_frac, 0.5, txt,
            color=col, fontsize=8.5, fontfamily="monospace",
            transform=ax_mid.transAxes, va="center", fontweight="bold",
        )

    # Thin horizontal separators
    for y in (0.0, 1.0):
        ax_mid.axhline(y, color="#2a2a2a", linewidth=0.8)

    # ---------------------------------------------------------------
    # Lower pane – portfolio backtest timeline
    # ---------------------------------------------------------------
    ax2d = fig.add_subplot(gs[2])
    ax2d.set_facecolor(_BG)

    hist_port = port["portfolio_value"].iloc[: t_idx + 1]
    hist_bh = bh_values.iloc[: t_idx + 1]
    hist_pos = port["position"].iloc[: t_idx + 1]
    hist_dates = dates[: t_idx + 1]

    # Shaded regime bars (non-long periods)
    _draw_regime_bars(ax2d, hist_dates, hist_pos)

    # S&P 500 buy-and-hold line (neon orange)
    ax2d.plot(hist_dates, hist_bh.values, color=_ORANGE, linewidth=1.0,
              zorder=4, alpha=0.9)

    # Strategy equity line (neon green)
    ax2d.plot(hist_dates, hist_port.values, color=_GREEN, linewidth=1.3,
              zorder=5)

    # Floating end-of-line value labels
    if len(hist_dates) > 0:
        ax2d.text(
            hist_dates[-1], float(hist_port.iloc[-1]),
            f"  ${hist_port.iloc[-1]:,.0f}",
            color=_GREEN, fontsize=6.5, va="center", fontfamily="monospace",
            clip_on=True,
        )
        ax2d.text(
            hist_dates[-1], float(hist_bh.iloc[-1]),
            f"  ${hist_bh.iloc[-1]:,.0f}",
            color=_ORANGE, fontsize=6.5, va="center", fontfamily="monospace",
            clip_on=True,
        )

    # Legend with coloured dots
    bh_handle = mlines.Line2D(
        [], [], color=_ORANGE, marker="o", markersize=5,
        label="S&P 500 Buy & Hold", linewidth=1.0,
    )
    strat_handle = mlines.Line2D(
        [], [], color=_GREEN, marker="o", markersize=5,
        label="Strategy", linewidth=1.3,
    )
    ax2d.legend(
        handles=[bh_handle, strat_handle],
        loc="upper left", fontsize=7,
        facecolor="#111111", edgecolor=_CYAN,
        labelcolor=[_ORANGE, _GREEN],
    )

    # Axes style
    ax2d.set_xlim(dates[0], dates[-1])
    ax2d.set_xlabel("Year", color=_TEXT, fontsize=8)
    ax2d.set_ylabel("Portfolio Value ($)", color=_TEXT, fontsize=8)
    ax2d.tick_params(colors=_TEXT, labelsize=7)
    for spine_name, spine in ax2d.spines.items():
        spine.set_color(_CYAN if spine_name in ("bottom", "left") else "#1e1e1e")
    ax2d.yaxis.grid(True, linestyle="--", linewidth=0.35, color="#1a1a2e", alpha=0.9)
    ax2d.xaxis.grid(False)

    # ---------------------------------------------------------------
    # Convert figure → BGR numpy array
    # ---------------------------------------------------------------
    img = _fig_to_bgr(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    """Serialize a matplotlib figure to a BGR numpy array (OpenCV format)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=_BG, bbox_inches="tight",
                pad_inches=0)
    buf.seek(0)
    raw = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return img


def _draw_regime_bars(ax, dates, position: pd.Series) -> None:
    """Shade non-long (flat / short) periods with semi-transparent orange bars."""
    if len(dates) < 2:
        return
    dates_arr = np.array(dates, dtype="datetime64[ns]")
    pos_arr = position.values
    in_regime = False
    start = None
    for d, p in zip(dates_arr, pos_arr):
        if p != 1 and not in_regime:
            in_regime = True
            start = d
        elif p == 1 and in_regime:
            ax.axvspan(start, d, color=_ORANGE, alpha=0.13, linewidth=0)
            in_regime = False
    if in_regime and start is not None:
        ax.axvspan(start, dates_arr[-1], color=_ORANGE, alpha=0.13, linewidth=0)


def _try_open_video(path: str) -> None:
    """Attempt to open the video with the system default media player."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif sys.platform.startswith("linux"):
        for player in ("xdg-open", "vlc", "mplayer", "ffplay"):
            try:
                subprocess.Popen(
                    [player, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break
            except FileNotFoundError:
                continue
    elif sys.platform == "win32":
        try:
            os.startfile(path)
        except (AttributeError, OSError):
            pass
