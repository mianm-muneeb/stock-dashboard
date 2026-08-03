"""Shared plotly chart helpers.

render_price_chart() supports four views (Performance, Price, Candlestick,
Area), all rendered with the light theme. Performance and Area
views split their line/fill into green (above baseline) and red (below
baseline) segments, interpolating the exact zero-crossing point so the color
change lands precisely on the baseline.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.signals import (
    atr_series,
    bollinger_bands,
    ema,
    macd_series,
    moving_average,
    obv_series,
    rsi_series,
    stochastic_oscillator,
    vwap_series,
)

GREEN = "#2ecc71"
RED = "#e74c3c"
LINE_BLUE = "#5dade2"
AMBER = "#f1c40f"
PURPLE = "#a78bfa"
GRAY = "#8b949e"

# TradingView dark theme palette
TV_BG = "#131722"
TV_PANEL = "#1e222d"
TV_GRID = "#2a2e39"
TV_CROSSHAIR = "#758696"
TV_GREEN = "#26a69a"
TV_RED = "#ef5350"

# TradingView light theme palette
LT_BG = "#ffffff"
LT_PANEL = "#f0f3fa"
LT_GRID = "#e0e3eb"
LT_CROSSHAIR = "#9598a1"
LT_TEXT = "#131722"


def _chart_theme() -> dict:
    """Return the TradingView dark chart palette."""
    return {
        "template": "plotly_dark",
        "paper_bgcolor": TV_BG,
        "plot_bgcolor": TV_BG,
        "font_color": "#d1d4dc",
        "grid": TV_GRID,
        "zeroline": TV_GRID,
        "spike": TV_CROSSHAIR,
        "axis_color": "#787b86",
        "hline_color": "rgba(255,255,255,0.20)",
    }


def split_traces(x, y, baseline):
    """Split a series into contiguous segments above/below `baseline`.

    Returns a list of (x_segment, y_segment, color) tuples where color is
    GREEN or RED. Crossing points are linearly interpolated so segments meet
    exactly at the baseline.
    """
    x = list(x)
    y = list(y)
    if not x:
        return []

    segments = []
    cur_x, cur_y = [x[0]], [y[0]]
    cur_color = GREEN if y[0] >= baseline else RED

    for i in range(1, len(x)):
        prev_y, curr_y = y[i - 1], y[i]
        prev_x, curr_x = x[i - 1], x[i]
        color = GREEN if curr_y >= baseline else RED

        if color != cur_color:
            if curr_y != prev_y:
                frac = (baseline - prev_y) / (curr_y - prev_y)
            else:
                frac = 0.0
            frac = min(max(frac, 0.0), 1.0)

            if isinstance(prev_x, (pd.Timestamp, dt.datetime)):
                cross_x = prev_x + (curr_x - prev_x) * frac
            else:
                cross_x = prev_x + (curr_x - prev_x) * frac

            cur_x.append(cross_x)
            cur_y.append(baseline)
            segments.append((cur_x, cur_y, cur_color))

            cur_x, cur_y = [cross_x], [baseline]
            cur_color = color

        cur_x.append(curr_x)
        cur_y.append(curr_y)

    segments.append((cur_x, cur_y, cur_color))
    return segments


def _infer_step(index: pd.DatetimeIndex):
    if len(index) >= 2:
        return index[1] - index[0]
    return dt.timedelta(days=1)


def _format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def render_price_chart(
    df: pd.DataFrame,
    view: str = "Performance",
    baseline_price: float | None = None,
    title: str | None = None,
    show_volume: bool = False,
    height: int = 450,
    show_badge: bool = True,
):
    """Render a price chart in one of four views.

    Parameters
    ----------
    df : DataFrame with columns Open/High/Low/Close[/Volume] and a
        DatetimeIndex, as returned by lib.market_data.get_history.
    view : "Performance" | "Price" | "Candlestick" | "Area"
    baseline_price : optional reference price (e.g. yesterday's close for a
        1D chart). When provided, Performance/Area views are baselined on it
        and a synthetic leading point is added so the line visibly starts
        from that level.
    show_volume : add a volume subplot beneath the price panel.
    """
    ct = _chart_theme()

    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=ct["template"],
            height=height,
            title=title,
            paper_bgcolor=ct["paper_bgcolor"],
            plot_bgcolor=ct["plot_bgcolor"],
            font=dict(color=ct["font_color"]),
            annotations=[dict(text="No data available", showarrow=False, font=dict(size=16))],
        )
        return fig

    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=ct["template"], height=height, title=title,
            paper_bgcolor=ct["paper_bgcolor"], plot_bgcolor=ct["plot_bgcolor"],
        )
        return fig

    x = df.index
    close = df["Close"]

    # Reference value for Performance/Area baselines.
    ref_price = baseline_price if baseline_price is not None else float(close.iloc[0])

    # Optionally prepend a synthetic point at `ref_price` so the chart
    # visibly starts from yesterday's close (1D charts).
    plot_x = list(x)
    plot_close = list(close.values)
    if baseline_price is not None:
        step = _infer_step(x)
        lead_x = x[0] - step
        plot_x = [lead_x] + plot_x
        plot_close = [baseline_price] + plot_close

    if show_volume and "Volume" in df.columns:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    end_return = None

    if view == "Performance":
        pct = [((p / ref_price) - 1.0) * 100 for p in plot_close]
        for seg_x, seg_y, color in split_traces(plot_x, pct, 0.0):
            fig.add_trace(
                go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", line=dict(color=color, width=2),
                    showlegend=False, hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.add_hline(y=0, line=dict(color=ct["hline_color"], width=1, dash="dot"), row=1, col=1)
        end_return = pct[-1]
        fig.update_yaxes(title_text="Change (%)", row=1, col=1)

    elif view == "Area":
        for seg_x, seg_y, color in split_traces(plot_x, plot_close, ref_price):
            fillcolor = "rgba(46,204,113,0.18)" if color == GREEN else "rgba(231,76,60,0.18)"
            fig.add_trace(
                go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", line=dict(color=color, width=2),
                    fill="tozeroy", fillcolor=fillcolor, showlegend=False,
                    hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.add_hline(y=ref_price, line=dict(color=ct["hline_color"], width=1, dash="dot"), row=1, col=1)
        fig.update_yaxes(range=[min(plot_close) * 0.98, max(plot_close) * 1.02], title_text="Price", row=1, col=1)
        end_return = ((plot_close[-1] / ref_price) - 1.0) * 100

    elif view == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False,
            ),
            row=1, col=1,
        )
        end_return = ((close.iloc[-1] / ref_price) - 1.0) * 100
        fig.update_yaxes(title_text="Price", row=1, col=1)

    else:  # "Price"
        color = GREEN if close.iloc[-1] >= ref_price else RED
        fig.add_trace(
            go.Scatter(
                x=x, y=close, mode="lines", line=dict(color=LINE_BLUE, width=2),
                showlegend=False, hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        end_return = ((close.iloc[-1] / ref_price) - 1.0) * 100
        fig.update_yaxes(title_text="Price", row=1, col=1)

    if show_volume and "Volume" in df.columns:
        vol_colors = [GREEN if c >= o else RED for o, c in zip(df["Open"], df["Close"])]
        fig.add_trace(
            go.Bar(x=x, y=df["Volume"], marker_color=vol_colors, showlegend=False, opacity=0.6),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    # End-of-period return badge anchored at the last point.
    if show_badge and end_return is not None:
        if view == "Performance":
            badge_y = end_return
        elif view in ("Area", "Price"):
            badge_y = float(close.iloc[-1])
        else:
            badge_y = float(close.iloc[-1])

        badge_color = GREEN if end_return >= 0 else RED
        fig.add_annotation(
            x=x[-1], y=badge_y, xref="x", yref="y" if view != "Candlestick" else "y",
            text=f"  {_format_pct(end_return)}  ",
            showarrow=False, font=dict(color="#0d1117", size=13, family="Arial Black"),
            bgcolor=badge_color, bordercolor=badge_color, borderwidth=1, borderpad=3,
            xanchor="left", row=1, col=1,
        )

    fig.update_layout(
        template=ct["template"],
        height=height,
        title=title,
        margin=dict(l=10, r=60, t=40 if title else 20, b=10),
        hovermode="x unified",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        paper_bgcolor=ct["paper_bgcolor"],
        plot_bgcolor=ct["plot_bgcolor"],
        font=dict(color=ct["font_color"]),
    )
    fig.update_xaxes(showgrid=True, gridcolor=ct["grid"], zerolinecolor=ct["zeroline"], color=ct["axis_color"])
    fig.update_yaxes(showgrid=True, gridcolor=ct["grid"], zerolinecolor=ct["zeroline"], color=ct["axis_color"])
    return fig


def render_gauge(value: float | None, color_scale: str = "red_green", height: int = 220):
    """Render a 0-100 needle gauge with a red-to-green (or green-to-red) band.

    `color_scale`:
      - "red_green": low values red, high values green (used for the
        Technical strength / Fundamental quality score gauges).
      - "green_red": low values green, high values red (used for the ETF
        risk gauge, where higher = historically more variable).
    """
    if value is None:
        value = 0

    ct = _chart_theme()
    tick_color = ct["axis_color"]
    needle_color = ct["font_color"]

    red_green = ["#e74c3c", "#e67e22", "#f1c40f", "#a3e635", "#2ecc71"]
    bands = red_green if color_scale == "red_green" else list(reversed(red_green))

    steps = [
        {"range": [i * 20, (i + 1) * 20], "color": bands[i]}
        for i in range(5)
    ]

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": tick_color},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": steps,
                "threshold": {"line": {"color": needle_color, "width": 4}, "thickness": 0.85, "value": value},
            },
        )
    )
    fig.update_layout(
        template=ct["template"],
        paper_bgcolor=ct["paper_bgcolor"],
        font=dict(color=ct["font_color"]),
        height=height,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


OVERLAY_OPTIONS = ["EMA20", "EMA50", "EMA200", "SMA20", "SMA50", "SMA200", "Bollinger Bands", "VWAP"]
PANEL_OPTIONS = ["Volume", "RSI", "MACD", "Stochastic", "ATR", "OBV"]


def render_technical_chart(
    df: pd.DataFrame,
    view: str = "Performance",
    baseline_price: float | None = None,
    title: str | None = None,
    overlays: list[str] | None = None,
    panels: list[str] | None = None,
    height: int = 650,
):
    """Render a multi-panel, TradingView-styled technical chart: price (with
    optional overlays and a volume overlay) on top, and a stacked sub-panel
    per indicator entry in `panels` below.

    `overlays` may include any of OVERLAY_OPTIONS and are drawn on the price
    panel (scaled to match the active `view`).
    `panels` may include any of PANEL_OPTIONS. "Volume" is rendered as a
    semi-transparent overlay on the price panel rather than its own row;
    the remaining entries ("RSI", "MACD", "Stochastic", "ATR", "OBV") are
    rendered as stacked subplots beneath the price panel, sharing the
    x-axis (60% main / split evenly among the rest).
    """
    ct = _chart_theme()
    title = title or ""
    overlays = overlays or []
    panels = [p for p in (panels or []) if p in PANEL_OPTIONS]
    show_volume = "Volume" in panels and "Volume" in df.columns
    sub_panels = [p for p in panels if p != "Volume"]

    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=ct["template"],
            height=height,
            title=title,
            paper_bgcolor=ct["paper_bgcolor"],
            plot_bgcolor=ct["plot_bgcolor"],
            font=dict(color=ct["font_color"]),
            annotations=[dict(text="No data available", showarrow=False, font=dict(size=16))],
        )
        return fig

    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template=ct["template"], height=height, title=title,
            paper_bgcolor=ct["paper_bgcolor"], plot_bgcolor=ct["plot_bgcolor"],
        )
        return fig

    x = df.index
    close = df["Close"]
    ref_price = baseline_price if baseline_price is not None else float(close.iloc[0])

    plot_x = list(x)
    plot_close = list(close.values)
    if baseline_price is not None:
        step = _infer_step(x)
        plot_x = [x[0] - step] + plot_x
        plot_close = [baseline_price] + plot_close

    n_sub = len(sub_panels)
    if n_sub == 0:
        row_heights = [1.0]
    else:
        main_h = 0.6
        other_h = (1 - main_h) / n_sub
        row_heights = [main_h] + [other_h] * n_sub

    specs = [[{"secondary_y": True}]] + [[{}] for _ in range(n_sub)]
    fig = make_subplots(
        rows=1 + n_sub, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03, specs=specs,
    )

    def to_scale(series: pd.Series) -> pd.Series:
        if view == "Performance":
            return (series / ref_price - 1.0) * 100
        return series

    end_return = None

    # -- Price panel -----------------------------------------------------
    if view == "Performance":
        pct = [((p / ref_price) - 1.0) * 100 for p in plot_close]
        for seg_x, seg_y, color in split_traces(plot_x, pct, 0.0):
            fig.add_trace(
                go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", line=dict(color=color, width=2),
                    showlegend=False, hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.add_hline(y=0, line=dict(color=ct["hline_color"], width=1, dash="dot"), row=1, col=1)
        end_return = pct[-1]
        fig.update_yaxes(title_text="Change (%)", row=1, col=1)

    elif view == "Area":
        for seg_x, seg_y, color in split_traces(plot_x, plot_close, ref_price):
            fillcolor = "rgba(46,204,113,0.18)" if color == GREEN else "rgba(231,76,60,0.18)"
            fig.add_trace(
                go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", line=dict(color=color, width=2),
                    fill="tozeroy", fillcolor=fillcolor, showlegend=False,
                    hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
        fig.add_hline(y=ref_price, line=dict(color=ct["hline_color"], width=1, dash="dot"), row=1, col=1)
        fig.update_yaxes(range=[min(plot_close) * 0.98, max(plot_close) * 1.02], title_text="Price", row=1, col=1)
        end_return = ((plot_close[-1] / ref_price) - 1.0) * 100

    elif view == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color=TV_GREEN, decreasing_line_color=TV_RED,
                increasing_fillcolor=TV_GREEN, decreasing_fillcolor=TV_RED,
                line=dict(width=1), showlegend=False,
                name="Price",
            ),
            row=1, col=1,
        )
        end_return = ((close.iloc[-1] / ref_price) - 1.0) * 100
        fig.update_yaxes(title_text="Price", row=1, col=1)

    else:  # "Price"
        fig.add_trace(
            go.Scatter(
                x=x, y=close, mode="lines", line=dict(color=LINE_BLUE, width=2),
                showlegend=False, hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>", name="Price",
            ),
            row=1, col=1,
        )
        end_return = ((close.iloc[-1] / ref_price) - 1.0) * 100
        fig.update_yaxes(title_text="Price", row=1, col=1)

    # -- Overlays (drawn on the price panel) ------------------------------
    overlay_colors = {
        "EMA20": "#FF9800", "EMA50": "#2962FF", "EMA200": "#F23645",
        "SMA20": "#f97316", "SMA50": "#2dd4bf", "SMA200": "#ec4899",
    }
    for ema_label, window in (("EMA20", 20), ("EMA50", 50), ("EMA200", 200)):
        if ema_label in overlays and len(close) >= window:
            ma = ema(close, window)
            fig.add_trace(
                go.Scatter(
                    x=x, y=to_scale(ma), mode="lines",
                    line=dict(color=overlay_colors[ema_label], width=1.3),
                    name=ema_label, showlegend=True,
                    hovertemplate=f"{ema_label}: " + "%{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

    for sma_label, window in (("SMA20", 20), ("SMA50", 50), ("SMA200", 200)):
        if sma_label in overlays and len(close) >= window:
            ma = moving_average(close, window)
            fig.add_trace(
                go.Scatter(
                    x=x, y=to_scale(ma), mode="lines",
                    line=dict(color=overlay_colors[sma_label], width=1.3, dash="dot"),
                    name=sma_label, showlegend=True,
                    hovertemplate=f"{sma_label}: " + "%{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

    if "Bollinger Bands" in overlays and len(close) >= 20:
        mid, upper, lower = bollinger_bands(close, 20, 2.0)
        fig.add_trace(
            go.Scatter(
                x=x, y=to_scale(upper), mode="lines", line=dict(color=GRAY, width=1, dash="dot"),
                name="BB Upper", showlegend=True, hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=to_scale(mid), mode="lines", line=dict(color=GRAY, width=1, dash="dash"),
                name="BB Mid (20)", showlegend=True, hovertemplate="BB Mid: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=to_scale(lower), mode="lines", line=dict(color=GRAY, width=1, dash="dot"),
                name="BB Lower", showlegend=True, hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if "VWAP" in overlays and "Volume" in df.columns:
        vwap = vwap_series(df)
        fig.add_trace(
            go.Scatter(
                x=x, y=to_scale(vwap), mode="lines", line=dict(color="#f97316", width=1.3, dash="dash"),
                name="VWAP", showlegend=True, hovertemplate="VWAP: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # -- Volume overlay (semi-transparent, bottom of the price panel) -----
    if show_volume:
        vol_colors = [
            "rgba(38,166,154,0.35)" if c >= o else "rgba(239,83,80,0.35)"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=x, y=df["Volume"], marker_color=vol_colors, showlegend=False, name="Volume",
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.update_yaxes(
            range=[0, float(df["Volume"].max()) * 4], showgrid=False, showticklabels=False,
            row=1, col=1, secondary_y=True,
        )

    # -- End-of-period return badge ---------------------------------------
    if end_return is not None:
        badge_y = end_return if view == "Performance" else float(close.iloc[-1])
        badge_color = GREEN if end_return >= 0 else RED
        fig.add_annotation(
            x=x[-1], y=badge_y, xref="x", yref="y1",
            text=f"  {_format_pct(end_return)}  ",
            showarrow=False, font=dict(color="#0d1117", size=13, family="Arial Black"),
            bgcolor=badge_color, bordercolor=badge_color, borderwidth=1, borderpad=3,
            xanchor="left", row=1, col=1,
        )

    # -- Sub-panels ---------------------------------------------------------
    for i, panel in enumerate(sub_panels):
        row = 2 + i

        if panel == "RSI":
            rsi = rsi_series(close)
            fig.add_trace(
                go.Scatter(x=x, y=rsi, mode="lines", line=dict(color=PURPLE, width=1.5), name="RSI", showlegend=False),
                row=row, col=1,
            )
            fig.add_hline(y=70, line=dict(color="rgba(231,76,60,0.5)", width=1, dash="dash"), row=row, col=1)
            fig.add_hline(y=30, line=dict(color="rgba(46,204,113,0.5)", width=1, dash="dash"), row=row, col=1)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)

        elif panel == "MACD":
            macd_line, signal_line, hist = macd_series(close)
            hist_colors = [GREEN if v >= 0 else RED for v in hist]
            fig.add_trace(
                go.Bar(x=x, y=hist, marker_color=hist_colors, showlegend=False, opacity=0.5, name="MACD Hist"),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=macd_line, mode="lines", line=dict(color=LINE_BLUE, width=1.5), name="MACD", showlegend=False),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=signal_line, mode="lines", line=dict(color=AMBER, width=1.5), name="Signal", showlegend=False),
                row=row, col=1,
            )
            fig.update_yaxes(title_text="MACD", row=row, col=1)

        elif panel == "Stochastic":
            percent_k, percent_d = stochastic_oscillator(df)
            fig.add_trace(
                go.Scatter(x=x, y=percent_k, mode="lines", line=dict(color=LINE_BLUE, width=1.5), name="%K", showlegend=False),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(x=x, y=percent_d, mode="lines", line=dict(color=AMBER, width=1.5), name="%D", showlegend=False),
                row=row, col=1,
            )
            fig.add_hline(y=80, line=dict(color="rgba(231,76,60,0.5)", width=1, dash="dash"), row=row, col=1)
            fig.add_hline(y=20, line=dict(color="rgba(46,204,113,0.5)", width=1, dash="dash"), row=row, col=1)
            fig.update_yaxes(title_text="Stoch", range=[0, 100], row=row, col=1)

        elif panel == "ATR":
            atr = atr_series(df)
            fig.add_trace(
                go.Scatter(x=x, y=atr, mode="lines", line=dict(color=AMBER, width=1.5), name="ATR", showlegend=False),
                row=row, col=1,
            )
            fig.update_yaxes(title_text="ATR", row=row, col=1)

        elif panel == "OBV":
            obv = obv_series(df)
            fig.add_trace(
                go.Scatter(x=x, y=obv, mode="lines", line=dict(color=PURPLE, width=1.5), name="OBV", showlegend=False),
                row=row, col=1,
            )
            fig.update_yaxes(title_text="OBV", row=row, col=1)

    last_row = 1 + n_sub

    fig.update_layout(
        template=ct["template"],
        height=height,
        title=dict(text=title, x=0, font=dict(size=14)),
        margin=dict(l=10, r=60, t=40 if title else 20, b=10),
        hovermode="x unified",
        showlegend=bool(overlays),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
        dragmode="pan",
        paper_bgcolor=ct["paper_bgcolor"],
        plot_bgcolor=ct["plot_bgcolor"],
        font=dict(color=ct["font_color"]),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=ct["grid"], zerolinecolor=ct["zeroline"],
        color=ct["axis_color"],
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor=ct["spike"], spikethickness=1, spikedash="solid",
        rangeslider_visible=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=ct["grid"], zerolinecolor=ct["zeroline"],
        color=ct["axis_color"],
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor=ct["spike"], spikethickness=1, spikedash="solid",
    )
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.06, row=last_row, col=1)
    return fig


def render_sparkline(df: pd.DataFrame, baseline_price: float | None = None, height: int = 60):
    """Tiny green/red split line chart with no axes - used in Market Pulse cards."""
    ct = _chart_theme()

    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=height, margin=dict(l=0, r=0, t=0, b=0),
            template=ct["template"],
            paper_bgcolor=ct["paper_bgcolor"],
            plot_bgcolor=ct["plot_bgcolor"],
        )
        return fig

    close = df["Close"].dropna()
    x = list(close.index)
    y = list(close.values)
    ref = baseline_price if baseline_price is not None else float(y[0])

    if baseline_price is not None:
        step = _infer_step(close.index)
        x = [x[0] - step] + x
        y = [baseline_price] + y

    fig = go.Figure()
    for seg_x, seg_y, color in split_traces(x, y, ref):
        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", line=dict(color=color, width=1.5), showlegend=False))

    fig.update_layout(
        template=ct["template"],
        paper_bgcolor=ct["paper_bgcolor"],
        plot_bgcolor=ct["plot_bgcolor"],
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        hovermode=False,
    )
    return fig
