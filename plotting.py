# fivepaisa_ai_dashboard/plotting.py
"""Plotting utilities for visualizing OHLCV data and technical indicators."""

from __future__ import annotations

from typing import Optional, Dict

import pandas as pd
import plotly.graph_objects as go

from config import (
    DEFAULT_INDICATORS,
    PATTERN_MARKER_CONFIG,
    CHART_EXPORT_FORMAT,
    CHART_EXPORT_SCALE,
)


def create_ohlcv_chart(
    ohlcv_df: pd.DataFrame,
    scrip_symbol: str,
    indicator_data: Optional[pd.DataFrame] = None,
    indicator_config: Optional[Dict[str, bool]] = None,
    pattern_data: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Return a Plotly Figure showing OHLCV bars with optional overlays."""
    if ohlcv_df is None or ohlcv_df.empty:
        raise ValueError("OHLCV dataframe is empty")

    x_vals = ohlcv_df["Datetime"] if "Datetime" in ohlcv_df.columns else ohlcv_df.index
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_vals,
                open=ohlcv_df["Open"],
                high=ohlcv_df["High"],
                low=ohlcv_df["Low"],
                close=ohlcv_df["Close"],
                name="OHLC",
            )
        ]
    )

    if indicator_data is not None and indicator_config:
        x_ind = indicator_data["Datetime"] if "Datetime" in indicator_data.columns else indicator_data.index
        for ind_key, enabled in indicator_config.items():
            if not enabled:
                continue
            cfg = DEFAULT_INDICATORS.get(ind_key)
            if not cfg:
                continue
            pinfo = cfg.get("plot_info", {})
            if not pinfo.get("on_price_chart", True):
                continue
            plot_type = pinfo.get("type", "line")
            color = pinfo.get("color")
            if plot_type == "line" and cfg.get("output_cols"):
                col = cfg["output_cols"][0]
                if col in indicator_data:
                    fig.add_trace(
                        go.Scatter(
                            x=x_ind,
                            y=indicator_data[col],
                            mode="lines",
                            line=dict(color=color),
                            name=cfg.get("label", ind_key),
                        )
                    )
            elif plot_type == "bands":
                lower = pinfo.get("lower_band_col")
                upper = pinfo.get("upper_band_col")
                middle = pinfo.get("middle_band_col")
                fill_color = pinfo.get("color", {}).get("bands_fill", "rgba(0,0,0,0.1)")
                if upper in indicator_data and lower in indicator_data:
                    fig.add_trace(
                        go.Scatter(
                            x=x_ind,
                            y=indicator_data[upper],
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_ind,
                            y=indicator_data[lower],
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=fill_color,
                            name=cfg.get("label", ind_key),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                if middle and middle in indicator_data:
                    mid_color = pinfo.get("color", {}).get("middle", color)
                    fig.add_trace(
                        go.Scatter(
                            x=x_ind,
                            y=indicator_data[middle],
                            mode="lines",
                            line=dict(color=mid_color, width=1),
                            name=f"{cfg.get('label', ind_key)} Mid",
                        )
                    )

    if pattern_data is not None and not pattern_data.empty:
        price_range = ohlcv_df["High"].max() - ohlcv_df["Low"].min()
        offset = price_range * PATTERN_MARKER_CONFIG.get("y_offset_percentage", 0.02)
        for _, row in pattern_data.iterrows():
            dt = row.get("Datetime")
            signal = str(row.get("Signal", "neutral")).lower()
            patt_name = row.get("Pattern", "pattern")
            candle = ohlcv_df[ohlcv_df["Datetime"] == dt]
            if candle.empty:
                continue
            high = float(candle.iloc[0]["High"])
            low = float(candle.iloc[0]["Low"])
            if signal == "bullish":
                y = low - offset
                color = PATTERN_MARKER_CONFIG.get("bullish_color")
                symbol = PATTERN_MARKER_CONFIG.get("bullish_symbol")
            elif signal == "bearish":
                y = high + offset
                color = PATTERN_MARKER_CONFIG.get("bearish_color")
                symbol = PATTERN_MARKER_CONFIG.get("bearish_symbol")
            else:
                y = high + offset
                color = PATTERN_MARKER_CONFIG.get("neutral_color")
                symbol = PATTERN_MARKER_CONFIG.get("neutral_symbol")
            fig.add_trace(
                go.Scatter(
                    x=[dt],
                    y=[y],
                    mode="markers",
                    marker=dict(color=color, symbol=symbol, size=PATTERN_MARKER_CONFIG.get("size", 8)),
                    name=patt_name,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{scrip_symbol} OHLCV Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#4A4A4A")
    fig.update_yaxes(showgrid=True, gridcolor="#4A4A4A")
    return fig


def export_figure_to_png(fig: go.Figure, file_path: str, scale: int | None = None) -> None:
    """Export a Plotly figure to a PNG file using configured defaults."""
    fig.write_image(file_path, format=CHART_EXPORT_FORMAT, scale=scale or CHART_EXPORT_SCALE)


__all__ = ["create_ohlcv_chart", "export_figure_to_png"]
