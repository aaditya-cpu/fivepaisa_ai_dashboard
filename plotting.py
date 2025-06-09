# fivepaisa_ai_dashboard/plotting.py
"""Plotting utilities for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional

from config import DEFAULT_INDICATORS, PATTERN_MARKER_CONFIG


def create_ohlcv_chart(
    ohlcv_df: pd.DataFrame,
    symbol: str,
    indicator_data: pd.DataFrame,
    enabled_indicators: Dict[str, bool],
    patterns_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Return a Plotly figure with OHLC candles, indicators and pattern markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlcv_df.index,
            open=ohlcv_df["Open"],
            high=ohlcv_df["High"],
            low=ohlcv_df["Low"],
            close=ohlcv_df["Close"],
            name="OHLC",
        )
    )

    for ind_key, enabled in enabled_indicators.items():
        if not enabled or ind_key not in DEFAULT_INDICATORS:
            continue
        conf = DEFAULT_INDICATORS[ind_key]
        color = conf.get("plot_info", {}).get("color", None)
        for col in conf.get("output_cols", []):
            if col in indicator_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data.index,
                        y=indicator_data[col],
                        name=col,
                        line=dict(color=color),
                        mode="lines",
                    )
                )

    if patterns_df is not None and not patterns_df.empty:
        bull_df = patterns_df[patterns_df["direction"] == "bullish"]
        bear_df = patterns_df[patterns_df["direction"] == "bearish"]
        if not bull_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull_df.index,
                    y=bull_df["price"],
                    mode="markers",
                    marker=dict(
                        symbol=PATTERN_MARKER_CONFIG["bullish_symbol"],
                        color=PATTERN_MARKER_CONFIG["bullish_color"],
                        size=PATTERN_MARKER_CONFIG["size"],
                    ),
                    name="Bullish Pattern",
                )
            )
        if not bear_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear_df.index,
                    y=bear_df["price"],
                    mode="markers",
                    marker=dict(
                        symbol=PATTERN_MARKER_CONFIG["bearish_symbol"],
                        color=PATTERN_MARKER_CONFIG["bearish_color"],
                        size=PATTERN_MARKER_CONFIG["size"],
                    ),
                    name="Bearish Pattern",
                )
            )

    fig.update_layout(
        title=f"{symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
    )
    return fig
