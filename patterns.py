# fivepaisa_ai_dashboard/patterns.py
"""Utilities for detecting candlestick patterns."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from typing import Optional

from config import CANDLESTICK_PATTERNS_TO_DETECT


def detect_candlestick_patterns(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of detected candlestick patterns.

    The returned DataFrame is indexed by Datetime and contains columns:
    ``pattern`` and ``direction`` ("bullish" or "bearish") and ``price`` which
    is the associated high/low price for plotting markers.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.DataFrame(columns=["pattern", "direction", "price"])

    results = []
    for pattern in CANDLESTICK_PATTERNS_TO_DETECT:
        try:
            series = ohlcv_df.ta.cdl_pattern(name=pattern)
        except Exception:
            # Skip patterns not supported by pandas_ta
            continue
        if series is None:
            continue
        for idx, val in series.items():
            if pd.isna(val) or val == 0:
                continue
            direction = "bullish" if val > 0 else "bearish"
            price = ohlcv_df.loc[idx, "Low"] if val > 0 else ohlcv_df.loc[idx, "High"]
            results.append({"Datetime": idx, "pattern": pattern, "direction": direction, "price": price})

    if not results:
        return pd.DataFrame(columns=["pattern", "direction", "price"])

    df = pd.DataFrame(results).set_index("Datetime")
    return df
