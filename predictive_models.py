# fivepaisa_ai_dashboard/predictive_models.py
"""Simple predictive model stubs."""

from __future__ import annotations

import pandas as pd
from typing import Dict


def get_prediction(
    ohlcv_df: pd.DataFrame,
    indicator_data: pd.DataFrame,
    symbol: str,
    horizon_key: str,
    horizon_config: Dict[str, any],
) -> Dict[str, any]:
    """Return a mock prediction for the given horizon.

    This function is a placeholder for real model inference.  It returns a
    basic momentum signal based on recent closing prices.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return {"signal": "N/A", "confidence": 0.0, "expected_return_pct": 0.0, "model_type": "na"}

    fwd = max(1, horizon_config.get("forward_periods", 1))
    if len(ohlcv_df) <= fwd:
        return {"signal": "N/A", "confidence": 0.0, "expected_return_pct": 0.0, "model_type": "na"}

    last_close = ohlcv_df["Close"].iloc[-1]
    prev_close = ohlcv_df["Close"].iloc[-fwd]
    pct_change = (last_close - prev_close) / prev_close
    signal = "UP" if pct_change > 0 else "DOWN"
    return {
        "signal": signal,
        "confidence": abs(pct_change),
        "expected_return_pct": pct_change,
        "model_type": "naive_momentum",
    }
