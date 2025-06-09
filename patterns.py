
"""Candlestick pattern detection utilities."""

from __future__ import annotations

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_candlestick_patterns(ohlcv_df: pd.DataFrame, pattern_list: list[str]) -> pd.DataFrame:
    """Detect candlestick patterns in OHLCV data using pandas-ta.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        DataFrame containing OHLCV data with Datetime index.
    pattern_list : list[str]
        List of candlestick pattern names compatible with ``pandas_ta``.

    Returns
    -------
    pd.DataFrame
        DataFrame of detected patterns with columns ``Datetime``, ``Pattern`` and ``Value``.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        logger.warning("Empty OHLCV DataFrame provided to detect_candlestick_patterns")
        return pd.DataFrame(columns=["Datetime", "Pattern", "Value"])

    results = []
    for pattern in pattern_list:
        try:
            series = ohlcv_df.ta.cdl_pattern(name=pattern)
        except Exception as exc:
            logger.error("Failed to compute pattern %s: %s", pattern, exc)
            continue

        if series is None:
            continue

        occurrences = series[series != 0]
        if occurrences.empty:
            continue

        pattern_df = pd.DataFrame({
            "Datetime": occurrences.index,
            "Pattern": pattern,
            "Value": occurrences.values,
        })
        results.append(pattern_df)

    if not results:
        return pd.DataFrame(columns=["Datetime", "Pattern", "Value"])

    result_df = pd.concat(results, ignore_index=True)
    result_df.sort_values("Datetime", inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df
