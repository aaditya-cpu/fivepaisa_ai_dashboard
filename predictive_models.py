
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd
import joblib

from config import (
    APP_TITLE,
    MODELS_STORE_DIR,
    MODEL_FILENAME_FORMAT,
    DEFAULT_MODEL_ARCHITECTURE,
    LAG_FEATURES_COUNT,
    ROLLING_WINDOW_FEATURES_SIZES,
)

logger = logging.getLogger(f"{APP_TITLE}.PredictiveModels")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_model(model_path: str) -> Optional[Any]:
    """Load a serialized model from ``model_path`` using ``joblib``.

    Parameters
    ----------
    model_path: str
        Path to the serialized model file.

    Returns
    -------
    The loaded model instance or ``None`` if loading failed.
    """
    try:
        logger.info(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
    except Exception as exc:  # pragma: no cover - generic error log
        logger.error(f"Failed to load model from {model_path}: {exc}")
    return None


def _construct_model_path(scrip_symbol: str, horizon_conf: dict) -> str:
    """Helper to build the absolute path to a stored model."""
    model_arch = DEFAULT_MODEL_ARCHITECTURE.get(
        horizon_conf.get("target_type", "classification"), "LightGBM_Classifier"
    )
    filename = MODEL_FILENAME_FORMAT.format(
        scrip_symbol=scrip_symbol,
        horizon_prefix=horizon_conf.get("model_prefix", "unknown"),
        model_arch=model_arch,
    )
    return os.path.join(MODELS_STORE_DIR, filename)


def get_prediction(
    ohlcv_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    scrip_symbol: str,
    horizon_key: str,
    horizon_conf: dict,
) -> Optional[Any]:
    """Generate a prediction for ``scrip_symbol`` using the appropriate model.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        DataFrame of OHLCV data.
    indicator_df : pd.DataFrame
        DataFrame of calculated indicator values aligned with ``ohlcv_df``.
    scrip_symbol : str
        Symbol for which the prediction is requested.
    horizon_key : str
        Key from :data:`config.PREDICTION_HORIZONS` describing the horizon.
    horizon_conf : dict
        Corresponding configuration dictionary.

    Returns
    -------
    The model prediction output or ``None`` if prediction failed.
    """
    model_path = _construct_model_path(scrip_symbol, horizon_conf)
    model = load_model(model_path)
    if model is None:
        logger.error(f"Prediction aborted; model missing for {scrip_symbol} {horizon_key}")
        return None

    # --- Feature Engineering Placeholders ---
    features = pd.concat([ohlcv_df, indicator_df], axis=1)

    # TODO: implement lag features for OHLCV columns
    # for i in range(1, LAG_FEATURES_COUNT + 1):
    #     for col in ["Open", "High", "Low", "Close", "Volume"]:
    #         features[f"{col}_lag_{i}"] = features[col].shift(i)

    # TODO: implement rolling window statistics such as means or std deviations
    # for window in ROLLING_WINDOW_FEATURES_SIZES:
    #     features[f"Close_roll_mean_{window}"] = features["Close"].rolling(window).mean()

    # For now, use the most recent row of features for prediction
    latest_features = features.tail(1).dropna(axis=1)

    try:
        if horizon_conf.get("target_type", "classification").startswith("classification"):
            if hasattr(model, "predict_proba"):
                prediction = model.predict_proba(latest_features)
            else:
                prediction = model.predict(latest_features)
        else:
            prediction = model.predict(latest_features)
        return prediction
    except Exception as exc:  # pragma: no cover - runtime model error
        logger.error(f"Error generating prediction for {scrip_symbol} {horizon_key}: {exc}")
        return None

