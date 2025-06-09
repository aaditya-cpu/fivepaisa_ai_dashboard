# fivepaisa_ai_dashboard/config.py
"""
Centralised, non-secret configuration for the 5paisa AI Dashboard.
------------------------------------------------------------------
* NO credentials live here – put them in `.streamlit/secrets.toml`.
* Anything that might be reused by two different modules belongs here.
* This file is designed to be comprehensive and production-ready.
"""

from __future__ import annotations # For type hinting with forward references (e.g., dict[str, IndicatorConfig])
import os
from datetime import timedelta
# import pandas_ta as ta  # Only imported for reference to ta.Strategy or specific constants if needed.
                         # Actual ta functions will be called in data_handler.py

# =============================================================================
# DIRECTORY & PATH CONFIGURATIONS
# =============================================================================
# Dynamically determines the base directory of the project.
# This ensures paths are constructed correctly regardless of where the script is run from.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

DATA_DIR: str = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")                # For raw OHLCV, scrip master
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")    # For feature-engineered data, model inputs
MODELS_STORE_DIR: str = os.path.join(BASE_DIR, "models_store")   # For serialized trained models
LOG_DIR: str = os.path.join(BASE_DIR, "logs")                    # For application logs

# Ensure all necessary directories exist upon module import.
for d_path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_STORE_DIR, LOG_DIR):
    os.makedirs(d_path, exist_ok=True)

# Scrip master file paths
SCRIPMASTER_CSV_PATH: str = os.path.join(RAW_DATA_DIR, "scrip_master.csv") # Human-readable
SCRIPMASTER_PARQUET_PATH: str = os.path.join(RAW_DATA_DIR, "scrip_master.parquet") # Faster for programmatic access

# =============================================================================
# API & DATA SOURCE CONFIGURATIONS
# =============================================================================
# Valid timeframes for 5paisa API and for UI selection
AVAILABLE_TIMEFRAMES: list[str] = ["1m", "5m", "10m", "15m", "30m", "60m", "1d"]
TIMEFRAME_TO_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "60m": 60, "1d": 24 * 60  # Standard minutes in a day
}
TIMEFRAME_TO_PANDAS_FREQ: dict[str, str] = { # For resampling or period calculations
    "1m": "1min", "5m": "5min", "10m": "10min", "15m": "15min",
    "30m": "30min", "60m": "60min", "1d": "D"
}


DEFAULT_HISTORICAL_TIMEFRAME: str = "1d"
DEFAULT_HISTORICAL_DAYS_FETCH: int = 730  # Approx. 2 years of daily data
MAX_INTRADAY_LOOKBACK_DAYS: int = 30      # Typically, 5paisa API limit for 1-min data

# WebSocket connection parameters
WEBSOCKET_RECONNECT_ATTEMPTS: int = 5
WEBSOCKET_RECONNECT_DELAY_SECONDS: int = 5

# API request rate limiting (soft limits, py5paisa might have its own handling)
REQUESTS_PER_MINUTE_SOFT_LIMIT: int = 90 # Stay below typical 100-200/min limits
HTTP_POLL_INTERVAL_SECONDS: int = 15     # For non-WebSocket data updates if needed

# =============================================================================
# UI & DASHBOARD DEFAULTS
# =============================================================================
APP_TITLE: str = "5paisa AI Predictive Dashboard"
PAGE_ICON: str = "📊" # Favicon for Streamlit app

# Default symbols to attempt to load on startup
DEFAULT_SYMBOL_INDEX: str = "NIFTY"     # E.g., NIFTY 50
DEFAULT_SYMBOL_EQUITY: str = "RELIANCE" # E.g., Reliance Industries

INITIAL_CANDLES_TO_DISPLAY: int = 150   # Number of candles on initial chart load
UI_DATA_REFRESH_INTERVAL_SECONDS: int = 60 # General UI refresh for non-critical data

# Chart export settings
CHART_EXPORT_FORMAT: str = "png"
CHART_EXPORT_SCALE: int = 2 # Multiplier for image resolution

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
# Standardized structure for defining indicators:
# Key (UI display name or internal key): {
#     "label": str, (User-friendly name for UI)
#     "enabled_by_default": bool,
#     "function_name_pd_ta": str, (Function name in pandas_ta library)
#     "params": dict, (Parameters to pass to the pandas_ta function)
#     "output_cols": list[str], (Names of columns pandas_ta will generate; essential for multi-output indicators)
#     "plot_info": {
#         "type": str, ('line', 'bands', 'histogram', 'multi_line', 'macd', 'supertrend', 'psar', 'ichimoku')
#         "on_price_chart": bool, (True if plotted on main price chart, False for subplot)
#         "color": str | list[str] | dict[str,str], (Single color, list for multi_line, or dict for complex plots)
#         "y_label": str, (Label for secondary y-axis if `on_price_chart` is False)
#         "range": list[float, float], (Optional y-axis range for subplots, e.g., [0, 100] for RSI)
#         "hline_y": list[float], (Optional horizontal lines, e.g., [30, 70] for RSI oversold/overbought)
#         "hline_color": list[str],
#         "hline_dash": list[str],
#         # Specific keys for band-type plots
#         "lower_band_col": str, (Column name for lower band, e.g., 'BBL_20_2.0')
#         "middle_band_col": str, (Column name for middle band)
#         "upper_band_col": str, (Column name for upper band)
#         "fill_color": str, (Color for the area between bands)
#         # Specific keys for MACD
#         "macd_line_col": str, "signal_line_col": str, "hist_col": str
#     }
# }

DEFAULT_INDICATORS: dict[str, dict] = {
    # --- Moving Average Family ---
    "SMA_20": {
        "label": "SMA (20)", "enabled_by_default": True, "function_name_pd_ta": "sma", "params": {"length": 20},
        "output_cols": ["SMA_20"],
        "plot_info": {"type": "line", "on_price_chart": True, "color": "royalblue"}
    },
    "SMA_50": {
        "label": "SMA (50)", "enabled_by_default": False, "function_name_pd_ta": "sma", "params": {"length": 50},
        "output_cols": ["SMA_50"],
        "plot_info": {"type": "line", "on_price_chart": True, "color": "orange"}
    },
    "EMA_20": {
        "label": "EMA (20)", "enabled_by_default": False, "function_name_pd_ta": "ema", "params": {"length": 20},
        "output_cols": ["EMA_20"],
        "plot_info": {"type": "line", "on_price_chart": True, "color": "limegreen"}
    },
    "VWAP_D": { # Daily VWAP
        "label": "VWAP (Daily)", "enabled_by_default": False, "function_name_pd_ta": "vwap", "params": {}, # params might need anchor='D' if pandas_ta supports it directly, or handle in data_handler
        "output_cols": ["VWAP_D"], # Ensure pandas_ta output column name matches or adjust
        "plot_info": {"type": "line", "on_price_chart": True, "color": "magenta"}
    },
    # --- Momentum / Oscillators ---
    "RSI_14": {
        "label": "RSI (14)", "enabled_by_default": True, "function_name_pd_ta": "rsi", "params": {"length": 14},
        "output_cols": ["RSI_14"],
        "plot_info": {"type": "line", "on_price_chart": False, "color": "purple", "y_label": "RSI",
                      "range": [0, 100], "hline_y": [30, 70], "hline_color": ["green", "red"], "hline_dash": ["dash", "dash"]}
    },
    "STOCH_14_3_3": { # Stochastic %K, %D
        "label": "Stochastic (14,3,3)", "enabled_by_default": False, "function_name_pd_ta": "stoch", "params": {"k": 14, "d": 3, "smooth_k": 3},
        "output_cols": ["STOCHk_14_3_3", "STOCHd_14_3_3"],
        "plot_info": {"type": "multi_line", "on_price_chart": False, "color": ["dodgerblue", "orange"],
                      "y_label": "Stochastic", "range": [0, 100], "hline_y": [20, 80], "hline_color": ["green", "red"], "hline_dash": ["dash", "dash"]}
    },
    "MACD_12_26_9": {
        "label": "MACD (12,26,9)", "enabled_by_default": False, "function_name_pd_ta": "macd", "params": {"fast": 12, "slow": 26, "signal": 9},
        "output_cols": ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"], # MACD line, MACD Histogram, Signal line
        "plot_info": {"type": "macd", "on_price_chart": False, "y_label": "MACD",
                      "macd_line_col": "MACD_12_26_9", "signal_line_col": "MACDs_12_26_9", "hist_col": "MACDh_12_26_9",
                      "color": {"macd": "blue", "signal": "orange", "hist_positive": "green", "hist_negative": "red"}}
    },
    "ADX_14": {
        "label": "ADX (14)", "enabled_by_default": False, "function_name_pd_ta": "adx", "params": {"length": 14},
        "output_cols": ["ADX_14", "DMP_14", "DMN_14"], # ADX, +DI, -DI
        "plot_info": {"type": "multi_line", "on_price_chart": False, "color": {"ADX": "gold", "+DI": "green", "-DI": "red"},
                      "y_label": "ADX/DI", "lines_to_plot": ["ADX_14"], "hline_y": [25], "hline_color": ["grey"], "hline_dash": ["dot"]} # Typically ADX > 25 indicates trend
    },
    # --- Volatility / Bands ---
    "BBANDS_20_2": {
        "label": "Bollinger Bands (20,2)", "enabled_by_default": False, "function_name_pd_ta": "bbands", "params": {"length": 20, "std": 2},
        "output_cols": ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"], # Lower, Middle, Upper, Bandwidth, %B
        "plot_info": {"type": "bands", "on_price_chart": True,
                      "lower_band_col": "BBL_20_2.0", "middle_band_col": "BBM_20_2.0", "upper_band_col": "BBU_20_2.0",
                      "color": {"middle": "orange", "bands_fill": "rgba(0,100,80,0.15)"}}
    },
    "ATR_14": {
        "label": "ATR (14)", "enabled_by_default": False, "function_name_pd_ta": "atr", "params": {"length": 14},
        "output_cols": ["ATRr_14"], # pandas-ta typically names it ATRp or ATRr for percentage/raw
        "plot_info": {"type": "line", "on_price_chart": False, "color": "teal", "y_label": "ATR"}
    },
    # --- Trend / Stop Systems ---
    "SUPERTREND_10_3": {
        "label": "SuperTrend (10,3)", "enabled_by_default": False, "function_name_pd_ta": "supertrend", "params": {"length": 10, "multiplier": 3},
        "output_cols": ["SUPERT_10_3.0", "SUPERTd_10_3.0", "SUPERTl_10_3.0", "SUPERTs_10_3.0"], # Trend, Direction, Long, Short lines
        "plot_info": {"type": "supertrend", "on_price_chart": True, "trend_col": "SUPERT_10_3.0", "direction_col": "SUPERTd_10_3.0",
                      "color": {"up": "green", "down": "red"}}
    },
    "PSAR_0.02_0.2": { # Parabolic SAR
        "label": "Parabolic SAR (0.02,0.2)", "enabled_by_default": False, "function_name_pd_ta": "psar", "params": {"step": 0.02, "max_step": 0.2},
        "output_cols": ["PSARl_0.02_0.2", "PSARs_0.02_0.2", "PSARaf_0.02_0.2", "PSARr_0.02_0.2"], # Long, Short, Accel Factor, Reversal
        "plot_info": {"type": "psar", "on_price_chart": True, "long_col": "PSARl_0.02_0.2", "short_col": "PSARs_0.02_0.2",
                      "color": {"long": "rgba(0,255,0,0.6)", "short": "rgba(255,0,0,0.6)"}, "marker_size": 5}
    },
    # --- Ichimoku Cloud ---
    "ICHIMOKU_9_26_52": {
        "label": "Ichimoku (9,26,52)", "enabled_by_default": False, "function_name_pd_ta": "ichimoku", "params": {"tenkan": 9, "kijun": 26, "senkou": 52}, # default pandas-ta uses these standard params
        "output_cols": ["ITS_9", "IKS_26", "ISA_9_26_52", "ISB_26_52", "ICS_26"], # Tenkan, Kijun, SenkouA, SenkouB, Chikou Span
        "plot_info": {"type": "ichimoku", "on_price_chart": True,
                      "tenkan_col": "ITS_9", "kijun_col": "IKS_26", "senkou_a_col": "ISA_9_26_52", "senkou_b_col": "ISB_26_52", "chikou_col": "ICS_26",
                      "color": {"tenkan": "blue", "kijun": "red", "chikou": "green", "senkou_a": "rgba(0,255,0,0.2)", "senkou_b": "rgba(255,0,0,0.1)"}}
    }
}

# =============================================================================
# PATTERN DETECTION
# =============================================================================
DETECT_CANDLE_PATTERNS_BY_DEFAULT: bool = True
DETECT_CHART_PATTERNS_BY_DEFAULT: bool = False # Chart patterns are more complex and usually not for real-time streaming

# List of pandas-ta compatible candlestick pattern names.
# `df.ta.cdl_pattern(name="XYZ")` will be used.
CANDLESTICK_PATTERNS_TO_DETECT: list[str] = [
    # Single candle patterns
    "hammer", "hangingman", "invertedhammer", "shootingstar",
    "doji", "dojistar", "dragonflydoji", "gravestonedoji",
    # Two candle patterns
    "engulfing", "harami", "haramicross", "piercing", "darkcloudcover",
    # Three candle patterns
    "morningstar", "eveningstar", "morningdojistar", "eveningdojistar",
    "3whitesoldiers", "3blackcrows", "3inside", "3outside",
    # Others
    "abandonedbaby", "marubozu"
]

PATTERN_MARKER_CONFIG: dict[str, any] = {
    "bullish_color": "rgba(0, 255, 0, 0.7)", "bearish_color": "rgba(255, 0, 0, 0.7)", "neutral_color": "rgba(128, 128, 128, 0.7)",
    "bullish_symbol": "triangle-up", "bearish_symbol": "triangle-down", "neutral_symbol": "circle",
    "size": 8, "y_offset_percentage": 0.02 # Percentage of price range to offset marker below low / above high
}

# =============================================================================
# PREDICTIVE MODELS & MACHINE LEARNING
# =============================================================================
PREDICTION_HORIZONS: dict[str, dict] = {
    "intraday_15m": {"label": "Intraday (Next 15m)", "forward_periods": 1, "source_timeframe": "15m", "target_type": "classification", "model_prefix": "intraday_15m"},
    "intraday_1h":  {"label": "Intraday (Next 1H)",  "forward_periods": 1, "source_timeframe": "60m", "target_type": "classification", "model_prefix": "intraday_1h"},
    "daily_1d":     {"label": "Next Day",            "forward_periods": 1, "source_timeframe": "1d",  "target_type": "classification", "model_prefix": "daily_1d"},
    "weekly_1w":    {"label": "Next Week (5D)",      "forward_periods": 5, "source_timeframe": "1d",  "target_type": "regression",     "model_prefix": "weekly_1w"},
    "monthly_1m":   {"label": "Next Month (20D)",    "forward_periods": 20, "source_timeframe": "1d", "target_type": "regression",     "model_prefix": "monthly_1m"},
    "quarterly_3m": {"label": "Next Quarter (60D)",  "forward_periods": 60, "source_timeframe": "1d", "target_type": "regression", "model_prefix": "quarterly_3m"},
    "yearly_1y":    {"label": "Next Year (252D)",    "forward_periods": 252, "source_timeframe": "1d", "target_type": "classification_long", "model_prefix": "yearly_1y"}
}

# Default model architecture to use for each target_type
DEFAULT_MODEL_ARCHITECTURE: dict[str, str] = {
    "classification": "LightGBM_Classifier",
    "regression": "LightGBM_Regressor",
    "classification_long": "LightGBM_Classifier" # Could be different for very long term
}

# Feature engineering settings
LAG_FEATURES_COUNT: int = 5 # Number of lagged OHLCV features
PRICE_CHANGE_THRESHOLD_FOR_CLASS: float = 0.005  # e.g., +/- 0.5% for UP/DOWN classification vs FLAT
ROLLING_WINDOW_FEATURES_SIZES: list[int] = [5, 10, 20] # For rolling means, stddevs etc.

# --- Model Specific Parameters ---
# LightGBM default parameters
LGBM_COMMON_PARAMS: dict[str, any] = {
    "boosting_type": "gbdt", "n_estimators": 200, "learning_rate": 0.03,
    "num_leaves": 41, "max_depth": -1, # -1 means no limit
    "feature_fraction": 0.85, "bagging_fraction": 0.8, "bagging_freq": 5,
    "lambda_l1": 0.1, "lambda_l2": 0.1, "min_child_samples": 20,
    "verbose": -1, "n_jobs": -1, "seed": 42
}
LGBM_PARAMS_CLASSIFIER: dict[str, any] = {
    **LGBM_COMMON_PARAMS, # Unpack common params
    "objective": "multiclass", "metric": "multi_logloss", "num_class": 3, # UP, DOWN, FLAT
}
LGBM_PARAMS_REGRESSOR: dict[str, any] = {
    **LGBM_COMMON_PARAMS, # Unpack common params
    "objective": "regression_l1", "metric": "rmse", # L1 (MAE) for objective, RMSE for metric
}

# CatBoost default parameters (for future use)
CATBOOST_COMMON_PARAMS: dict[str, any] = {
    "iterations": 400, "learning_rate": 0.05, "depth": 6,
    "l2_leaf_reg": 3, "random_seed": 42, "verbose": 0,
    "early_stopping_rounds": 20
}
CATBOOST_PARAMS_CLASSIFIER: dict[str, any] = {
    **CATBOOST_COMMON_PARAMS,
    "loss_function": "MultiClass", "eval_metric": "MultiClass"
}
CATBOOST_PARAMS_REGRESSOR: dict[str, any] = {
    **CATBOOST_COMMON_PARAMS,
    "loss_function": "MAE", "eval_metric": "RMSE"
}

# Optuna hyperparameter optimization settings
OPTUNA_N_TRIALS: int = 30 # Number of trials for optimization (can be increased for more thorough search)
OPTUNA_CV_SPLITS: int = 5  # Number of splits for TimeSeriesSplit
OPTUNA_TIMEOUT_SECONDS: int = 60 * 30 # Max 30 minutes per Optuna study

# Model artifact naming convention
MODEL_FILENAME_FORMAT: str = "model_{scrip_symbol}_{horizon_prefix}_{model_arch}.joblib" # Using joblib for sklearn-compat models

# =============================================================================
# RISK MANAGEMENT & TRADING LOGIC (Placeholders for future trading integration)
# =============================================================================
MAX_POSITION_RISK_PER_TRADE_PCT: float = 0.01  # Max 1% of (hypothetical) capital at risk per trade
DAILY_MAX_DRAWDOWN_PCT: float = 0.05           # Max 5% daily loss before halting (hypothetical) trading
DEFAULT_STOP_LOSS_ATR_MULTIPLIER: float = 2.0  # Default SL as 2x ATR
DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER: float = 3.0 # Default TP as 3x ATR

# =============================================================================
# ALERTING & NOTIFICATIONS (Placeholders)
# =============================================================================
ENABLE_EMAIL_ALERTS: bool = False
EMAIL_ALERT_RECIPIENTS: list[str] = ["your_email@example.com"]
ENABLE_SLACK_ALERTS: bool = False
SLACK_WEBHOOK_URL: str = "YOUR_SLACK_WEBHOOK_URL_HERE"
ALERT_THROTTLE_SECONDS: int = 300 # Min 5 minutes between similar alerts for the same scrip

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL: str = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(module)-15s:%(lineno)-4d | %(message)s"
LOG_FILE_PATH: str = os.path.join(LOG_DIR, "dashboard_app.log")
LOG_MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5 # Number of old log files to keep

# =============================================================================
# MISCELLANEOUS & APPLICATION WIDE CONSTANTS
# =============================================================================
# Default watchlist for easy access in UI (symbols need mapping to ScripCodes)
DEFAULT_WATCHLIST_SYMBOLS: list[str] = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY",
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "ITC"
]

# Standard OHLCV column names to ensure consistency across modules
OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume", "Datetime"]
# Datetime column name from 5paisa API might be 'Time'. Need to standardize in data_handler.

# Small epsilon value for float comparisons (e.g., checking if price touched a level)
FLOAT_COMPARISON_EPSILON: float = 1e-9

# =============================================================================
# UTILITY FUNCTION FOR CONFIG ACCESS (Optional convenience)
# =============================================================================
_CONFIG_GLOBALS = globals() # Capture all global variables defined in this file

def get_config_value(path: str, default: any = None) -> any:
    """
    Retrieves a configuration value using a dot-separated path.
    Example: get_config_value("LGBM_COMMON_PARAMS.learning_rate")
    """
    keys = path.split(".")
    value = _CONFIG_GLOBALS
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif hasattr(value, key): # Should not happen with dicts, but for module-level vars
            value = getattr(value, key)
        else:
            return default
    return value

# =============================================================================
# SELF-TEST / DIAGNOSTIC (when run directly: python config.py)
# =============================================================================
if __name__ == "__main__":
    print("--- Configuration File Self-Test ---")
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Models Storage Directory: {MODELS_STORE_DIR}")
    print(f"Default ScripMaster CSV Path: {SCRIPMASTER_CSV_PATH}")

    print("\n--- Default Enabled Indicators ---")
    for key, conf in DEFAULT_INDICATORS.items():
        if conf.get("enabled_by_default", False):
            print(f"- {conf.get('label', key)} (Function: {conf.get('function_name_pd_ta')})")

    print("\n--- Prediction Horizons ---")
    for key, conf in PREDICTION_HORIZONS.items():
        print(f"- {conf.get('label', key)}: Source TF {conf['source_timeframe']}, Target {conf['target_type']}, Model Prefix {conf['model_prefix']}")

    print(f"\nLGBM Classifier Default Learning Rate: {LGBM_PARAMS_CLASSIFIER.get('learning_rate')}")
    print(f"Test get_config_value('LGBM_PARAMS_REGRESSOR.n_estimators'): {get_config_value('LGBM_PARAMS_REGRESSOR.n_estimators')}")
    print(f"Test get_config_value('NON_EXISTENT.key', 'Not Found'): {get_config_value('NON_EXISTENT.key', 'Not Found')}")
    print("\n--- Configuration Self-Test Complete ---")