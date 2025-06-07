# fivepaisa_ai_dashboard/utils.py

"""
Utility Module

This file contains general-purpose helper functions and classes used across
the 5paisa AI Dashboard application. This includes:
- Logging setup and configuration.
- Date and time manipulation utilities.
- Simple data validation or transformation helpers.
- Other miscellaneous helper functions.
"""

import logging
import logging.handlers
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union, Dict
import pandas as pd
import re

# Import configurations for logging paths and levels
from config import (
    APP_TITLE, LOG_LEVEL, LOG_FORMAT, LOG_FILE_PATH,
    LOG_MAX_FILE_SIZE_BYTES, LOG_BACKUP_COUNT
)

# --- Global Logger Setup ---
# We'll set up a root logger or specific application logger here.
# This function can be called once from app.py at startup.

_app_logger_configured = False # Flag to ensure setup_logging is called only once

def setup_global_logging(
    app_name: str = APP_TITLE,
    level: str = LOG_LEVEL,
    log_format: str = LOG_FORMAT,
    log_file: Optional[str] = LOG_FILE_PATH,
    max_bytes: int = LOG_MAX_FILE_SIZE_BYTES,
    backup_count: int = LOG_BACKUP_COUNT
) -> logging.Logger:
    """
    Configures a global application logger with console and optional file output.

    Args:
        app_name (str): The base name for the logger.
        level (str): The logging level (e.g., "INFO", "DEBUG").
        log_format (str): The format string for log messages.
        log_file (Optional[str]): Path to the log file. If None, only console logging.
        max_bytes (int): Maximum size of the log file before rotation.
        backup_count (int): Number of backup log files to keep.

    Returns:
        logging.Logger: The configured application root logger.
    """
    global _app_logger_configured
    if _app_logger_configured:
        return logging.getLogger(app_name) # Return existing logger if already configured

    logger_instance = logging.getLogger(app_name)
    logger_instance.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Prevent propagation to root logger if we want isolated app logging
    # logger_instance.propagate = False 

    formatter = logging.Formatter(log_format)

    # Console Handler (always add for Streamlit visibility and general dev)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger_instance.handlers):
        logger_instance.addHandler(console_handler)

    # File Handler (optional, based on config)
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir): # Check if log_dir is not empty string
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger_instance.handlers):
                logger_instance.addHandler(file_handler)
            logger_instance.info(f"File logging configured at: {log_file}")
        except Exception as e:
            logger_instance.error(f"Failed to configure file logging for {log_file}: {e}", exc_info=True)

    logger_instance.info(f"Global logging for '{app_name}' initialized at level {level}.")
    _app_logger_configured = True
    return logger_instance

# Initialize the logger when this module is imported for the first time.
# Other modules can then get this logger by name.
# Example: `logger = logging.getLogger(APP_TITLE)`
# Or, they can get a child logger: `logger = logging.getLogger(f"{APP_TITLE}.ModuleName")`
# If app.py calls setup_global_logging() at startup, that will configure the main logger.
# For now, individual modules like data_handler create their own loggers, which is fine.
# This function is here if a more centralized approach is desired.


# --- Date/Time Utility Functions ---

def format_datetime_for_api(dt_object: datetime, api_format: str = "%Y-%m-%d") -> str:
    """
    Formats a datetime object into a string suitable for API requests.

    Args:
        dt_object (datetime): The datetime object to format.
        api_format (str): The desired string format (e.g., "%Y-%m-%d", "%Y%m%dT%H%M").

    Returns:
        str: Formatted datetime string.
    """
    try:
        return dt_object.strftime(api_format)
    except Exception as e:
        logger = logging.getLogger(APP_TITLE) # Get app logger
        logger.error(f"Error formatting datetime object {dt_object} to format {api_format}: {e}")
        # Fallback to a common format or raise error
        return dt_object.strftime("%Y-%m-%d") # Default fallback


def get_date_range_for_fetch(days_back: int, end_date: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """
    Calculates a start and end date range given number of days back.

    Args:
        days_back (int): Number of days to go back from the end_date.
        end_date (Optional[datetime]): The end date of the range. Defaults to now (UTC).

    Returns:
        tuple[datetime, datetime]: (start_date, end_date)
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc) # Use timezone-aware datetime

    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def parse_api_datetime_string(
    date_string: str,
    expected_formats: Optional[list[str]] = None
) -> Optional[datetime]:
    """
    Parses a date/time string from an API response into a datetime object.
    Tries a list of common formats if `expected_formats` is not provided.

    Args:
        date_string (str): The date string to parse.
        expected_formats (Optional[list[str]]): A list of strptime format strings to try.

    Returns:
        Optional[datetime]: Parsed datetime object, or None if parsing fails.
    """
    if not date_string or not isinstance(date_string, str):
        return None

    if expected_formats is None:
        # Common formats used by APIs (add more as encountered)
        expected_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO 8601 with milliseconds and Zulu
            "%Y-%m-%dT%H:%M:%SZ",     # ISO 8601 with Z Suffix
            "%Y-%m-%d %H:%M:%S.%f",   # Common SQL like with milliseconds
            "%Y-%m-%d %H:%M:%S",      # Common SQL like
            "%Y-%m-%d",               # Date only
            "%Y%m%dT%H%M",            # Alpha Vantage news time_from/time_to
            "%Y%m%d",                 # Date only compact
            "%b %d %Y %I:%M%p",       # E.g., "Jan 01 2023 03:30PM"
            "%d/%m/%Y %H:%M:%S",      # DD/MM/YYYY HH:MM:SS
            "%m/%d/%Y %H:%M:%S",      # MM/DD/YYYY HH:MM:SS
        ]

    for fmt in expected_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue # Try next format
    
    logger = logging.getLogger(APP_TITLE)
    logger.warning(f"Failed to parse datetime string '{date_string}' with any of the provided/default formats.")
    return None

# --- Data Cleaning and Validation Utilities ---

def sanitize_string(input_str: Any, allow_alphanumeric_only: bool = False) -> str:
    """
    Sanitizes a string by stripping whitespace.
    Optionally removes all non-alphanumeric characters (except spaces if not strictly alphanumeric).

    Args:
        input_str (Any): The input to sanitize, will be converted to string.
        allow_alphanumeric_only (bool): If True, removes non-alphanumeric characters.

    Returns:
        str: The sanitized string.
    """
    if input_str is None:
        return ""
    
    s = str(input_str).strip()
    
    if allow_alphanumeric_only:
        # Remove non-alphanumeric, but keep spaces by default unless a stricter regex is needed
        s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip() # Normalize multiple spaces to single
    return s


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely converts a value to a float. If conversion fails, returns a default value.
    Handles strings like "N/A", "None", "-", empty strings, etc.

    Args:
        value: The value to convert.
        default (float): The default value to return on conversion failure.

    Returns:
        float: The converted float or the default.
    """
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ["na", "n/a", "none", "-", ""]:
            return default
        try:
            # Remove commas for numbers like "1,234.56"
            return float(value.replace(",", ""))
        except ValueError:
            return default
    try: # Try a generic conversion for other types
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely converts a value to an integer.

    Args:
        value: The value to convert.
        default (int): The default value to return on conversion failure.

    Returns:
        int: The converted integer or the default.
    """
    float_val = safe_float_conversion(value, default=float('nan'))
    if pd.isna(float_val):
        return default
    return int(float_val)


def flatten_nested_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flattens a nested dictionary.
    Example: {'a': 1, 'b': {'c': 2, 'd': 3}} -> {'a': 1, 'b_c': 2, 'b_d': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Option 1: Convert list to string
            # items.append((new_key, str(v)))
            # Option 2: Enumerate list items if they are simple or dicts themselves
            for i, item in enumerate(v):
                list_item_key = f"{new_key}{sep}{i}"
                if isinstance(item, dict):
                    items.extend(flatten_nested_dict(item, list_item_key, sep=sep).items())
                else:
                    items.append((list_item_key, item))
        else:
            items.append((new_key, v))
    return dict(items)


# --- Miscellaneous Utilities ---
def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Generates a simple cache key string from arguments and keyword arguments.
    Useful for custom caching logic if Streamlit's @st.cache_data/resource is not suitable.
    """
    # Sort kwargs for consistency
    sorted_kwargs = sorted(kwargs.items())
    key_parts = [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted_kwargs]
    return "_".join(key_parts)


# --- Test Suite for Utility Functions ---
if __name__ == "__main__":
    import streamlit as st
    from datetime import datetime
    # This block executes only when utils.py is run directly.
    st.set_page_config(page_title="Utilities Module Test", layout="centered")
    st.title("🔧 Utilities Module Test Suite")
    st.markdown("---")

    # Initialize global logger for testing purposes
    test_logger = setup_global_logging(level="DEBUG", log_file=None) # Console only for test

    st.header("Logging Test")
    test_logger.debug("This is a DEBUG message from utils test.")
    test_logger.info("This is an INFO message from utils test.")
    test_logger.warning("This is a WARNING message from utils test.")
    test_logger.error("This is an ERROR message from utils test.")
    st.success("Logging messages should appear in your console if setup is correct.")
    st.markdown("---")

    st.header("Date/Time Utilities Test")
    now_dt = datetime.now()
    st.write(f"Current datetime: `{now_dt}`")
    
    formatted_date_api = format_datetime_for_api(now_dt)
    st.write(f"Formatted for API (default YYYY-MM-DD): `{formatted_date_api}`")
    
    formatted_datetime_compact = format_datetime_for_api(now_dt, "%Y%m%dT%H%M%S")
    st.write(f"Formatted for API (compact YYYYMMDDTHHMMSS): `{formatted_datetime_compact}`")

    start_range, end_range = get_date_range_for_fetch(days_back=7, end_date=now_dt)
    st.write(f"Date range for 7 days back from now: Start=`{start_range}`, End=`{end_range}`")

    test_date_str1 = "2023-10-26T10:30:00Z"
    parsed_dt1 = parse_api_datetime_string(test_date_str1)
    st.write(f"Parsing '{test_date_str1}': `{parsed_dt1}` (Type: {type(parsed_dt1)})")
    
    test_date_str2 = "20231115T1445"
    parsed_dt2 = parse_api_datetime_string(test_date_str2)
    st.write(f"Parsing '{test_date_str2}': `{parsed_dt2}` (Type: {type(parsed_dt2)})")

    test_date_str3 = "Invalid Date String"
    parsed_dt3 = parse_api_datetime_string(test_date_str3)
    st.write(f"Parsing '{test_date_str3}': `{parsed_dt3}` (Should be None)")
    st.markdown("---")

    st.header("Data Cleaning & Validation Test")
    str_to_sanitize = "  Extra   Spaces &*!@# Special Chars  "
    st.write(f"Original string: `'{str_to_sanitize}'`")
    st.write(f"Sanitized (spaces only): `'{sanitize_string(str_to_sanitize)}'`")
    st.write(f"Sanitized (alphanumeric only): `'{sanitize_string(str_to_sanitize, allow_alphanumeric_only=True)}'`")
    
    st.write(f"Safe float for '1,234.56': `{safe_float_conversion('1,234.56')}`")
    st.write(f"Safe float for 'N/A': `{safe_float_conversion('N/A', default=-1.0)}`")
    st.write(f"Safe float for 100: `{safe_float_conversion(100)}`")
    st.write(f"Safe float for None: `{safe_float_conversion(None)}`")

    st.write(f"Safe int for '123.45': `{safe_int_conversion('123.45')}`")
    st.write(f"Safe int for 'Invalid': `{safe_int_conversion('Invalid', default=-99)}`")
    st.markdown("---")

    st.header("Dictionary Flattening Test")
    nested_dict_example = {
        'name': 'Test Corp',
        'year': 2023,
        'financials': {
            'revenue': 1000000,
            'profit': {'gross': 200000, 'net': 100000}
        },
        'tags': ['finance', 'tech', {'type': 'public'}]
    }
    st.write("Original Nested Dict:")
    st.json(nested_dict_example)
    flattened_dict = flatten_nested_dict(nested_dict_example)
    st.write("Flattened Dict:")
    st.json(flattened_dict)
    st.markdown("---")

    st.header("Cache Key Generation Test")
    key1 = generate_cache_key("func_name", "symbol_A", timeframe="1d", param_x=10)
    key2 = generate_cache_key("func_name", "symbol_A", param_x=10, timeframe="1d") # Test kwargs order
    st.write(f"Generated Key 1: `{key1}`")
    st.write(f"Generated Key 2 (same params, different kwarg order): `{key2}`")
    st.write(f"Keys should be identical: {key1 == key2}")

    st.success("Utility function tests complete.")