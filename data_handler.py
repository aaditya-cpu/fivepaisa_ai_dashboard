# fivepaisa_ai_dashboard/data_handler.py

"""
Data Handling Module

Responsible for:
- Fetching Scrip Master and historical OHLCV data from 5paisa.
- Fetching fundamental data, news, and market intelligence from Alpha Vantage.
- Calculating technical indicators using pandas-ta.
- Caching data to optimize performance and respect API limits.
- Preparing data for display and model input.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from py5paisa import FivePaisaClient
import pandas_ta as ta

# Import configurations
from config import (
    APP_TITLE, RAW_DATA_DIR, PROCESSED_DATA_DIR, SCRIPMASTER_CSV_PATH, SCRIPMASTER_PARQUET_PATH,
    DEFAULT_HISTORICAL_DAYS_FETCH, MAX_INTRADAY_LOOKBACK_DAYS, AVAILABLE_TIMEFRAMES,
    DEFAULT_INDICATORS, OHLCV_COLUMNS, TIMEFRAME_TO_PANDAS_FREQ,
    ALPHA_VANTAGE_BASE_URL, ALPHA_VANTAGE_API_KEY_SECRET_NAME,
    ALPHA_VANTAGE_REQUESTS_PER_MINUTE_LIMIT, ALPHA_VANTAGE_REQUESTS_PER_DAY_LIMIT, # For future rate limiting
    ALPHA_VANTAGE_CACHE_DIR, ALPHA_VANTAGE_FUNCTIONS
)

# Configure logger for this module
logger = logging.getLogger(f"{APP_TITLE}.DataHandler")
# Basic logging configuration (can be expanded in a central logging setup if needed)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Alpha Vantage API Helper Class ---

class AlphaVantageAPI:
    """
    Helper class to interact with the Alpha Vantage API,
    including API key management, request construction, caching, and basic error handling.
    """
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = st.secrets.get(ALPHA_VANTAGE_API_KEY_SECRET_NAME)

        if not self.api_key:
            msg = "Alpha Vantage API key not found in Streamlit secrets."
            logger.error(msg)
            # We might not want to raise an exception here, but rather let individual fetch functions fail gracefully.
            # Or, app.py can check for this key upfront.
            # For now, functions using this class will check self.api_key.

        self.base_url = ALPHA_VANTAGE_BASE_URL
        # For more sophisticated rate limiting (if not handled by a library like 'ratelimit')
        self.request_timestamps: List[datetime] = []


    def _get_cache_filepath(self, symbol: str, function_name: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Constructs a cache filepath for Alpha Vantage data."""
        symbol_dir = os.path.join(ALPHA_VANTAGE_CACHE_DIR, symbol.upper().replace(":", "_")) # Sanitize for dir name
        os.makedirs(symbol_dir, exist_ok=True)

        # Create a stable filename based on function and key params
        filename_parts = [function_name]
        if params:
            # Sort params to ensure consistent filename for same query
            sorted_params = sorted(params.items())
            for k, v in sorted_params:
                # Exclude generic/dynamic params from filename if they don't define the core query content
                if k.lower() not in ['apikey', 'function', 'symbol', 'datatype', 'tickers', 'topics', 'time_from', 'time_to', 'limit', 'sort']:
                    filename_parts.append(f"{k}_{v}")
                elif k.lower() in ['tickers', 'topics']: # For these, include the value if not too long
                    filename_parts.append(f"{k}_{str(v)[:20]}") # Truncate long values
        filename = "_".join(filename_parts) + ".json" # Default to .json, will be changed for CSV text
        return os.path.join(symbol_dir, filename)

    def _read_from_cache(self, filepath: str, cache_ttl_hours: int = 24) -> Optional[Any]: # Return type Any for text or dict
        """Reads data from cache if file exists and is not too old."""
        if os.path.exists(filepath):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if datetime.now() - file_mod_time < timedelta(hours=cache_ttl_hours):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        # If it's a CSV text cache, it's just text. Otherwise, try JSON.
                        if filepath.endswith(".csv_text"):
                            content = f.read()
                            logger.info(f"CSV text cache hit for {filepath}")
                            return content # Return raw text
                        else:
                            data = json.load(f)
                            logger.info(f"JSON cache hit for {filepath}")
                            return data
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted JSON cache file: {filepath}. Will re-fetch.")
                    os.remove(filepath) # Remove corrupted file
                except Exception as e:
                    logger.error(f"Error reading cache file {filepath}: {e}")
        return None

    def _write_to_cache(self, filepath: str, data: Union[Dict[str, Any], str]):
        """Writes data to a cache file. Data can be dict (for JSON) or str (for raw text)."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, str): # Raw text (e.g. CSV content)
                    f.write(data)
                else: # Dictionary for JSON
                    json.dump(data, f, indent=4)
            logger.info(f"Data cached to {filepath}")
        except Exception as e:
            logger.error(f"Error writing to cache file {filepath}: {e}")

    def _make_request(self, params: Dict[str, Any], symbol_for_cache: Optional[str] = None,
                      cache_ttl_hours: int = 24, is_csv_endpoint: bool = False) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
        """
        Makes a request to Alpha Vantage, handling caching and basic errors.
        If is_csv_endpoint is True, expects and parses CSV.
        """
        if not self.api_key:
            st.error("Alpha Vantage API Key is not configured. Cannot fetch data.")
            logger.error("Alpha Vantage API Key missing for _make_request.")
            return None

        params["apikey"] = self.api_key
        function_name = params.get("function", "UNKNOWN_FUNCTION")
        
        cache_symbol_key = symbol_for_cache if symbol_for_cache else params.get("symbol", "GLOBAL_DATA")
        
        cache_filepath = self._get_cache_filepath(cache_symbol_key, function_name, params)
        
        if is_csv_endpoint:
            cache_filepath = cache_filepath.replace(".json", ".csv_text")

        cached_content = self._read_from_cache(cache_filepath, cache_ttl_hours)
        if cached_content:
            if is_csv_endpoint:
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(cached_content)) # cached_content is string here
                    logger.info(f"Parsed CSV from cache for {function_name} of {cache_symbol_key}")
                    return df
                except Exception as e:
                    logger.error(f"Error parsing cached CSV for {function_name} of {cache_symbol_key}: {e}")
            else: # JSON data
                return cached_content # This is already a dict

        now = datetime.now()
        self.request_timestamps = [t for t in self.request_timestamps if now - t < timedelta(days=1)]
        if len(self.request_timestamps) >= ALPHA_VANTAGE_REQUESTS_PER_DAY_LIMIT -1 :
            minute_requests = [t for t in self.request_timestamps if now - t < timedelta(minutes=1)]
            if len(minute_requests) >= ALPHA_VANTAGE_REQUESTS_PER_MINUTE_LIMIT -1 :
                logger.warning(f"Alpha Vantage API rate limit potentially approaching/exceeded. Last minute requests: {len(minute_requests)}.")
                st.warning("Alpha Vantage API rate limit might be approaching. Please try again later.")
                time.sleep(10) # Longer pause if hitting limits frequently

        logger.info(f"Fetching from Alpha Vantage: {function_name} for {params.get('symbol', 'N/A') if 'symbol' in params else params.get('tickers', 'N/A')}")
        try:
            response = requests.get(self.base_url, params=params, timeout=30) # Increased timeout for potentially large data
            response.raise_for_status()
            self.request_timestamps.append(datetime.now())

            if is_csv_endpoint:
                csv_text = response.text
                # A more robust check for CSV errors
                if response.headers.get('Content-Type', '').lower().startswith('application/json') or \
                   csv_text.strip().startswith('{') and "Error Message" in csv_text:
                    try:
                        error_json = json.loads(csv_text)
                        err_msg = error_json.get("Error Message", "Unknown JSON error in CSV endpoint.")
                        logger.error(f"Alpha Vantage returned JSON error for CSV endpoint {function_name}: {err_msg}")
                        st.error(f"API Error ({function_name}): {err_msg}")
                    except json.JSONDecodeError:
                        logger.error(f"Alpha Vantage returned non-CSV content for {function_name}: {csv_text[:200]}")
                        st.error(f"Error fetching CSV data for {function_name}. Unexpected format.")
                    return None
                
                self._write_to_cache(cache_filepath, csv_text)
                from io import StringIO
                data_df = pd.read_csv(StringIO(csv_text))
                return data_df
            else:
                data = response.json()
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error for {function_name}: {data['Error Message']}")
                    st.error(f"API Error ({function_name}): {data['Error Message']}")
                    return None
                if "Information" in data and ("API call frequency" in data["Information"] or "Thank you for using Alpha Vantage" in data["Information"]): # Check for rate limit messages
                    logger.warning(f"Alpha Vantage rate limit or info message: {data['Information']}")
                    st.warning(f"Alpha Vantage API Info: {data['Information']}. This might indicate a rate limit or usage note.")
                    # If it's purely informational but contains data, we can still cache and return it.
                    # If it's a hard rate limit error without data, then returning None is appropriate.
                    # The previous check for "Error Message" should catch actual errors.
                    # For now, we assume "Information" might still contain valid data if "Error Message" is not present.
                
                self._write_to_cache(cache_filepath, data)
                return data

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred with Alpha Vantage ({function_name}): {http_err} - Response: {http_err.response.text[:200] if http_err.response else 'N/A'}")
            st.error(f"HTTP error fetching data for {function_name}: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error with Alpha Vantage ({function_name}): {conn_err}")
            st.error(f"Connection error. Please check your internet connection.")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error with Alpha Vantage ({function_name}): {timeout_err}")
            st.error(f"Request timed out for {function_name}.")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"General request error with Alpha Vantage ({function_name}): {req_err}")
            st.error(f"An error occurred while fetching data for {function_name}.")
        except json.JSONDecodeError as json_err: # Only for non-CSV endpoints
            logger.error(f"Error decoding JSON response from Alpha Vantage ({function_name}): {json_err}. Response text: {response.text[:200] if 'response' in locals() and response else 'N/A'}")
            st.error(f"Invalid data format received for {function_name}.")
        except Exception as e:
            logger.critical(f"Unexpected error fetching Alpha Vantage data ({function_name}): {e}", exc_info=True)
            st.error(f"An unexpected error occurred fetching {function_name}. Check logs.")
        return None

    # --- Existing Specific Alpha Vantage Data Fetching Functions ---
    def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        params = {"function": "OVERVIEW", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=7*24)

    def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        params = {"function": "INCOME_STATEMENT", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=30*24)

    def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        params = {"function": "BALANCE_SHEET", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=30*24)

    def get_cash_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        params = {"function": "CASH_FLOW", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=30*24)

    def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        params = {"function": "EARNINGS", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=30*24)

    def get_dividends(self, symbol: str) -> Optional[Dict[str, Any]]: # Changed name to match AV doc
        params = {"function": "DIVIDENDS", "symbol": symbol}
        # This is documented as JSON, but its structure might be simple list or dict.
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=30*24)

    def get_splits(self, symbol: str) -> Optional[Dict[str, Any]]: # Changed name to match AV doc
        params = {"function": "SPLITS", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=90*24)

    def get_news_sentiment(self, tickers: Optional[str] = None, topics: Optional[str] = None,
                           time_from: Optional[str] = None, time_to: Optional[str] = None,
                           sort: str = "LATEST", limit: int = 50) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {"function": "NEWS_SENTIMENT", "sort": sort, "limit": limit}
        if tickers: params["tickers"] = tickers
        if topics: params["topics"] = topics
        if time_from: params["time_from"] = time_from
        if time_to: params["time_to"] = time_to
        
        cache_key_symbol_parts = []
        if tickers: cache_key_symbol_parts.append(f"t_{tickers}")
        if topics: cache_key_symbol_parts.append(f"p_{topics}")
        if not cache_key_symbol_parts: cache_key_symbol_parts.append("global")
        cache_key_symbol = "_".join(cache_key_symbol_parts).replace(",","_").replace(":","_")[:50] # Ensure filename friendly

        return self._make_request(params, symbol_for_cache=cache_key_symbol, cache_ttl_hours=1)

    def get_earnings_calendar(self, symbol: Optional[str] = None, horizon: str = "3month") -> Optional[pd.DataFrame]:
        params: Dict[str, Any] = {"function": "EARNINGS_CALENDAR", "horizon": horizon}
        if symbol: params["symbol"] = symbol
        cache_key_symbol = symbol if symbol else "GLOBAL_EARNINGS_CALENDAR"
        return self._make_request(params, symbol_for_cache=cache_key_symbol, cache_ttl_hours=12, is_csv_endpoint=True)

    def get_ipo_calendar(self) -> Optional[pd.DataFrame]:
        params: Dict[str, Any] = {"function": "IPO_CALENDAR"}
        return self._make_request(params, symbol_for_cache="GLOBAL_IPO_CALENDAR", cache_ttl_hours=12, is_csv_endpoint=True)

    # --- NEWLY IMPLEMENTED Alpha Vantage Functions ---

    def get_listing_status(self, date: Optional[str] = None, state: str = "active") -> Optional[pd.DataFrame]:
        """
        Fetches active or delisted US stocks and ETFs.
        Args:
            date (Optional[str]): YYYY-MM-DD. If None, latest trading day.
            state (str): "active" or "delisted".
        Returns:
            DataFrame or None. This is a CSV endpoint.
        """
        params: Dict[str, Any] = {"function": "LISTING_STATUS", "state": state}
        if date:
            params["date"] = date
        
        cache_key_symbol = f"LISTING_{state}_{date if date else 'LATEST'}"
        return self._make_request(params, symbol_for_cache=cache_key_symbol, cache_ttl_hours=24, is_csv_endpoint=True)

    def get_top_gainers_losers(self) -> Optional[Dict[str, Any]]:
        """
        Fetches top 20 gainers, losers, and most actively traded US tickers.
        Returns:
            Dict (JSON response) or None.
        """
        params: Dict[str, Any] = {"function": "TOP_GAINERS_LOSERS"}
        return self._make_request(params, symbol_for_cache="GLOBAL_TOP_MOVERS", cache_ttl_hours=6) # Update a few times a day if needed

    def get_insider_transactions(self, symbol: str) -> Optional[Dict[str, Any]]: # API doc example uses IBM, seems JSON
        """
        Fetches latest and historical insider transactions for a company.
        The example URL in AV docs was for EARNINGS_CALL_TRANSCRIPT but pointed to function=INSIDER_TRANSACTIONS.
        Assuming function=INSIDER_TRANSACTIONS based on title.
        Returns:
            Dict (JSON response) or None.
        """
        params: Dict[str, Any] = {"function": "INSIDER_TRANSACTIONS", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=24)

    def get_etf_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches key ETF metrics and holdings.
        Returns:
            Dict (JSON response) or None.
        """
        params: Dict[str, Any] = {"function": "ETF_PROFILE", "symbol": symbol}
        return self._make_request(params, symbol_for_cache=symbol, cache_ttl_hours=7*24) # ETF profiles don't change daily

    def get_earnings_call_transcript(self, symbol: str, quarter: str) -> Optional[Dict[str, Any]]:
        """
        Fetches earnings call transcript for a given company and quarter.
        Args:
            symbol (str): Ticker symbol.
            quarter (str): Fiscal quarter in YYYYQM format (e.g., "2024Q1").
        Returns:
            Dict (JSON response) or None.
        """
        params: Dict[str, Any] = {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": symbol, "quarter": quarter}
        cache_key_symbol = f"{symbol}_TRANSCRIPT_{quarter}"
        return self._make_request(params, symbol_for_cache=cache_key_symbol, cache_ttl_hours=90*24) # Transcripts are static

    # --- Advanced Analytics (placeholders, as they are more complex) ---
    # def get_analytics_fixed_window(self, symbols_list: List[str], range_start: str, range_end: str,
    #                                interval: str, ohlc_field: str = "close",
    #                                calculations: List[str]) -> Optional[Dict[str, Any]]:
    #     params = {
    #         "function": "ANALYTICS_FIXED_WINDOW",
    #         "SYMBOLS": ",".join(symbols_list),
    #         "RANGE": [range_start, range_end], # Needs to be passed as two RANGE params in URL
    #         "INTERVAL": interval,
    #         "OHLC": ohlc_field,
    #         "CALCULATIONS": ",".join(calculations)
    #     }
    #     # Special handling for multiple RANGE params if requests library doesn't do it automatically
    #     # Or construct URL string manually for this one.
    #     # For now, placeholder.
    #     logger.warning("ANALYTICS_FIXED_WINDOW not fully implemented due to multiple RANGE params.")
    #     return None

    # def get_analytics_sliding_window(self, symbols_list: List[str], range_def: str, interval: str,
    #                                  window_size: int, calculations: List[str],
    #                                  ohlc_field: str = "close") -> Optional[Dict[str, Any]]:
    #     params = {
    #         "function": "ANALYTICS_SLIDING_WINDOW",
    #         "SYMBOLS": ",".join(symbols_list),
    #         "RANGE": range_def,
    #         "INTERVAL": interval,
    #         "WINDOW_SIZE": window_size,
    #         "CALCULATIONS": ",".join(calculations),
    #         "OHLC": ohlc_field
    #     }
    #     logger.warning("ANALYTICS_SLIDING_WINDOW not fully implemented.")
    #     return None

# ... (rest of the data_handler.py: Global AV client, 5paisa functions, indicator calculation, if __name__ block) ...

# Ensure the global instance is defined after the class
alpha_vantage_client = AlphaVantageAPI()




# --- 5paisa Data Functions ---

@st.cache_data(ttl=3600 * 6, show_spinner="Fetching 5paisa Scrip Master...") # Cache for 6 hours
def fetch_scrip_master(client: FivePaisaClient) -> Optional[pd.DataFrame]:
    """
    Retrieve the full Scrip-Master, validate it, cache locally, and return a DataFrame.
    """
    if not client:
        logger.error("5paisa client not authenticated for fetching scrip master.")
        return None

    try:
        # Return cached parquet if valid
        if os.path.exists(SCRIPMASTER_PARQUET_PATH):
            df_cached = pd.read_parquet(SCRIPMASTER_PARQUET_PATH)
            req_cols = ["ScripCode", "Symbol", "FullName", "Exch", "ExchType", "TickSize"]
            if not df_cached.empty and all(c in df_cached.columns for c in req_cols):
                logger.info("Loaded Scrip Master from local cache.")
                return df_cached
            logger.warning("Cached Scrip Master missing columns – refetching.")

        logger.info("Fetching Scrip Master from 5paisa API…")
        scrips_data = client.get_scrips()
        if not isinstance(scrips_data, list) or not scrips_data:
            st.error("Error fetching Scrip Master from 5paisa.")
            logger.error("Empty / invalid Scrip Master data.")
            return None

        df_scrips = pd.DataFrame(scrips_data)
        req_cols = ["ScripCode", "Symbol", "FullName", "Exch", "ExchType", "TickSize"]
        if not all(c in df_scrips.columns for c in req_cols):
            st.error("Unexpected Scrip Master format from 5paisa.")
            logger.error("Missing expected columns in Scrip Master.")
            return None

        df_scrips["ScripCode"] = pd.to_numeric(df_scrips["ScripCode"], errors="coerce").fillna(0).astype(int)
        df_scrips["Symbol"] = df_scrips["Symbol"].astype(str).str.strip()

        df_scrips.to_parquet(SCRIPMASTER_PARQUET_PATH, index=False)
        df_scrips.to_csv(SCRIPMASTER_CSV_PATH, index=False)
        logger.info("Saved Scrip Master to Parquet & CSV.")

        return df_scrips

    except Exception as e:  # noqa: BLE001
        logger.critical("Exception in fetch_scrip_master: %s", e, exc_info=True)
        st.error(f"An error occurred with Scrip Master: {e}")
        return None
def get_scrip_details(scrip_master_df: Optional[pd.DataFrame], symbol: str) -> Optional[pd.Series]:
    """
    Finds scrip details (ScripCode, Exchange, etc.) from the Scrip Master DataFrame.
    Prioritizes NSE, then BSE if duplicates exist for equities.
    """
    if scrip_master_df is None or scrip_master_df.empty:
        logger.warning("Scrip Master is not loaded. Cannot find scrip details.")
        return None
    if not symbol:
        logger.warning("No symbol provided to get_scrip_details.")
        return None

    try:
        # Exact match on symbol
        matches = scrip_master_df[scrip_master_df['Symbol'].str.upper() == symbol.upper()]

        if matches.empty:
            logger.warning(f"No scrip found for symbol: {symbol}")
            return None

        if len(matches) == 1:
            return matches.iloc[0]
        else:
            # Handle multiple matches (e.g., same symbol on NSE and BSE)
            # Prioritize 'C' (Cash/Equity) segment, then 'N' (NSE)
            equity_matches = matches[matches['ExchType'] == 'C']
            if not equity_matches.empty:
                nse_match = equity_matches[equity_matches['Exch'] == 'N']
                if not nse_match.empty:
                    logger.info(f"Multiple matches for {symbol}. Prioritizing NSE equity.")
                    return nse_match.iloc[0]
                bse_match = equity_matches[equity_matches['Exch'] == 'B']
                if not bse_match.empty:
                    logger.info(f"Multiple matches for {symbol}. Prioritizing BSE equity.")
                    return bse_match.iloc[0]
                logger.info(f"Multiple equity matches for {symbol}. Returning first one.")
                return equity_matches.iloc[0] # Fallback to first equity match
            else: # No equity matches, could be derivatives or index
                nse_match = matches[matches['Exch'] == 'N']
                if not nse_match.empty:
                     logger.info(f"Multiple non-equity matches for {symbol}. Prioritizing NSE.")
                     return nse_match.iloc[0]
                logger.info(f"Multiple matches for {symbol} (non-equity, non-NSE). Returning first overall match.")
                return matches.iloc[0] # Fallback to first overall match

    except Exception as e:
        logger.error(f"Error finding scrip details for {symbol}: {e}", exc_info=True)
        return None


# @st.cache_data(ttl=60*5, max_entries=20, show_spinner="Fetching historical stock data...") # Cache for 5 mins
def fetch_historical_data(
    client: FivePaisaClient,
    scrip_details: pd.Series,
    timeframe: str,
    days_back: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV history for a symbol and timeframe; return a cleaned DataFrame.
    """
    if not client:
        logger.error("Unauthenticated 5paisa client.")
        return None
    if scrip_details is None or not {"ScripCode", "Exch", "ExchType"}.issubset(scrip_details.index):
        st.error("Invalid scrip details.")
        return None
    if timeframe not in AVAILABLE_TIMEFRAMES:
        st.error(f"Invalid timeframe: {timeframe}")
        return None

    scrip_code = int(scrip_details["ScripCode"])
    exch, exch_type = scrip_details["Exch"], scrip_details["ExchType"]
    symbol_log = scrip_details.get("Symbol", str(scrip_code))

    if days_back is None:
        days_back = MAX_INTRADAY_LOOKBACK_DAYS if timeframe != "1d" else DEFAULT_HISTORICAL_DAYS_FETCH

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    from_date_str, to_date_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    logger.info("5paisa hist-data %s (%s) TF=%s  %s→%s", symbol_log, scrip_code, timeframe, from_date_str, to_date_str)

    try:
        df_ohlcv = client.historical_data(
            Exch=exch,
            ExchangeSegment=exch_type,
            ScripCode=scrip_code,
            time=timeframe,
            From=from_date_str,
            To=to_date_str,
        )

        if isinstance(df_ohlcv, str):
            st.error(f"5paisa error: {df_ohlcv}"); logger.error(df_ohlcv); return None
        if df_ohlcv is None or df_ohlcv.empty:
            logger.warning("No historical data for %s.", symbol_log)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if "Time" in df_ohlcv.columns:
            df_ohlcv.rename(columns={"Time": "Datetime"}, inplace=True)
        df_ohlcv["Datetime"] = pd.to_datetime(df_ohlcv["Datetime"])
        df_ohlcv.set_index("Datetime", inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df_ohlcv.columns:
                df_ohlcv[col] = pd.to_numeric(df_ohlcv[col], errors="coerce")
            else:
                df_ohlcv[col] = np.nan
                logger.warning("Column '%s' missing in API response.", col)

        df_ohlcv = df_ohlcv[[c for c in OHLCV_COLUMNS if c != "Datetime"]]
        df_ohlcv.sort_index(inplace=True)
        df_ohlcv.dropna(subset=["Close"], inplace=True)

        logger.info("Processed historical data for %s. Shape=%s", symbol_log, df_ohlcv.shape)
        return df_ohlcv

    except Exception as e:  # noqa: BLE001
        logger.critical("Exception fetching historical data: %s", e, exc_info=True)
        st.error(f"Error fetching historical data for {symbol_log}: {e}")
        return None

# @st.cache_data(ttl=60*1, max_entries=50, show_spinner="Calculating technical indicators...") # Cache for 1 minute
def calculate_technical_indicators(
    ohlcv_df: pd.DataFrame,
    enabled_indicators_config: Dict[str, bool]
) -> pd.DataFrame:
    """
    Add selected pandas-ta indicators to an OHLCV DataFrame.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        logger.warning("OHLCV data empty – indicators skipped.")
        return pd.DataFrame()

    data_with_indicators = ohlcv_df.copy()

    # pandas_ta expects lowercase columns
    lowercase_map = {col: col.lower() for col in OHLCV_COLUMNS if col in data_with_indicators.columns}
    temp_df_for_ta = data_with_indicators.rename(columns=lowercase_map).copy()

    for ind_key, enabled in enabled_indicators_config.items():
        if not enabled:
            continue
        if ind_key not in DEFAULT_INDICATORS:
            logger.warning("Indicator %s enabled but not in config.", ind_key)
            continue

        cfg = DEFAULT_INDICATORS[ind_key]
        func_name, params = cfg["function_name_pd_ta"], cfg["params"]

        try:
            ta_func = getattr(temp_df_for_ta.ta, func_name)
            indicator_output = ta_func(**params, append=False)

            if indicator_output is None or indicator_output.empty:
                logger.warning("Indicator %s produced no data.", ind_key)
                continue

            if isinstance(indicator_output, pd.Series):
                out_name = cfg["output_cols"][0] if cfg["output_cols"] else indicator_output.name
                data_with_indicators[out_name] = indicator_output.rename(out_name)
            else:  # DataFrame
                if len(indicator_output.columns) == len(cfg["output_cols"]):
                    indicator_output.columns = cfg["output_cols"]
                for col in indicator_output.columns:
                    data_with_indicators[col] = indicator_output[col]

            logger.debug("Calculated %s.", ind_key)

        except AttributeError:
            logger.error("pandas_ta missing function '%s' for indicator '%s'.", func_name, ind_key)
        except Exception as e:  # noqa: BLE001
            logger.error("Error calculating %s: %s", ind_key, e, exc_info=True)

    logger.info("Indicators added. Final shape: %s", data_with_indicators.shape)
    return data_with_indicators


# --- Placeholder for Live Market Data (WebSocket or Polling) ---
# This will be a more complex implementation
# def get_live_market_data(client: FivePaisaClient, scrip_codes: List[int], on_message_callback):
#     pass


# --- Example Usage (for testing this module directly) ---
# fivepaisa_ai_dashboard/data_handler.py
# ... (all the existing code for imports, logger, AlphaVantageAPI class, 
#      global alpha_vantage_client, 5paisa functions, calculate_technical_indicators) ...

# --- Example Usage & Testing (Driver Function) ---
if __name__ == "__main__":
    # This block executes only when data_handler.py is run directly.
    # Ideal for testing the data fetching and processing functions in isolation.

    st.set_page_config(page_title="Data Handler Module Test", layout="wide", initial_sidebar_state="collapsed")
    st.title("🛠️ Data Handler Module Test Suite")
    st.markdown("---")
    st.info(
        "This page allows testing of individual functions within `data_handler.py`. "
        "Ensure your API keys are configured in `.streamlit/secrets.toml` for live Alpha Vantage tests. "
        "5paisa functions are tested using mocks here; for live 5paisa tests, use the main `app.py` after logging in."
    )
    st.markdown("---")

    # --- Alpha Vantage API Tests ---
    st.header("🧪 Alpha Vantage API Functions Test")
    av_api_key_for_test = st.secrets.get(ALPHA_VANTAGE_API_KEY_SECRET_NAME)

    if not av_api_key_for_test:
        st.error(
            f"Alpha Vantage API Key ('{ALPHA_VANTAGE_API_KEY_SECRET_NAME}') not found in `.streamlit/secrets.toml`. "
            "Please add it to run Alpha Vantage tests."
        )
    else:
        st.success("Alpha Vantage API Key found. You can proceed with AV tests.")
        # Use the globally instantiated client which should have picked up the key
        av_test_client_instance = alpha_vantage_client

        av_test_symbol_ui = st.text_input(
            "Enter Symbol for Alpha Vantage (e.g., IBM, AAPL, MSFT, QQQ for ETF):",
            value="IBM",
            key="av_test_symbol"
        )
        av_test_quarter_ui = st.text_input(
            "Enter Quarter for Transcript (e.g., 2023Q4):",
            value="2023Q4",
            key="av_test_quarter"
        )
        av_test_date_ui = st.text_input(
            "Enter Date for Listing Status (YYYY-MM-DD, optional):",
            value="", #datetime.now().strftime("%Y-%m-%d"),
            key="av_test_date"
        )


        st.subheader("Fundamental Data & Corporate Actions:")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Fetch Company Overview", key="av_overview"):
                with st.spinner(f"Fetching Overview for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_company_overview(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})
            if st.button("Fetch Earnings (EPS)", key="av_earnings"):
                with st.spinner(f"Fetching Earnings for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_earnings(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})

        with col2:
            if st.button("Fetch Income Statement", key="av_income"):
                with st.spinner(f"Fetching Income Stmt for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_income_statement(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})
            if st.button("Fetch Dividends", key="av_dividends"):
                with st.spinner(f"Fetching Dividends for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_dividends(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})

        with col3:
            if st.button("Fetch Balance Sheet", key="av_balance"):
                with st.spinner(f"Fetching Balance Sheet for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_balance_sheet(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})
            if st.button("Fetch Stock Splits", key="av_splits"):
                with st.spinner(f"Fetching Splits for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_splits(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})
        
        st.markdown("---")
        st.subheader("Market Intelligence & News:")
        col4, col5, col6 = st.columns(3)
        with col4:
            if st.button("Fetch News & Sentiment (AAPL)", key="av_news"):
                with st.spinner("Fetching News for AAPL (limit 5)..."):
                    data = av_test_client_instance.get_news_sentiment(tickers="AAPL", limit=5)
                    if data and data.get("feed"):
                        for item in data["feed"][:3]: # Show a few
                            st.write(f"**{item['title']}** ({item['source']}) - Sent: {item['overall_sentiment_label']}")
                    else:
                         st.json({"error": "No news data or failed."})
            if st.button(f"Fetch ETF Profile ({av_test_symbol_ui})", key="av_etf"):
                with st.spinner(f"Fetching ETF Profile for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_etf_profile(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})

        with col5:
            if st.button("Fetch Earnings Calendar (Global)", key="av_earn_cal"):
                with st.spinner("Fetching Global Earnings Calendar (3mo)..."):
                    df = av_test_client_instance.get_earnings_calendar(horizon="3month")
                    st.dataframe(df.head() if df is not None and not df.empty else pd.DataFrame({"error": ["No data or failed."]}))
            if st.button(f"Fetch Insider Transactions ({av_test_symbol_ui})", key="av_insider"):
                with st.spinner(f"Fetching Insider Tx for {av_test_symbol_ui}..."):
                    data = av_test_client_instance.get_insider_transactions(av_test_symbol_ui)
                    st.json(data if data else {"error": "No data or failed."})
        
        with col6:
            if st.button("Fetch IPO Calendar (Global)", key="av_ipo_cal"):
                with st.spinner("Fetching Global IPO Calendar..."):
                    df = av_test_client_instance.get_ipo_calendar()
                    st.dataframe(df.head() if df is not None and not df.empty else pd.DataFrame({"error": ["No data or failed."]}))
            if st.button(f"Fetch Transcript ({av_test_symbol_ui} {av_test_quarter_ui})", key="av_transcript"):
                with st.spinner(f"Fetching Transcript for {av_test_symbol_ui} {av_test_quarter_ui}..."):
                    data = av_test_client_instance.get_earnings_call_transcript(av_test_symbol_ui, av_test_quarter_ui)
                    if data:
                        st.text_area("Transcript Excerpt", str(data)[:1000]+"...", height=150)
                    else:
                        st.json({"error": "No transcript data or failed."})

        st.markdown("---")
        st.subheader("Market Status & Movers:")
        col7, col8 = st.columns(2)
        with col7:
            if st.button("Fetch Listing Status (Active)", key="av_list_active"):
                with st.spinner(f"Fetching Active Listing Status (Date: {av_test_date_ui if av_test_date_ui else 'Latest'})..."):
                    df = av_test_client_instance.get_listing_status(date=av_test_date_ui if av_test_date_ui else None, state="active")
                    st.dataframe(df.head() if df is not None and not df.empty else pd.DataFrame({"error": ["No data or failed."]}))
        with col8:
            if st.button("Fetch Top Gainers/Losers (US)", key="av_top_movers"):
                with st.spinner("Fetching Top Gainers/Losers..."):
                    data = av_test_client_instance.get_top_gainers_losers()
                    st.json(data if data else {"error": "No data or failed."})

    st.markdown("---")
    st.header("🛠️ 5paisa API Functions Test (Mocked Data)")

    # Mock client for offline testing of function structure
    class MockFivePaisaClient:
        def __init__(self, cred=None): self.cred = cred; self.client_code = "MOCKUSER"
        def get_scrips(self):
            return [{'ScripCode': 1660, 'Symbol': 'RELIANCE', 'FullName': 'RELIANCE INDUSTRIES LTD', 'Exch': 'N', 'ExchType': 'C', 'TickSize': 0.05, 'LotSize':1, 'Name': 'RELIANCE'},
                    {'ScripCode': 2885, 'Symbol': 'ITC', 'FullName': 'ITC LTD', 'Exch': 'N', 'ExchType': 'C', 'TickSize': 0.05, 'LotSize':1, 'Name': 'ITC'},
                    {'ScripCode': 999901, 'Symbol': 'NIFTY', 'FullName': 'NIFTY 50 INDEX', 'Exch': 'N', 'ExchType': 'I', 'TickSize': 0.05, 'LotSize':50, 'Name': 'NIFTY'}]
        def historical_data(self, Exchange, ExchangeType, ScripCode, TimeFrame, FromDate, ToDate):
            mock_dates = pd.to_datetime(pd.date_range(start=FromDate, end=ToDate, freq=TIMEFRAME_TO_PANDAS_FREQ.get(TimeFrame, 'B')))
            if mock_dates.empty: # Ensure some data for shorter TFs or small date ranges
                periods = 100 if TimeFrame != '1d' else (pd.to_datetime(ToDate) - pd.to_datetime(FromDate)).days + 1
                periods = max(2, periods) # Ensure at least 2 periods for TA calculations
                mock_dates = pd.to_datetime(pd.date_range(end=ToDate, periods=periods, freq=TIMEFRAME_TO_PANDAS_FREQ.get(TimeFrame, '15min' if TimeFrame != '1d' else 'B')))
            
            data_size = len(mock_dates)
            if data_size == 0: return pd.DataFrame()
            base_price = 100 + int(ScripCode) % 1000 
            np.random.seed(int(ScripCode)) # Make mock data consistent for a given scripcode
            opens = base_price + np.cumsum(np.random.normal(0, base_price * 0.005, data_size))
            closes = opens + np.random.normal(0, base_price * 0.01, data_size)
            highs = np.maximum(opens, closes) + np.random.uniform(0, base_price * 0.005, data_size)
            lows = np.minimum(opens, closes) - np.random.uniform(0, base_price * 0.005, data_size)
            # Ensure OHLC logic
            opens_closes_min = np.minimum(opens, closes)
            opens_closes_max = np.maximum(opens, closes)
            lows = np.minimum(lows, opens_closes_min)
            highs = np.maximum(highs, opens_closes_max)

            df = pd.DataFrame({
                'Time': mock_dates, 'Open': opens, 'High': highs, 'Low': lows, 'Close': closes,
                'Volume': np.random.randint(1000, 100000, data_size).astype(float)
            })
            return df

    mock_5p_client_instance = MockFivePaisaClient()

    col_5p_1, col_5p_2 = st.columns(2)
    with col_5p_1:
        if st.button("Test Fetch Scrip Master (Mock)", key="mock_sm"):
            with st.spinner("Fetching Scrip Master (Mock)..."):
                sm_df = fetch_scrip_master(mock_5p_client_instance)
                if sm_df is not None and not sm_df.empty:
                    st.subheader("Scrip Master (Mocked)")
                    st.dataframe(sm_df.head())
                    reliance_details = get_scrip_details(sm_df, "RELIANCE")
                    st.write("Details for RELIANCE (Mock):")
                    st.json(reliance_details.to_dict() if reliance_details is not None else {"error": "Not found"})
                else:
                    st.error("Failed to fetch/process mock scrip master.")
    with col_5p_2:
        if st.button("Test Fetch Hist. Data & Indicators (Mock RELIANCE, 1d)", key="mock_hist_ind"):
            sm_df_hist_test = fetch_scrip_master(mock_5p_client_instance)
            if sm_df_hist_test is not None:
                details = get_scrip_details(sm_df_hist_test, "RELIANCE")
                if details is not None:
                    with st.spinner("Fetching Historical Data & Indicators (Mock)..."):
                        ohlcv = fetch_historical_data(mock_5p_client_instance, details, "1d", days_back=250) # Enough for TA
                        if ohlcv is not None and not ohlcv.empty:
                            st.subheader("OHLCV Data for RELIANCE (Mocked, 1d)")
                            st.dataframe(ohlcv.tail())
                            
                            st.subheader("Calculating Indicators on Mock Data...")
                            enabled_inds_test = {
                                indicator_key: True for indicator_key in 
                                ["SMA_20", "RSI_14", "MACD_12_26_9", "BBANDS_20_2"] # Test a few diverse ones
                            }
                            data_with_inds = calculate_technical_indicators(ohlcv, enabled_inds_test)
                            st.dataframe(data_with_inds.tail())
                        else:
                            st.error("Failed to fetch/process mock historical data.")
                else:
                    st.error("Could not get mock scrip details for RELIANCE.")
            else:
                st.error("Mock Scrip Master failed for historical data test.")