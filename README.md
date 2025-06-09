
## Project Structure

```


├── app.py                # Main Streamlit application
├── auth_5paisa.py        # Authentication helpers for 5paisa
├── config.py             # Central configuration constants
├── data_handler.py       # Data fetching and processing logic
├── patterns.py           # Candlestick pattern detection helpers
├── plotting.py           # Plotly chart generation utilities
├── predictive_models.py  # Basic prediction stubs
├── requirements.txt      # Python dependencies
└── utils.py              # Miscellaneous helpers
```

### Installation

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

The `setuptools` package is required so that `pandas-ta` can import `pkg_resources`.

### app.py
* Configures the Streamlit page and applies a custom dark theme.
* Initializes values in `st.session_state` via `initialize_session_state()`.
* Provides helper functions such as `load_scrip_master_once()` and `update_chart_data()` which fetch data, compute indicators and render charts.
* Renders the sidebar controls and main dashboard tabs using Streamlit widgets. Login UI is provided by `display_login_form()` from `auth_5paisa`.
* Includes a `__main__` guard that prints a message if the file is run outside of Streamlit, avoiding repeated "missing ScriptRunContext" warnings.

### auth_5paisa.py
* Handles login via 5paisa's TOTP authentication flow.
* Handles login via 5paisa's OAuth flow (user logs in on the 5paisa website and provides a `RequestToken`).
* Loads API credentials from a `.env` file using `python-dotenv`.
  * Required variables: `APP_NAME`, `APP_SOURCE`, `USER_ID`, `PASSWORD`, `USER_KEY`, `ENCRYPTION_KEY`, `CLIENT_CODE`, `PIN`.
  * Required variables: `APP_NAME`, `APP_SOURCE`, `USER_ID`, `PASSWORD`, `USER_KEY`, `ENCRYPTION_KEY`.
  * Optional: `OAUTH_RESPONSE_URL` to override the OAuth redirect URL.
* Defines session-state keys such as `SESSION_STATE_KEY_CLIENT` and `SESSION_STATE_KEY_ACCESS_TOKEN`.
* Main functions:
  * `_load_credentials_from_env()` – returns a dictionary of credentials or `None` if any are missing.
  * `_get_totp_login_details_from_env()` – retrieves `CLIENT_CODE` and `PIN` from the environment.
  * `login_via_totp_session(totp_code)` – performs the login request and stores the authenticated client in `st.session_state`.
  * `generate_oauth_login_url()` – constructs the URL to initiate OAuth login.
  * `login_via_oauth_request_token(token)` – exchanges the RequestToken for an AccessToken and caches the authenticated client.
  * `logout()` – clears authentication details from the session.
  * `get_authenticated_client()` – fetches the currently logged-in `FivePaisaClient`.
  * `display_login_form()` – renders the login UI in the sidebar.

### config.py
Centralized settings used across the project.  Highlights include:
* Directory paths (`BASE_DIR`, `DATA_DIR`, `MODELS_STORE_DIR`, etc.) which are created on import.
* UI defaults (app title, watchlist symbols, candle count, etc.).
* Dictionaries describing available technical indicators (`DEFAULT_INDICATORS`) and candlestick pattern names (`CANDLESTICK_PATTERNS_TO_DETECT`).
* Machine‑learning configuration such as `PREDICTION_HORIZONS` and LightGBM parameters (`LGBM_PARAMS_CLASSIFIER`, `LGBM_PARAMS_REGRESSOR`).
* Logging options, email/slack alert toggles, and other misc constants.
* Provides `get_config_value(path, default)` helper for retrieving nested values.

### data_handler.py
Implements data access helpers. Key parts:
* `AlphaVantageAPI` – a helper class to call Alpha Vantage endpoints with caching logic. Methods include `get_company_overview`, `get_income_statement`, `get_earnings_calendar`, etc.
* Global instance `alpha_vantage_client` for re-use across the app.
* 5paisa data functions such as `fetch_scrip_master()`, `get_scrip_details()`, and `fetch_historical_data()` (the latter heavily commented as it is still under development).
* `calculate_technical_indicators()` – merges pandas‑ta indicators into OHLCV data based on `DEFAULT_INDICATORS` settings.
* An interactive `__main__` block allows testing of these functions via Streamlit when this module is executed directly.

### utils.py
Utility helpers for logging setup and small data‑processing functions.
* `setup_global_logging()` – configures a rotating log file and console output based on values from `config.py`.
* Date/time helpers like `format_datetime_for_api`, `get_date_range_for_fetch`, and `parse_api_datetime_string`.
