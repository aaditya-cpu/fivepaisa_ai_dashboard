# ── Standard & third-party ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np                     # ← keep: still handy for quick calcs
import plotly.graph_objects as go

# ── Project modules ────────────────────────────────────────────────────
from config import (
    APP_TITLE, PAGE_ICON, DEFAULT_HISTORICAL_TIMEFRAME, AVAILABLE_TIMEFRAMES,
    DEFAULT_INDICATORS, CANDLESTICK_PATTERNS_TO_DETECT, PREDICTION_HORIZONS,
    DEFAULT_WATCHLIST_SYMBOLS, DEFAULT_SYMBOL_INDEX, DEFAULT_SYMBOL_EQUITY
)

from auth_5paisa import (
    display_login_form,
    is_user_logged_in,
    get_authenticated_client,
    logout,
)

from data_handler import (
    fetch_scrip_master,
    get_scrip_details,
    fetch_historical_data,
    calculate_technical_indicators,
)

from plotting import create_ohlcv_chart
from patterns import detect_candlestick_patterns
from predictive_models import get_prediction

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Night Theme and Styling ---
# (Keep this updated and organized)
NIGHT_THEME_CSS = """
<style>
    /* Base an_configd Font */
    html, body, [class*="st-"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #D4D4D4; /* Primary text color */
    }

    /* Main App Background */
    .stApp {
        background-color: #1E1E1E; /* Very dark grey/charcoal */
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #252526; /* Slightly lighter dark grey */
        border-right: 1px solid #4A4A4A;
    }
    .stSidebar [data-testid="stMarkdownContainer"] p,
    .stSidebar [data-testid="stMarkdownContainer"] li,
    .stSidebar [data-testid="stExpander"] summary p {
        color: #D4D4D4;
    }
    .stSidebar .stButton>button {
        background-color: #007ACC; /* Accent blue */
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stSidebar .stButton>button:hover {
        background-color: #005C99;
    }
    .stSidebar .stExpander {
        border: 1px solid #4A4A4A;
        border-radius: 5px;
        background-color: #2D2D30; /* Card background */
    }
    .stSidebar [data-testid="stExpander"] summary {
        font-size: 1.1em;
        font-weight: bold;
    }


    /* Main Content Area */
    [data-testid="stVerticalBlock"] { /* Main content block */
        /* background-color: #1E1E1E; */ /* Handled by stApp */
    }

    /* Expander and Card-like elements in main area */
    .main [data-testid="stExpander"], .stTabs [data-testid="stExpander"] {
        background-color: #2D2D30;
        border: 1px solid #4A4A4A;
        border-radius: 8px;
    }
    .main [data-testid="stExpander"] summary p {
        color: #E0E0E0; /* Slightly brighter for expander titles */
        font-weight: bold;
    }

    /* Tabs Styling */
    [data-testid="stTabs"] {
        background-color: #1E1E1E;
    }
    [data-testid="stTabs"] [data-testid="stMarkdownContainer"] p {
        color: #D4D4D4;
    }
    [data-testid="stTabs"] button[role="tab"] { /* Tab buttons */
        background-color: #2D2D30;
        color: #A0A0A0; /* Inactive tab text */
        border-radius: 5px 5px 0 0;
        border: 1px solid #4A4A4A;
        border-bottom: none;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { /* Active tab button */
        background-color: #3C3C40; /* Slightly different for active tab */
        color: #00AACC; /* Active tab text - Accent */
        font-weight: bold;
        border-color: #00AACC;
    }
    [data-testid="stTabContent"] { /* Content area of a tab */
        background-color: #2D2D30; /* Card background for tab content */
        border: 1px solid #4A4A4A;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #00AACC; /* Accent color for headings */
    }
    h1 {
        border-bottom: 2px solid #007ACC;
        padding-bottom: 0.3em;
    }

    /* Buttons in Main Area */
    .main .stButton>button { /* More specific selector for main area buttons */
        background-color: #007ACC;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .main .stButton>button:hover {
        background-color: #005C99;
    }

    /* Input Fields */
    .stTextInput input, .stSelectbox select, .stDateInput input, .stNumberInput input {
        background-color: #3C3C40 !important;
        color: #D4D4D4 !important;
        border: 1px solid #4A4A4A !important;
        border-radius: 5px !important;
    }

    /* Dataframes (if any are shown directly) */
    .stDataFrame {
        background-color: #2D2D30;
        border: 1px solid #4A4A4A;
    }

    /* Plotly Chart Background - Target Plotly's default bg */
    .js-plotly-plot .plotly {
        background-color: transparent !important; /* Make Plotly bg transparent to inherit from container */
    }
    /* Or, explicitly set Plotly bg to match if transparency is an issue */
    /*
    .js-plotly-plot .plotly .main-svg {
        background-color: #2D2D30 !important;
    }
    */

    /* Success, Error, Warning, Info Messages */
    [data-testid="stAlert"] {
        border-radius: 5px;
        color: #1E1E1E; /* Dark text for light backgrounds */
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="positive"] { /* Success */
        background-color: #4CAF50;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="negative"] { /* Error */
        background-color: #F44336;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="warning"] { /* Warning */
        background-color: #FF9800;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="info"] { /* Info */
        background-color: #2196F3;
    }

    /* Footer placeholder style */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #252526;
        color: #A0A0A0;
        text-align: center;
        padding: 5px 0;
        font-size: 0.8em;
        border-top: 1px solid #4A4A4A;
    }

</style>
"""
st.markdown(NIGHT_THEME_CSS, unsafe_allow_html=True)

# --- Initialize Session State Variables ---
# (Crucial for interactivity and persisting user choices/data)
def initialize_session_state():
    # Authentication related (often handled by auth_5paisa.py, but good to ensure)
    if "5paisa_logged_in" not in st.session_state:
        st.session_state["5paisa_logged_in"] = False

    # UI related
    if "selected_scrip_symbol" not in st.session_state:
        st.session_state["selected_scrip_symbol"] = DEFAULT_SYMBOL_EQUITY # Start with an equity
    if "selected_timeframe" not in st.session_state:
        st.session_state["selected_timeframe"] = DEFAULT_HISTORICAL_TIMEFRAME
    if "enabled_indicators" not in st.session_state:
        st.session_state["enabled_indicators"] = {
            k: v["enabled_by_default"] for k, v in DEFAULT_INDICATORS.items()
        }
    if "show_candlestick_patterns" not in st.session_state:
        st.session_state["show_candlestick_patterns"] = True # From config if defined

    # Data related
    if "scrip_master_df" not in st.session_state:
        st.session_state["scrip_master_df"] = None # pd.DataFrame()
    if "current_ohlcv_df" not in st.session_state:
        st.session_state["current_ohlcv_df"] = None # pd.DataFrame()
    if "current_indicator_data" not in st.session_state:
        st.session_state["current_indicator_data"] = {}
    if "detected_patterns_data" not in st.session_state:
        st.session_state["detected_patterns_data"] = None
    if "current_predictions" not in st.session_state:
        st.session_state["current_predictions"] = {} # Keyed by horizon

    # Control flags
    if "data_load_trigger" not in st.session_state: # To trigger data reload
        st.session_state["data_load_trigger"] = 0


initialize_session_state()

# --- Helper Functions for App ---
# (These will call functions from other modules)

def load_scrip_master_once():
    """
    Fetch the 5 Paisa scrip-master exactly once per session (after login).
    """
    if st.session_state["scrip_master_df"] is not None:
        return  # already cached

    client = get_authenticated_client()
    if not client:
        st.warning("Cannot load Scrip Master. Please log in.")
        return

    with st.spinner("Fetching Scrip Master…"):
        df = fetch_scrip_master(client)

    if df is None or df.empty:
        st.error("Failed to load Scrip Master. Some features may be limited.")
    else:
        st.session_state["scrip_master_df"] = df
        st.success("Scrip Master loaded.")

def update_chart_data():
    """Pull OHLCV → indicators → patterns → predictions for the chosen scrip."""
    symbol = st.session_state["selected_scrip_symbol"]
    if not symbol:
        st.warning("Please select a scrip.")
        return

    client = get_authenticated_client()
    if not client:
        st.warning("Please log in to fetch data.")
        return

    scrip_details = get_scrip_details(
        st.session_state["scrip_master_df"],
        symbol,
    )
    if scrip_details is None:
        st.error(f"Could not find details for scrip: {symbol}")
        return

    with st.spinner(f"Loading data for {symbol} ({st.session_state['selected_timeframe']})…"):
        # 1 — OHLCV
        st.session_state["current_ohlcv_df"] = fetch_historical_data(
            client,
            scrip_details,
            st.session_state["selected_timeframe"],
        )
        if (
            st.session_state["current_ohlcv_df"] is None
            or st.session_state["current_ohlcv_df"].empty
        ):
            st.error(f"Failed to fetch OHLCV data for {symbol}.")
            return

        # 2 — Indicators
        st.session_state["current_indicator_data"] = calculate_technical_indicators(
            st.session_state["current_ohlcv_df"],
            st.session_state["enabled_indicators"],
        )

        # 3 — Candlestick patterns
        if st.session_state["show_candlestick_patterns"]:
            st.session_state["detected_patterns_data"] = detect_candlestick_patterns(
                st.session_state["current_ohlcv_df"]
            )
        else:
            st.session_state["detected_patterns_data"] = None

        # 4 — AI predictions
        predictions: dict[str, dict] = {}
        for horizon_key, horizon_cfg in PREDICTION_HORIZONS.items():
            predictions[horizon_key] = get_prediction(
                st.session_state["current_ohlcv_df"],
                st.session_state["current_indicator_data"],
                symbol,
                horizon_key,
                horizon_cfg,
            )
        st.session_state["current_predictions"] = predictions
# --- Main Application Layout ---

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown("---")

    # Authentication Section
    display_login_form() # From auth_5paisa.py
    st.markdown("---")

    if is_user_logged_in():
        # Load scrip master after login
        load_scrip_master_once()

        st.subheader("⚙️ Controls")

        # Scrip Selection
        if st.session_state["scrip_master_df"] is not None and not st.session_state["scrip_master_df"].empty:
            # Sort symbols for better UX
            available_symbols = sorted(st.session_state["scrip_master_df"]['Symbol'].unique().tolist())
            try:
                current_symbol_index = available_symbols.index(st.session_state["selected_scrip_symbol"])
            except ValueError:
                current_symbol_index = 0 # Default to first if not found
                st.session_state["selected_scrip_symbol"] = available_symbols[0]


            selected_symbol = st.selectbox(
                "Select Scrip:",
                options=available_symbols,
                index=current_symbol_index,
                key="sb_selected_scrip"
            )
            if selected_symbol != st.session_state["selected_scrip_symbol"]:
                st.session_state["selected_scrip_symbol"] = selected_symbol
                st.session_state["data_load_trigger"] += 1 # Trigger data reload
        else:
            st.selectbox("Select Scrip:", options=["Login to load scrips"], disabled=True)

        # Timeframe Selection
        selected_tf = st.radio(
            "Select Timeframe:",
            options=AVAILABLE_TIMEFRAMES,
            index=AVAILABLE_TIMEFRAMES.index(st.session_state["selected_timeframe"]),
            horizontal=True,
            key="radio_selected_tf"
        )
        if selected_tf != st.session_state["selected_timeframe"]:
            st.session_state["selected_timeframe"] = selected_tf
            st.session_state["data_load_trigger"] += 1

        # Reload data button (manual trigger)
        if st.button("🔄 Reload Data & Chart", use_container_width=True):
            st.session_state["data_load_trigger"] += 1

        st.markdown("---")
        # Tools for Graphs (Indicators & Patterns)
        with st.expander("📊 Chart Tools", expanded=True):
            st.markdown("**Technical Indicators**")
            cols_indicators = st.columns(2)
            i = 0
            for ind_key, ind_config in DEFAULT_INDICATORS.items():
                with cols_indicators[i % 2]:
                    label = ind_config.get('label', ind_key) # Use label from config
                    is_enabled = st.checkbox(
                        label,
                        value=st.session_state["enabled_indicators"].get(ind_key, False),
                        key=f"cb_ind_{ind_key}"
                    )
                    if is_enabled != st.session_state["enabled_indicators"].get(ind_key, False):
                        st.session_state["enabled_indicators"][ind_key] = is_enabled
                        st.session_state["data_load_trigger"] += 1 # Redraw chart with new indicators
                i += 1

            st.markdown("**Pattern Overlays**")
            show_patterns = st.checkbox(
                "Show Candlestick Patterns",
                value=st.session_state["show_candlestick_patterns"],
                key="cb_show_patterns"
            )
            if show_patterns != st.session_state["show_candlestick_patterns"]:
                st.session_state["show_candlestick_patterns"] = show_patterns
                st.session_state["data_load_trigger"] += 1

        st.markdown("---")
        # Export Chart placeholder (functionality to be added to plotting.py)
        # if st.button("Export Chart as PNG", use_container_width=True):
        #     st.info("Export functionality to be implemented.")


# --- Main Content Area ---
if not is_user_logged_in():
    st.warning("👋 Please log in via the sidebar to access the dashboard features.")
    st.image("https://5paisa.com/media/images/logo/5paisa_Logo RGB.svg", width=300) # Example image
else:
    # If data_load_trigger changed, update chart data
    # This simple trigger helps avoid re-fetching on every rerun unless needed
    if "last_processed_trigger" not in st.session_state or \
       st.session_state["last_processed_trigger"] != st.session_state["data_load_trigger"]:
        update_chart_data()
        st.session_state["last_processed_trigger"] = st.session_state["data_load_trigger"]


    # Main Dashboard Layout
    st.header(f"📊 Dashboard: {st.session_state.get('selected_scrip_symbol', 'N/A')}")
    st.markdown("---")

    # Charting Area
    chart_container = st.container()
    with chart_container:
        st.subheader("Price Chart & Indicators")
        if st.session_state["current_ohlcv_df"] is not None and not st.session_state["current_ohlcv_df"].empty:
            fig = create_ohlcv_chart(
        st.session_state["current_ohlcv_df"],
        st.session_state["selected_scrip_symbol"],
        st.session_state["current_indicator_data"],
        st.session_state["enabled_indicators"],
        st.session_state["detected_patterns_data"],
            )
            fig.update_layout(
                title=f"{st.session_state['selected_scrip_symbol']} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",  # Use dark theme for better visibility
                height=600,
            )

            # Add export button (functionality to be implemented in plotting.py)
            # if st.button("Export Chart as PNG", use_container_width=True):
            #     st.info("Export functionality to be implemented.")

            st.plotly_chart(fig, use_container_width=True)
            st.session_state["current_ohlcv_fig"] = fig  # Store for later use if needed
        else:
            st.info("No chart data to display. Select a scrip and timeframe, then click 'Reload Data'.")

    st.markdown("---")

    # Analysis and Predictions Area (Using Tabs)
    st.subheader("AI Predictions & Analysis")

    pred_tabs = st.tabs([horizon_config["label"] for horizon_key, horizon_config in PREDICTION_HORIZONS.items()] + ["Pattern Insights"])

    for i, (horizon_key, horizon_config) in enumerate(PREDICTION_HORIZONS.items()):
        with pred_tabs[i]:
            st.markdown(f"#### {horizon_config['label']} Forecast")
            prediction_data = st.session_state["current_predictions"].get(horizon_key)
            if prediction_data:
                st.metric("Signal",           str(prediction_data.get("signal", "N/A")))
                st.metric("Confidence",       f"{prediction_data.get('confidence',0)*100:.2f}%")
                st.metric("Expected Return",  f"{prediction_data.get('expected_return_pct',0)*100:.2f}%")
                st.write(f"Model used: {prediction_data.get('model_type','N/A')}")
            else:
                st.info(f"No {horizon_config['label']} prediction available for the current scrip.")

with pred_tabs[-1]: # Pattern Insights Tab
    st.markdown("#### Candlestick Pattern Insights")
    if (
        st.session_state["detected_patterns_data"] is not None
        and not st.session_state["detected_patterns_data"].empty
    ):
        st.dataframe(st.session_state["detected_patterns_data"].tail())
    else:
        st.info("No specific candlestick patterns detected recently, or pattern detection is off.")

# --- Footer ---
st.markdown(
    """<div class="footer">
    Disclaimer: For educational and informational purposes only. Not financial advice. Trading involves risk.
    </div>""",
    unsafe_allow_html=True
)

# --- Main execution for when app.py is run directly ---
if __name__ == "__main__":
    # Streamlit apps should be launched with `streamlit run`. If this file is
    # executed directly via `python app.py`, the Streamlit context is not
    # available which leads to many "missing ScriptRunContext" warnings.  This
    # guard prints a helpful message and exits early when the app is started
    # incorrectly.
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    import sys

    if get_script_run_ctx() is None:
        print(
            "This application is a Streamlit app. "
            "Run it using:  streamlit run app.py"
        )
        sys.exit(0)

    # When executed within Streamlit this block can be used for one time start
    # up tasks.  Currently we simply show a fun greeting if the user is not yet
    # logged in.
    if not is_user_logged_in():
        st.balloons()  # Just a little welcome if not logged in