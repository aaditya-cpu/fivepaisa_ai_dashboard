# fivepaisa_ai_dashboard/auth_5paisa.py
"""
Authentication Module for 5paisa API

Handles the login process, token generation, and session management
for interacting with the 5paisa trading APIs using the `py5paisa` SDK.

Key Responsibilities:
- Load API credentials from environment variables (via a `.env` file).
- Provide a UI for TOTP and PIN entry.
- Manage the `RequestToken` and `AccessToken`.
- Store client and token information in Streamlit's session state.
- Offer methods to get the authenticated 5paisa client instance.
"""

import streamlit as st
from py5paisa import FivePaisaClient
import logging
import os
from dotenv import load_dotenv

# Load environment variables once at import time. Safe even if `.env` is absent.
load_dotenv()

# Import configurations
from config import APP_TITLE  # For consistent logging/messaging

# --------------------------------------------------------------------------- #
#                                 Logging                                     #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(f"{APP_TITLE}.Auth")
if not logger.handlers:  # Avoid adding multiple handlers if the module reloads
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#                               Session‑state keys                            #
# --------------------------------------------------------------------------- #
SESSION_STATE_KEY_CLIENT = "5paisa_client_instance"
SESSION_STATE_KEY_CLIENT_CODE = "5paisa_client_code"
SESSION_STATE_KEY_ACCESS_TOKEN = "5paisa_access_token"
SESSION_STATE_KEY_REQUEST_TOKEN = "5paisa_request_token"  # For OAuth, rarely used
SESSION_STATE_KEY_LOGGED_IN_STATUS = "5paisa_logged_in"
SESSION_STATE_KEY_USER_KEY = "5paisa_user_key"  # May be required for refresh

# --------------------------------------------------------------------------- #
#                                Helper functions                             #
# --------------------------------------------------------------------------- #

def _load_credentials_from_env() -> dict | None:
    """Return dict for `FivePaisaClient(cred=…)` or *None* if incomplete."""
    load_dotenv(override=False)                     # reload = no-op if already done
    required = ["APP_NAME", "APP_SOURCE", "USER_ID",
                "PASSWORD", "USER_KEY", "ENCRYPTION_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        msg = f"Missing credentials in .env: {', '.join(missing)}"
        logger.error(msg); st.error(msg)
        return None

    creds = {k: os.getenv(k) for k in required}
    st.session_state.setdefault(SESSION_STATE_KEY_USER_KEY, creds["USER_KEY"])
    logger.info("Successfully loaded API credentials from .env.")
    return creds

def _get_totp_login_details_from_env() -> tuple[str | None, str | None]:
    """Fetch `CLIENT_CODE` & `PIN` from env vars."""
    code, pin = os.getenv("CLIENT_CODE"), os.getenv("PIN")
    if not code:
        st.error("CLIENT_CODE missing in .env.")
    if not pin:
        st.error("PIN missing in .env.")
    return code, pin

# --------------------------------------------------------------------------- #
#                          Main authentication functions                      #
# --------------------------------------------------------------------------- #

# auth_5paisa.py
def login_via_totp_session(totp_code: str) -> bool:
    """Return *True* on successful TOTP login, else *False*."""
    st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False  # reset flag

    creds = _load_credentials_from_env()
    code, pin = _get_totp_login_details_from_env()
    if not creds or not code or not pin:
        return False

    try:
        with st.spinner("Authenticating with 5paisa…"):
            client = FivePaisaClient(cred=creds)
            logger.info("Attempting TOTP login for Client Code: %s", code)

            # py5paisa expects positional args: client_code, totp, pin
            api_resp = client.get_totp_session(code, totp_code, pin)
            logger.debug("Raw TOTP login response: %s", api_resp)

            token = client.get_access_token()
            if token and client.client_code:
                st.session_state.update({
                    SESSION_STATE_KEY_CLIENT: client,
                    SESSION_STATE_KEY_CLIENT_CODE: client.client_code,
                    SESSION_STATE_KEY_ACCESS_TOKEN: token,
                    SESSION_STATE_KEY_LOGGED_IN_STATUS: True,
                })
                logger.info("Login successful – AccessToken cached.")
                st.success(f"Logged in as {client.client_code}")
                return True

            # parse API error
            if api_resp is None:
                st.error("No response from 5paisa API. Please try again later.")
                logger.error("Login failed: API response was None.")
                return False

            status = api_resp.get("body", {}).get("Status")
            message = api_resp.get("body", {}).get("Message", "Unknown error")
            st.error("Invalid credentials or PIN." if status == 1 else
                     "Invalid/expired TOTP." if status == 2 else message)
            logger.error("Login failed: status=%s, message=%s", status, message)
            return False

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled exception during login: %s", exc)
        st.error(f"Unexpected error: {exc}")
        return False


def logout():
    """Clear all 5paisa‑related keys from Streamlit session_state."""
    keys_to_clear = [
        SESSION_STATE_KEY_CLIENT,
        SESSION_STATE_KEY_CLIENT_CODE,
        SESSION_STATE_KEY_ACCESS_TOKEN,
        SESSION_STATE_KEY_REQUEST_TOKEN,
        SESSION_STATE_KEY_LOGGED_IN_STATUS,
        SESSION_STATE_KEY_USER_KEY,
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    logger.info("User logged out. Session cleared.")
    st.info("You have been logged out.")
    st.rerun()


def get_authenticated_client() -> FivePaisaClient | None:
    """Return a cached, authenticated `FivePaisaClient` instance or *None*."""
    if st.session_state.get(SESSION_STATE_KEY_LOGGED_IN_STATUS):
        client = st.session_state.get(SESSION_STATE_KEY_CLIENT)
        if client and isinstance(client, FivePaisaClient):
            return client

        logger.warning(
            "Logged‑in status was True but no valid client instance found. "
            "Resetting status flag."
        )
        st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False
    return None


def is_user_logged_in() -> bool:
    """Quick check whether the user is currently logged in."""
    return st.session_state.get(SESSION_STATE_KEY_LOGGED_IN_STATUS, False)

# --------------------------------------------------------------------------- #
#                           UI helper (Streamlit)                             #
# --------------------------------------------------------------------------- #

def display_login_form(cols=None):
    """Render collapsible login widget (sidebar by default)."""
    container = cols[0] if cols else st.sidebar
    with container.expander("🔒 **5paisa Login**", expanded=not is_user_logged_in()):
        if is_user_logged_in():
            st.success(f"Logged in as **{st.session_state.get(SESSION_STATE_KEY_CLIENT_CODE)}**")
            if st.button("Logout", key="btn_logout"):
                logout()
            return

        # Verify .env completeness before showing form
        if (not _load_credentials_from_env()
                or not all(_get_totp_login_details_from_env())):
            st.warning("Please ensure all credentials, CLIENT_CODE & PIN are set in `.env`.")
            return

        st.info("Enter the 6-digit code from your authenticator app.")
        with st.form("login_form"):
            totp = st.text_input("TOTP Code", max_chars=6, type="password", placeholder="******")
            if st.form_submit_button("Login"):
                if not (totp.isdigit() and len(totp) == 6):
                    st.error("Please enter a valid 6-digit TOTP.")
                elif login_via_totp_session(totp):
                    st.rerun()
# --------------------------------------------------------------------------- #
#                          Stand‑alone test entry‑point                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    st.set_page_config(page_title="Auth Test", layout="wide")
    st.title("5paisa Authentication Module Test")

    # Prepare session keys for direct execution
    st.session_state.setdefault(SESSION_STATE_KEY_LOGGED_IN_STATUS, False)

    display_login_form()

    st.divider()
    st.subheader("Session State Inspector:")
    st.json(dict(st.session_state))

    if is_user_logged_in():
        st.success("User is logged in.")
        client = get_authenticated_client()
        if client:
            try:
                with st.spinner("Fetching holdings as a quick test…"):
                    holdings = client.holdings()
                    st.json(holdings)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Error fetching holdings: {exc}")
    else:
        st.warning("User is not logged in.")
