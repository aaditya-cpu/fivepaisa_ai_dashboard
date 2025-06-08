# fivepaisa_ai_dashboard/auth_5paisa.py
"""
Authentication Module for 5paisa API (OAuth flow)

This version migrates from the in‑SDK **TOTP/PIN** login to the official
**browser‑based OAuth** procedure:

1. Dashboard generates a *Vendor login URL* and opens it in a new tab.
2. User authenticates on 5paisa → redirected back with a `RequestToken`.
3. User pastes that token into the Streamlit form.
4. We exchange the RequestToken for an AccessToken via the SDK.

Key points
----------
* Credentials still loaded from **`.env`**.
* `generate_oauth_login_url()` builds the redirect link.
* `login_via_oauth_request_token()` finishes the handshake.
* Auth state cached in `st.session_state` keys.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import streamlit as st
from dotenv import load_dotenv
from py5paisa import FivePaisaClient

from config import APP_TITLE

# ─────────────────────────── Logging setup ────────────────────────────
logger = logging.getLogger(f"{APP_TITLE}.Auth")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ─────────────────────────── Environment & constants ───────────────────
load_dotenv()

SESSION_STATE_KEY_CLIENT = "5paisa_client_instance"
SESSION_STATE_KEY_CLIENT_CODE = "5paisa_client_code"
SESSION_STATE_KEY_ACCESS_TOKEN = "5paisa_access_token"
SESSION_STATE_KEY_REQUEST_TOKEN = "5paisa_request_token"
SESSION_STATE_KEY_LOGGED_IN_STATUS = "5paisa_logged_in"
SESSION_STATE_KEY_USER_KEY = "5paisa_user_key"

# ─────────────────────────── Internal helpers ──────────────────────────

def _load_credentials_from_env() -> dict | None:
    """Return dict for `FivePaisaClient(cred=…)` or *None* if incomplete."""
    load_dotenv(override=False)
    required = [
        "APP_NAME",
        "APP_SOURCE",
        "USER_ID",
        "PASSWORD",
        "USER_KEY",
        "ENCRYPTION_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        msg = f"Missing credentials in .env: {', '.join(missing)}"
        logger.error(msg); st.error(msg)
        return None
    creds = {k: os.getenv(k) for k in required}
    st.session_state.setdefault(SESSION_STATE_KEY_USER_KEY, creds["USER_KEY"])
    logger.info("Successfully loaded API credentials from .env.")
    return creds


def _get_oauth_redirect_url() -> str:
    """Return the callback URL used in OAuth flow (env var or default)."""
    default_url = "https://www.google.com"
    url = os.getenv("OAUTH_RESPONSE_URL", default_url)
    if url == default_url:
        logger.warning("OAUTH_RESPONSE_URL not set; using default %s", default_url)
    return url


def generate_oauth_login_url(state: str = "STREAMLIT") -> str:
    """Compose the vendor‑login URL for the user to authenticate."""
    creds = _load_credentials_from_env()
    if not creds:
        return ""
    vendor_key = creds["USER_KEY"]
    resp_url = _get_oauth_redirect_url()
    base = "https://dev-openapi.5paisa.com/WebVendorLogin/VLogin/Index"
    return f"{base}?VendorKey={vendor_key}&ResponseURL={resp_url}&State={state}"

# ─────────────────────────── Auth core (OAuth) ─────────────────────────

def login_via_oauth_request_token(request_token: str) -> bool:
    """Use RequestToken → obtain AccessToken; cache client; return success."""
    st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False

    creds = _load_credentials_from_env()
    if not creds:
        return False

    try:
        with st.spinner("Exchanging RequestToken for AccessToken…"):
            client = FivePaisaClient(cred=creds)
            logger.info("Attempting OAuth login with RequestToken: %s", request_token)
            access_token = client.get_access_token(request_token)

            if access_token and client.client_code:
                st.session_state.update(
                    {
                        SESSION_STATE_KEY_CLIENT: client,
                        SESSION_STATE_KEY_CLIENT_CODE: client.client_code,
                        SESSION_STATE_KEY_ACCESS_TOKEN: access_token,
                        SESSION_STATE_KEY_LOGGED_IN_STATUS: True,
                    }
                )
                logger.info("OAuth login successful for Client %s", client.client_code)
                st.success(f"Logged in as {client.client_code}")
                return True

            err = "OAuth login failed. Verify that the RequestToken is valid."
            logger.error(err); st.error(err)
            return False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected OAuth error: %s", exc)
        st.error(f"Unexpected error: {exc}")
        return False

# ─────────────────────────── Public helpers ────────────────────────────

def logout() -> None:
    """Clear auth keys and rerun UI."""
    for k in [
        SESSION_STATE_KEY_CLIENT,
        SESSION_STATE_KEY_CLIENT_CODE,
        SESSION_STATE_KEY_ACCESS_TOKEN,
        SESSION_STATE_KEY_REQUEST_TOKEN,
        SESSION_STATE_KEY_LOGGED_IN_STATUS,
        SESSION_STATE_KEY_USER_KEY,
    ]:
        st.session_state.pop(k, None)
    logger.info("User logged out (session cleared).")
    st.info("You have been logged out.")
    st.rerun()


def get_authenticated_client() -> FivePaisaClient | None:
    if st.session_state.get(SESSION_STATE_KEY_LOGGED_IN_STATUS):
        client = st.session_state.get(SESSION_STATE_KEY_CLIENT)
        if isinstance(client, FivePaisaClient):
            return client
        logger.warning("Logged‑in flag set but client missing; resetting flag.")
        st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False
    return None


def is_user_logged_in() -> bool:
    return bool(st.session_state.get(SESSION_STATE_KEY_LOGGED_IN_STATUS))

# ─────────────────────────── Streamlit UI component ────────────────────

def display_login_form(cols=None):
    """Sidebar widget for OAuth login."""
    container = cols[0] if cols else st.sidebar
    with container.expander("🔒 **5paisa Login**", expanded=not is_user_logged_in()):
        if is_user_logged_in():
            st.success(f"Logged in as **{st.session_state.get(SESSION_STATE_KEY_CLIENT_CODE)}**")
            if st.button("Logout", key="logout_btn"):
                logout()
            return

        if not _load_credentials_from_env():
            st.warning("Please set API credentials in `.env`."); return

        oauth_url = generate_oauth_login_url()
        if not oauth_url:
            st.error("Failed to build OAuth URL."); return

        st.markdown(f"[**Open 5paisa Login Page**]({oauth_url})", help="Opens in new tab")
        st.info("After login, copy the `RequestToken` from the redirected URL and paste below.")

        with st.form("oauth_form"):
            token_input = st.text_input("RequestToken", help="Paste the token here")
            if st.form_submit_button("Login"):
                if not token_input:
                    st.error("RequestToken is required.")
                elif login_via_oauth_request_token(token_input):
                    st.rerun()

# ─────────────────────────── Stand‑alone test ──────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="Auth Test", layout="wide")
    st.title("5paisa OAuth Authentication Test")
    display_login_form()
    st.divider(); st.json(dict(st.session_state), expanded=False)
