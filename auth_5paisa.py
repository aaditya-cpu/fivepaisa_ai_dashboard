# fivepaisa_ai_dashboard/auth_5paisa.py

"""
Authentication Module for 5paisa API

Handles the login process, token generation, and session management
for interacting with the 5paisa trading APIs using the `py5paisa` SDK.

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

# Load environment variables from .env file
load_dotenv()

# Import configurations
from config import APP_TITLE # For consistent logging/messaging

# Configure a dedicated logger for this module
logger = logging.getLogger(f"{APP_TITLE}.Auth")
# Basic logging configuration (can be expanded in a central logging setup if needed)
if not logger.handlers: # Avoid adding multiple handlers if reloaded
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Constants for Session State ---
SESSION_STATE_KEY_CLIENT = "5paisa_client_instance"
SESSION_STATE_KEY_CLIENT_CODE = "5paisa_client_code"
SESSION_STATE_KEY_ACCESS_TOKEN = "5paisa_access_token"
SESSION_STATE_KEY_REQUEST_TOKEN = "5paisa_request_token" # For OAuth, less used with TOTP
SESSION_STATE_KEY_LOGGED_IN_STATUS = "5paisa_logged_in"
SESSION_STATE_KEY_USER_KEY = "5paisa_user_key" # Storing for potential use with Access Token generation

# --- Helper Functions ---

def _load_credentials_from_secrets() -> dict | None:
    """
    (Deprecated) Load API credentials from Streamlit's ``secrets.toml``.
    Kept for backward compatibility but not used.
    """
    required_secrets = [
        "APP_NAME", "APP_SOURCE", "USER_ID",
        "PASSWORD", "USER_KEY", "ENCRYPTION_KEY"
    ]
    missing_secrets = [key for key in required_secrets if key not in st.secrets]

    if missing_secrets:
        logger.error(f"Missing required API credentials in secrets.toml: {', '.join(missing_secrets)}")
        st.error(
            f"Critical API credentials missing in `.streamlit/secrets.toml`: {', '.join(missing_secrets)}. "
            "Please configure them to proceed."
        )
        return None

    cred = {
        "APP_NAME": st.secrets["APP_NAME"],
        "APP_SOURCE": st.secrets["APP_SOURCE"],
        "USER_ID": st.secrets["USER_ID"],
        "PASSWORD": st.secrets["PASSWORD"],
        "USER_KEY": st.secrets["USER_KEY"],
        "ENCRYPTION_KEY": st.secrets["ENCRYPTION_KEY"]
    }
    if SESSION_STATE_KEY_USER_KEY not in st.session_state:
        st.session_state[SESSION_STATE_KEY_USER_KEY] = st.secrets["USER_KEY"]
    logger.info("Successfully loaded API credentials from secrets.")
    return cred

def _get_totp_login_details_from_secrets() -> tuple[str | None, str | None]:
    """
    (Deprecated) Load Client Code and PIN for TOTP login from ``secrets.toml``.
    Kept for backward compatibility but not used.
    """
    client_code = st.secrets.get("CLIENT_CODE")
    pin = st.secrets.get("PIN")

    if not client_code:
        logger.error("CLIENT_CODE missing from secrets.toml for TOTP login.")
        st.error("CLIENT_CODE is missing in `.streamlit/secrets.toml`. This is required for login.")
    if not pin:
        logger.error("PIN missing from secrets.toml for TOTP login.")
        st.error("PIN (MPIN) is missing in `.streamlit/secrets.toml`. This is required for login.")

    return client_code, pin

def _load_credentials_from_env() -> dict | None:
    """Loads 5paisa API credentials from environment variables (.env)."""
    required_envs = [
        "APP_NAME", "APP_SOURCE", "USER_ID",
        "PASSWORD", "USER_KEY", "ENCRYPTION_KEY"
    ]
    missing_envs = [key for key in required_envs if not os.getenv(key)]

    if missing_envs:
        logger.error(
            f"Missing required API credentials in .env: {', '.join(missing_envs)}"
        )
        st.error(
            f"Critical API credentials missing in `.env`: {', '.join(missing_envs)}. "
            "Please configure them to proceed."
        )
        return None

    cred = {
        "APP_NAME": os.getenv("APP_NAME"),
        "APP_SOURCE": os.getenv("APP_SOURCE"),
        "USER_ID": os.getenv("USER_ID"),
        "PASSWORD": os.getenv("PASSWORD"),
        "USER_KEY": os.getenv("USER_KEY"),
        "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY"),
    }
    if SESSION_STATE_KEY_USER_KEY not in st.session_state:
        st.session_state[SESSION_STATE_KEY_USER_KEY] = os.getenv("USER_KEY")
    logger.info("Successfully loaded API credentials from .env.")
    return cred


def _get_totp_login_details_from_env() -> tuple[str | None, str | None]:
    """Loads Client Code and PIN for TOTP login from environment variables."""
    client_code = os.getenv("CLIENT_CODE")
    pin = os.getenv("PIN")

    if not client_code:
        logger.error("CLIENT_CODE missing from .env for TOTP login.")
        st.error("CLIENT_CODE is missing in `.env`. This is required for login.")
    if not pin:
        logger.error("PIN missing from .env for TOTP login.")
        st.error("PIN (MPIN) is missing in `.env`. This is required for login.")

    return client_code, pin


# --- Main Authentication Functions ---

def login_via_totp_session(totp_code: str) -> bool:
    """
    Attempts to log in using the TOTP session method.
    Stores client instance and tokens in session_state on success.

    Args:
        totp_code (str): The TOTP code from the authenticator app.

    Returns:
        bool: True if login was successful, False otherwise.
    """
    st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False # Reset status

    creds = _load_credentials_from_env()
    if not creds:
        return False

    client_code_from_env, pin_from_env = _get_totp_login_details_from_env()
    if not client_code_from_env or not pin_from_env:
        return False

    try:
        with st.spinner("Attempting login with 5paisa... Please wait."):
            client = FivePaisaClient(cred=creds)
            logger.info(f"Attempting TOTP login for Client Code: {client_code_from_env}")

            # The py5paisa get_totp_session directly sets up the access token internally.
            # It implicitly handles the Request Token -> Access Token flow.
            # The response of get_totp_session is usually the underlying raw API response.
            login_response = client.get_totp_session(
                ClientCode=client_code_from_env,
                TOTP=totp_code,
                PIN=pin_from_env
            )
            logger.debug(f"Raw 5paisa TOTP Login API Response: {login_response}")

            # py5paisa's get_totp_session, upon success, should make the client usable.
            # We should verify this by trying to fetch the access token or making a simple call.
            # The access token is stored within the client object by py5paisa.
            access_token = client.get_access_token() # This should return the token if login was successful

            if access_token and client.client_code: # client.client_code is also set by py5paisa
                st.session_state[SESSION_STATE_KEY_CLIENT] = client
                st.session_state[SESSION_STATE_KEY_CLIENT_CODE] = client.client_code
                st.session_state[SESSION_STATE_KEY_ACCESS_TOKEN] = access_token
                st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = True
                logger.info(f"Login successful for Client Code: {client.client_code}. Access Token obtained.")
                st.success(f"Successfully logged in as Client Code: {client.client_code}")
                return True
            else:
                # Attempt to parse specific error messages if possible
                # Based on 5paisa documentation, the response from TOTPLogin includes:
                # Status: 0 (Success), 1 (Invalid login/password), 2 (OTP used/invalid TOTP)
                # Message: Description string
                status = login_response.get("body", {}).get("Status")
                message = login_response.get("body", {}).get("Message", "Unknown login error.")

                if status == 1:
                    error_msg = f"Login Failed: Invalid credentials or PIN. ({message})"
                elif status == 2:
                    error_msg = f"Login Failed: Invalid or expired TOTP. ({message})"
                else:
                    error_msg = f"Login Failed: {message} (Status: {status})"

                logger.error(error_msg)
                st.error(error_msg)
                return False

    except Exception as e:
        logger.critical(f"An unexpected error occurred during TOTP login: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}. Check logs for details.")
        return False

def logout():
    """
    Clears session state related to 5paisa login.
    Note: This does not invalidate the AccessToken on the 5paisa server side immediately,
          but removes it from the current browser session.
    """
    keys_to_clear = [
        SESSION_STATE_KEY_CLIENT,
        SESSION_STATE_KEY_CLIENT_CODE,
        SESSION_STATE_KEY_ACCESS_TOKEN,
        SESSION_STATE_KEY_REQUEST_TOKEN,
        SESSION_STATE_KEY_LOGGED_IN_STATUS,
        SESSION_STATE_KEY_USER_KEY
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("User logged out. Session cleared.")
    st.info("You have been logged out.")
    st.rerun() # Rerun to reflect logout state in UI

def get_authenticated_client() -> FivePaisaClient | None:
    """
    Retrieves the authenticated 5paisa client instance from session state.
    Returns None if not logged in or client not found.
    """
    if st.session_state.get(SESSION_STATE_KEY_LOGGED_IN_STATUS, False):
        client = st.session_state.get(SESSION_STATE_KEY_CLIENT)
        if client and isinstance(client, FivePaisaClient):
            # Optionally, you can add a check here to see if the token is still valid
            # by making a lightweight API call, but this adds overhead.
            # For now, we assume if it's in session, it's good.
            return client
        else:
            logger.warning("Logged in status is true, but client instance not found or invalid in session.")
            st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False # Correct the state
            return None
    return None

def is_user_logged_in() -> bool:
    """Checks if the user is currently marked as logged in."""
    return st.session_state.get(SESSION_STATE_KEY_LOGGED_IN_STATUS, False)

# --- UI Component for Login ---
def display_login_form(cols=None):
    """
    Displays the login form (TOTP input) in the Streamlit UI.
    Uses provided columns for layout if available.
    """
    if cols:
        login_container = cols[0] # Or st depending on layout
    else:
        login_container = st.sidebar # Default to sidebar

    with login_container.expander("🔒 **5paisa Login**", expanded=not is_user_logged_in()):
        if is_user_logged_in():
            client_code = st.session_state.get(SESSION_STATE_KEY_CLIENT_CODE, "N/A")
            st.success(f"Logged in as: **{client_code}**")
            if st.button("Logout", key="auth_logout_button"):
                logout()
            return

        # Ensure base credentials are configurable before showing login form
        creds_check = _load_credentials_from_env()
        client_code_check, pin_check = _get_totp_login_details_from_env()
        if not creds_check or not client_code_check or not pin_check:
            st.warning(
                "Please ensure all API credentials, Client Code, and PIN are correctly set in your `.env` file."
            )
            return  # Don't show form if base credentials are missing

        st.markdown("Enter your TOTP from your authenticator app.")
        with st.form(key="login_form"):
            totp_input = st.text_input(
                "TOTP Code",
                max_chars=6,
                type="password",
                placeholder="******",
                help="6-digit code from Google Authenticator or similar app."
            )
            submit_button = st.form_submit_button(label="Login")

            if submit_button:
                if not totp_input or not totp_input.isdigit() or len(totp_input) != 6:
                    st.error("Please enter a valid 6-digit TOTP.")
                else:
                    if login_via_totp_session(totp_input):
                        st.rerun() # Rerun to update UI state post-login
                    # Error messages are handled within login_via_totp_session

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    st.set_page_config(page_title="Auth Test", layout="wide")
    st.title("5paisa Authentication Module Test")

    st.info("This page tests the authentication module. Configure secrets in `.streamlit/secrets.toml`.")

    # Initialize session state keys if not present for testing
    if SESSION_STATE_KEY_LOGGED_IN_STATUS not in st.session_state:
        st.session_state[SESSION_STATE_KEY_LOGGED_IN_STATUS] = False

    display_login_form() # Display in sidebar

    st.divider()
    st.subheader("Session State Inspector:")
    st.json(st.session_state)

    if is_user_logged_in():
        st.success("User is logged in!")
        client = get_authenticated_client()
        if client:
            st.write("Authenticated client instance retrieved.")
            try:
                with st.spinner("Fetching holdings as a test..."):
                    holdings = client.holdings() # Example API call
                    st.write("Holdings Response:")
                    st.json(holdings if holdings else "No holdings data or error.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        else:
            st.error("Could not retrieve authenticated client instance.")
    else:
        st.warning("User is not logged in.")