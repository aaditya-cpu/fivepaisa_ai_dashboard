+61-0
# 5paisa AI Predictive Dashboard

This project is an experimental Streamlit dashboard for exploring trading data from the [5paisa](https://www.5paisa.com/) API.  It provides a foundation for building charts, fetching historical prices, computing technical indicators and, eventually, predictive models.

## Features

- **TOTP based authentication** with the `py5paisa` SDK
- Modular layout for data handling, plotting and utilities
- Ready for additional predictive models and pattern detection

## Setup

1. **Create a virtual environment** (recommended)
    ```bash
    python3 -m venv stocky
    source stocky/bin/activate
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Add your API credentials** in `.streamlit/secrets.toml`:
    ```toml
    APP_NAME = "YOUR_APP_NAME"
    APP_SOURCE = "YOUR_APP_SOURCE"
    USER_ID = "YOUR_USER_ID"
    PASSWORD = "YOUR_PASSWORD"
    USER_KEY = "YOUR_USER_KEY"
    ENCRYPTION_KEY = "YOUR_ENCRYPTION_KEY"
    CLIENT_CODE = "YOUR_CLIENT_CODE"
    PIN = "YOUR_PIN"
    ```
    These values come from your 5paisa account. They are required for the login form shown in the sidebar.

## Running the dashboard

Start Streamlit with:

```bash
streamlit run app.py
```

The app runs on <http://localhost:8501> by default.

## Development and testing

Check that all modules compile and that code style passes:

```bash
python -m py_compile auth_5paisa.py app.py data_handler.py predictive_models.py config.py plotting.py patterns.py utils.py
flake8
```

Install `flake8` first if it is not available:

```bash
pip install flake8
```

---
This repository is an early work in progress. Many modules are placeholders for future functionality. Refer to the source code comments for more details on their intended scope.