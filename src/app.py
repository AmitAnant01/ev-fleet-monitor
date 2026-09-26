"""
app.py
------
Entry point. Run with: streamlit run src/app.py

Applies the global theme, handles sign-in/sign-up, routes to the
correct dashboard by role.
"""

import streamlit as st
import pandas as pd
import os

from theme import apply_theme
from auth import login_page, logout_button, account_sidebar
from admin_dashboard import show_admin_dashboard
from driver_dashboard import show_driver_dashboard

st.set_page_config(page_title="EV Fleet Monitor", layout="wide", page_icon="🔋")
apply_theme()

DATA_PATH = os.path.join("data", "processed", "Processed_EV_Data.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


def main():
    # Data is loaded before auth so the Sign Up form can offer a real
    # "which car do you drive" dropdown sourced from the fleet dataset.
    df = load_data()

    logged_in, role = login_page(df)

    if not logged_in:
        return

    account_sidebar()
    logout_button()
    st.sidebar.divider()

    if role == "Admin":
        show_admin_dashboard(df)
    elif role == "Driver":
        show_driver_dashboard(df)


if __name__ == "__main__":
    main()
