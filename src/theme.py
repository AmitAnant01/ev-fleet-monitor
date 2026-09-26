"""
theme.py
--------
Global visual style for the whole app — clean, minimal, high-contrast,
inspired by Apple's product pages: soft gray background, white cards,
tight typography, single blue accent color, generous whitespace.

Call apply_theme() once at the top of app.py, before rendering anything.
"""

import streamlit as st

ACCENT = "#0071e3"
INK = "#1d1d1f"
MUTED = "#86868b"
BG = "#f5f5f7"


def apply_theme():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background: {BG};
        }}

        /* Hide Streamlit chrome for a cleaner, branded look */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{background: transparent;}}

        /* Headings */
        h1 {{
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            color: {INK} !important;
        }}
        h2, h3 {{
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: {INK} !important;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: #ffffff;
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.04);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {MUTED} !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {INK} !important;
            font-weight: 700 !important;
        }}

        /* Buttons */
        .stButton button {{
            border-radius: 10px;
            font-weight: 600;
            border: none;
            background: {INK};
            color: white;
            transition: background 0.15s ease;
        }}
        .stButton button:hover {{
            background: {ACCENT};
            color: white;
        }}
        .stButton button[kind="primary"] {{
            background: {ACCENT};
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid rgba(0,0,0,0.06);
        }}

        /* Select boxes / inputs */
        .stSelectbox div[data-baseweb="select"], .stTextInput input {{
            border-radius: 10px !important;
        }}

        /* Dataframes */
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(0,0,0,0.06);
        }}

        /* Divider */
        hr {{
            border-color: rgba(0,0,0,0.08) !important;
        }}
        </style>
    """, unsafe_allow_html=True)