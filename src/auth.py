"""
auth.py
-------
Sign In / Sign Up screen, backed by db.py (SQLite — real persistence).

v2 (pro upgrade):
- Sign up now collects full name, email (optional but validated) and,
  for Driver accounts, the real Car_ID they drive — pulled from the
  fleet dataset so the dropdown is never out of sync with reality.
- Confirm-password field with a match check + a simple strength hint.
- Friendly inline validation errors instead of one generic message.
- Disabled-account handling (an admin can deactivate a user).
- Sidebar "Account" block (avatar initial, name, role, last login)
  used by app.py once someone's signed in.

IMPORTANT: uses st.container(border=True) for the card, not a raw
st.markdown('<div>...') — Streamlit widgets called after a markdown
div don't actually render inside that div (it can't reach forward),
so a fake div just floats there empty. st.container(border=True) is
a real DOM element everything inside "with card:" renders into.
"""

import streamlit as st
import pandas as pd
import db


def _auth_styles():
    st.markdown("""
        <style>
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }
        .auth-title {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #1d1d1f;
            margin-bottom: 2px;
        }
        .auth-subtitle {
            font-size: 14px;
            color: #86868b;
            margin-bottom: 20px;
        }
        .auth-hint {
            font-size: 12px;
            color: #86868b;
            margin-top: -8px;
            margin-bottom: 10px;
        }
        .auth-demo {
            font-size: 12px;
            color: #86868b;
            background: #f5f5f7;
            border-radius: 10px;
            padding: 10px 14px;
            margin-top: 16px;
        }
        .account-block {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 4px 14px 4px;
        }
        .account-avatar {
            width: 38px; height: 38px; border-radius: 50%;
            background: #0071e3; color: white;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 15px; flex-shrink: 0;
        }
        .account-name { font-weight: 700; font-size: 14px; color: #1d1d1f; }
        .account-role { font-size: 12px; color: #86868b; }
        </style>
    """, unsafe_allow_html=True)


def _password_strength(password: str) -> str:
    if not password:
        return ""
    score = 0
    if len(password) >= 8:
        score += 1
    if any(ch.isupper() for ch in password):
        score += 1
    if any(ch.isdigit() for ch in password):
        score += 1
    if any(not ch.isalnum() for ch in password):
        score += 1
    return ["Weak", "Weak", "Okay", "Good", "Strong"][score]


def login_page(df: pd.DataFrame = None):
    """
    Renders the Sign In / Sign Up card. Returns (is_logged_in, role).
    Call this once at the top of app.py before showing any dashboard.

    `df` (optional): the fleet dataframe, used only to populate the
    "which car do you drive" dropdown on sign up with real Car_IDs.
    """
    db.init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.full_name = None
        st.session_state.assigned_car_id = None

    if st.session_state.logged_in:
        return True, st.session_state.role

    _auth_styles()

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.write("")
        st.write("")
        card = st.container(border=True)
        with card:
            st.markdown('<div class="auth-title">⚡ EV Fleet Monitor</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">Sign in to your fleet account</div>', unsafe_allow_html=True)

            tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])

            # ----------------------------- SIGN IN -----------------------------
            with tab_signin:
                with st.form("signin_form"):
                    username = st.text_input("Username", key="signin_user", placeholder="e.g. driver")
                    password = st.text_input("Password", type="password", key="signin_pass")
                    remember_hint = st.checkbox("Keep me signed in for this session", value=True,
                                                 key="signin_remember",
                                                 help="Streamlit sessions reset on a full browser refresh either way — this just reflects that.")
                    submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        result = db.verify_user(username, password)
                        if result is None:
                            st.error("Incorrect username or password.")
                        elif result.get("error") == "disabled":
                            st.error("This account has been disabled. Contact your fleet admin.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.role = result["role"]
                            st.session_state.username = result["username"]
                            st.session_state.full_name = result["full_name"] or result["username"]
                            st.session_state.assigned_car_id = result["assigned_car_id"]
                            db.log_audit(result["username"], "Signed in")
                            st.rerun()

                st.markdown(
                    '<div class="auth-demo">Demo logins — Admin: <b>admin</b> / <b>admin123</b> '
                    '&nbsp;|&nbsp; Driver: <b>driver</b> / <b>driver123</b></div>',
                    unsafe_allow_html=True,
                )

            # ----------------------------- SIGN UP -----------------------------
            with tab_signup:
                car_options = []
                if df is not None and "Car_ID" in df.columns:
                    car_options = sorted(df["Car_ID"].unique().tolist())

                with st.form("signup_form"):
                    new_full_name = st.text_input("Full name", key="signup_name", placeholder="e.g. Priya Sharma")
                    new_username = st.text_input("Choose a username", key="signup_user", placeholder="letters, numbers, dot, underscore")
                    new_email = st.text_input("Email (optional)", key="signup_email", placeholder="you@example.com")
                    new_role = st.selectbox("Role", ["Driver", "Admin"], key="signup_role")

                    new_car = None
                    if new_role == "Driver":
                        if car_options:
                            new_car = st.selectbox("Which car do you drive?", car_options, key="signup_car")
                        else:
                            st.info("Fleet data not loaded — you can set your car later from your profile.")

                    new_password = st.text_input("Choose a password", type="password", key="signup_pass")
                    strength = _password_strength(st.session_state.get("signup_pass", ""))
                    if strength:
                        st.markdown(f'<div class="auth-hint">Password strength: {strength}</div>', unsafe_allow_html=True)
                    confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")

                    agree = st.checkbox("I confirm my details are accurate", key="signup_agree")
                    signup_submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if signup_submitted:
                    if not agree:
                        st.error("Please confirm your details are accurate before continuing.")
                    else:
                        success, message = db.register_user(
                            username=new_username,
                            password=new_password,
                            confirm_password=confirm_password,
                            role=new_role,
                            full_name=new_full_name,
                            email=new_email,
                            assigned_car_id=new_car or "",
                        )
                        if success:
                            db.log_audit(new_username, "Account created", details=f"role={new_role}")
                            st.success(message + " Switch to the Sign In tab.")
                            st.balloons()
                        else:
                            st.error(message)

    return False, None


def account_sidebar():
    """Small avatar + name/role block shown at the top of the sidebar
    once someone's logged in."""
    name = st.session_state.get("full_name") or st.session_state.get("username", "?")
    role = st.session_state.get("role", "")
    initial = (name or "?").strip()[0].upper()
    st.sidebar.markdown(
        f"""
        <div class="account-block">
            <div class="account-avatar">{initial}</div>
            <div>
                <div class="account-name">{name}</div>
                <div class="account-role">{role}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def logout_button():
    if st.sidebar.button("Log out", use_container_width=True):
        db.log_audit(st.session_state.get("username"), "Signed out")
        for key in ("logged_in", "role", "username", "full_name", "assigned_car_id"):
            st.session_state[key] = False if key == "logged_in" else None
        st.rerun()
