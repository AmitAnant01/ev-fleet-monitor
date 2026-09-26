"""
driver_dashboard.py
--------------------
Driver-facing view — pro upgrade.

Tabs:
  1. Overview        — hero card, live status, quick stats, alerts
  2. Trip History     — filterable trip table, trend charts, CSV export
  3. Driving Score     — eco-driving gauge + safe/rash breakdown + tips
  4. Costs             — maintenance & charging spend, fleet comparison
  5. Range Prediction   — ML range predictor (original logic, kept intact)
  6. My Profile         — account details, edit profile, change password

Design notes carried over from v1 (still true, don't "fix" these):
- HTML blocks built as flush-left string concatenation, never an
  indented multi-line string — 4+ spaces after a blank line reads as a
  Markdown code block and the HTML shows up as literal text.
- st.container(border=True) instead of a fake markdown <div>, for the
  same reach-forward reason documented in auth.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

import db

MODEL_PATH = os.path.join("src", "models", "range_predictor.pkl")

STATUS_COLORS = {
    "Running": ("#e6f7ec", "#1e7e34"),
    "Charging": ("#e8f2ff", "#0071e3"),
    "Idle": ("#f2f2f4", "#6e6e73"),
    "Under Maintenance": ("#fdeceb", "#d70015"),
}


# ---------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------

def _styles():
    st.markdown("""
        <style>
        .driver-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 28px 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }
        .car-name { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; color: #1d1d1f; }
        .car-sub { font-size: 14px; color: #86868b; margin-bottom: 14px; }
        .status-pill { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; }
        .battery-track { width: 100%; height: 14px; background: #f2f2f4; border-radius: 999px; overflow: hidden; margin-top: 8px; }
        .battery-fill { height: 100%; border-radius: 999px; }
        .location-line { font-size: 15px; color: #1d1d1f; margin-top: 16px; }
        .location-label { font-size: 12px; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em; }
        .predict-result { background: linear-gradient(135deg, #0071e3, #00c6ff); border-radius: 20px; padding: 32px; text-align: center; color: white; margin-top: 8px; }
        .predict-result .big-number { font-size: 48px; font-weight: 800; letter-spacing: -0.02em; }
        .predict-result .caption { font-size: 14px; opacity: 0.9; margin-top: 4px; }
        .alert-chip {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 8px 14px; border-radius: 12px; font-size: 13px; font-weight: 600;
            margin-right: 10px; margin-bottom: 10px;
        }
        .alert-danger { background: #fdeceb; color: #d70015; }
        .alert-warning { background: #fff6e5; color: #b26a00; }
        .alert-ok { background: #e6f7ec; color: #1e7e34; }
        .score-badge {
            display: inline-block; padding: 4px 12px; border-radius: 999px;
            font-size: 13px; font-weight: 700;
        }
        .profile-card {
            background: #ffffff; border-radius: 18px; padding: 22px 26px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 18px;
        }
        .profile-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f2f2f4; font-size: 14px; }
        .profile-row:last-child { border-bottom: none; }
        .profile-label { color: #86868b; }
        .profile-value { color: #1d1d1f; font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)


def _battery_color(pct):
    if pct >= 50:
        return "#34c759"
    elif pct >= 20:
        return "#ff9f0a"
    return "#ff3b30"


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _eco_score(car_df: pd.DataFrame) -> float:
    """Composite 0-100 eco-driving score: mostly how often the driver
    stayed 'Safe' rather than 'Rash', with a penalty for harsh events."""
    if car_df.empty:
        return 0.0
    total = len(car_df)
    safe_pct = 100.0 * (car_df["Driving_Style"] == "Safe").sum() / total
    avg_harsh = car_df["Harsh_Events"].mean()
    score = 0.75 * safe_pct + 0.25 * max(0.0, 100.0 - avg_harsh * 20.0)
    return float(np.clip(score, 0, 100))


def _score_color(score):
    if score >= 80:
        return "#1e7e34", "#e6f7ec"
    elif score >= 55:
        return "#b26a00", "#fff6e5"
    return "#d70015", "#fdeceb"


def _gauge_chart(score: float):
    color, _ = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number={"suffix": " / 100", "font": {"size": 34, "color": "#1d1d1f"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#86868b"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 55], "color": "#fdeceb"},
                {"range": [55, 80], "color": "#fff6e5"},
                {"range": [80, 100], "color": "#e6f7ec"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, sans-serif"})
    return fig


def _resolve_default_car(all_cars, assigned_car_id):
    if assigned_car_id and assigned_car_id in all_cars:
        return all_cars.index(assigned_car_id)
    return 0


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def show_driver_dashboard(df: pd.DataFrame):
    _styles()

    st.title("My Vehicle")
    display_name = st.session_state.get("full_name") or st.session_state.get("username", "driver")
    st.caption(f"Welcome back, {display_name}")

    all_cars = sorted(df["Car_ID"].unique())
    assigned_car_id = st.session_state.get("assigned_car_id")
    default_idx = _resolve_default_car(all_cars, assigned_car_id)

    car_id = st.selectbox(
        "Select your car", all_cars, index=default_idx,
        help="Defaults to the car linked to your account. You can browse any car's data here.",
    )
    car_df = df[df["Car_ID"] == car_id].sort_values("Date")
    car_data = car_df.sort_values("Date", ascending=False).iloc[0]

    rash_event_count = int((car_df["Driving_Style"] == "Rash").sum())
    status = car_data["Fleet_Status"]
    battery_pct = float(car_data["Battery_Remaining_Percent"])

    tab_overview, tab_trips, tab_score, tab_costs, tab_predict, tab_profile = st.tabs(
        ["🏠 Overview", "🧭 Trip History", "🌱 Driving Score", "💰 Costs", "🔮 Range Prediction", "👤 My Profile"]
    )

    with tab_overview:
        _render_overview(car_id, car_data, status, battery_pct, rash_event_count)

    with tab_trips:
        _render_trip_history(car_id, car_df)

    with tab_score:
        _render_driving_score(car_id, car_df)

    with tab_costs:
        _render_costs(car_id, car_df, df)

    with tab_predict:
        _render_range_prediction(car_id, car_data)

    with tab_profile:
        _render_profile(all_cars)


# ---------------------------------------------------------------------
# tab: overview
# ---------------------------------------------------------------------

def _render_overview(car_id, car_data, status, battery_pct, rash_event_count):
    bg_color, text_color = STATUS_COLORS.get(status, ("#f2f2f4", "#6e6e73"))
    battery_color = _battery_color(battery_pct)

    # ---------- Alerts ----------
    alerts = []
    if battery_pct < 20:
        alerts.append(("alert-danger", f"🔋 Low battery — only {battery_pct:.0f}% remaining"))
    if status == "Under Maintenance":
        alerts.append(("alert-danger", "🛠️ Vehicle is under maintenance"))
    if car_data["Driving_Style"] == "Rash":
        alerts.append(("alert-warning", f"⚠️ Latest reading shows rash driving at {car_data['Speed_kmph']:.0f} km/h"))
    if not alerts:
        alerts.append(("alert-ok", "✅ No active alerts — vehicle looks healthy"))

    alerts_html = "".join(f'<span class="alert-chip {cls}">{text}</span>' for cls, text in alerts)
    st.markdown(alerts_html, unsafe_allow_html=True)

    hero_html = (
        '<div class="driver-card">'
        f'<div class="car-name">{car_data["Company"]} {car_data["Model"]}</div>'
        f'<div class="car-sub">{car_id} &middot; Driver: {car_data["Driver_Name"]} &middot; '
        f'{int(car_data["Seating_Capacity"])} seats</div>'
        f'<span class="status-pill" style="background:{bg_color}; color:{text_color};">{status}</span>'
        '<div class="location-label" style="margin-top:20px;">Battery Remaining</div>'
        f'<div style="font-size:22px; font-weight:700; color:#1d1d1f;">{battery_pct:.1f}%</div>'
        '<div class="battery-track">'
        f'<div class="battery-fill" style="width:{battery_pct}%; background:{battery_color};"></div>'
        '</div>'
        '<div class="location-label" style="margin-top:20px;">Current Location</div>'
        f'<div class="location-line">{car_data["Location"]}, {car_data["City"]}</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Speed", f"{car_data['Speed_kmph']:.0f} km/h")
    col2.metric("Trip Distance", f"{car_data['Distance_km']:.0f} km")
    col3.metric("Driving Mode", car_data["Driving_Mode"])

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Odometer", f"{car_data['Total_Odometer_km']:,.0f} km")
    col5.metric("Motor Power", f"{car_data['Motor_Power_kW']:.0f} kW")
    col6.metric("Torque", f"{car_data['Torque_Nm']:.0f} Nm")

    col7, col8, col9 = st.columns(3)
    col7.metric("Car Max Range (spec)", f"{car_data['Car_Max_Range_km']:.0f} km")
    col8.metric("Car Weight", f"{car_data['Car_Weight_kg']:.0f} kg")
    driving_style_icon = "⚠️ Rash" if car_data["Driving_Style"] == "Rash" else "✅ Safe"
    col9.metric("Driving Style (now)", driving_style_icon)

    col10, _, _ = st.columns(3)
    col10.metric("Times Reached 110+ km/h (all-time)", f"{rash_event_count}")
    st.caption(f"'Driving Style (now)' above reflects only the latest reading ({car_data['Speed_kmph']:.0f} km/h) — "
               f"a car can have past rash events but be driving safely right now, or vice versa.")
    st.divider()

    st.subheader("Current Trip")
    trip_col1, trip_col2, trip_col3 = st.columns(3)
    trip_col1.metric("Trip Status", car_data["Trip_Status"])
    trip_col2.metric("Minutes Elapsed", f"{int(car_data['Minutes_Elapsed'])} / {int(car_data['Trip_Duration_Minutes'])} min")
    trip_col3.metric("Recognized Revenue", f"₹{car_data['Recognized_Revenue']:,.0f}")
    if car_data["Trip_Status"] == "In Transit":
        st.caption("Revenue is withheld until the trip reaches 100% completion.")


# ---------------------------------------------------------------------
# tab: trip history
# ---------------------------------------------------------------------

def _render_trip_history(car_id, car_df):
    st.subheader(f"Trip History — {car_id}")

    months = ["All"] + sorted(car_df["Month"].unique().tolist(), key=lambda m: car_df[car_df["Month"] == m]["Month_Num"].iloc[0])
    col_a, col_b = st.columns([1, 3])
    with col_a:
        month_filter = st.selectbox("Month", months, key="trip_month_filter")

    filtered = car_df if month_filter == "All" else car_df[car_df["Month"] == month_filter]
    filtered = filtered.sort_values("Date")

    if filtered.empty:
        st.info("No trips found for this filter.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trips (records)", f"{len(filtered)}")
    c2.metric("Total Distance", f"{filtered['Distance_km'].sum():,.0f} km")
    c3.metric("Avg Speed", f"{filtered['Speed_kmph'].mean():.0f} km/h")
    c4.metric("Total Revenue", f"₹{filtered['Recognized_Revenue'].sum():,.0f}")

    fig_battery = px.line(
        filtered, x="Date", y="Battery_Remaining_Percent",
        title="Battery Remaining % Over Time", markers=True,
    )
    fig_battery.update_traces(line_color="#0071e3")
    fig_battery.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"family": "Inter, sans-serif"})
    st.plotly_chart(fig_battery, use_container_width=True)

    daily_distance = filtered.groupby(filtered["Date"].dt.date)["Distance_km"].sum().reset_index()
    fig_distance = px.bar(
        daily_distance, x="Date", y="Distance_km", title="Distance Driven Per Day",
    )
    fig_distance.update_traces(marker_color="#34c759")
    fig_distance.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font={"family": "Inter, sans-serif"})
    st.plotly_chart(fig_distance, use_container_width=True)

    st.markdown("##### Trip Log")
    display_cols = [
        "Date", "Trip_Status", "Distance_km", "Speed_kmph", "Driving_Style",
        "Driving_Mode", "Trip_Duration_Minutes", "Recognized_Revenue",
        "Charging_Cost_INR", "Exact_Maintenance_Cost_INR",
    ]
    table = filtered[display_cols].sort_values("Date", ascending=False).reset_index(drop=True)
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download trip log as CSV", data=csv_bytes,
        file_name=f"{car_id}_trip_history.csv", mime="text/csv",
    )


# ---------------------------------------------------------------------
# tab: driving score
# ---------------------------------------------------------------------

def _render_driving_score(car_id, car_df):
    st.subheader(f"Driving Score — {car_id}")
    score = _eco_score(car_df)
    color, bg = _score_color(score)

    col_gauge, col_breakdown = st.columns([1.1, 1])

    with col_gauge:
        st.plotly_chart(_gauge_chart(score), use_container_width=True)
        label = "Excellent" if score >= 80 else "Needs Improvement" if score >= 55 else "Poor"
        st.markdown(
            f'<span class="score-badge" style="background:{bg}; color:{color};">{label}</span>',
            unsafe_allow_html=True,
        )

    with col_breakdown:
        safe_count = int((car_df["Driving_Style"] == "Safe").sum())
        rash_count = int((car_df["Driving_Style"] == "Rash").sum())
        style_df = pd.DataFrame({"Style": ["Safe", "Rash"], "Count": [safe_count, rash_count]})
        fig_pie = px.pie(
            style_df, names="Style", values="Count", hole=0.55,
            color="Style", color_discrete_map={"Safe": "#34c759", "Rash": "#ff3b30"},
            title="Safe vs Rash Driving (all-time)",
        )
        fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, sans-serif"})
        st.plotly_chart(fig_pie, use_container_width=True)

        st.metric("Avg Harsh Events / Trip", f"{car_df['Harsh_Events'].mean():.2f}")

    st.divider()
    monthly = car_df.groupby("Month_Num").agg(
        Harsh_Events=("Harsh_Events", "mean"),
        Rash_Trips=("Driving_Style", lambda s: (s == "Rash").sum()),
    ).reset_index().sort_values("Month_Num")
    if not monthly.empty:
        fig_trend = px.bar(monthly, x="Month_Num", y="Rash_Trips", title="Rash-Driving Trips by Month")
        fig_trend.update_traces(marker_color="#ff9f0a")
        fig_trend.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font={"family": "Inter, sans-serif"},
                                 xaxis_title="Month #", yaxis_title="Rash trips")
        st.plotly_chart(fig_trend, use_container_width=True)

    with st.expander("💡 Tips to improve your score"):
        st.markdown(
            "- Keep speeds under **110 km/h** — that's the threshold counted as rash driving.\n"
            "- Smooth acceleration and braking reduces **harsh events**, which also extends battery life.\n"
            "- Highway mode at steady speed is generally more efficient than frequent city stop-and-go.\n"
            "- A consistently high score can factor into fleet performance reviews."
        )


# ---------------------------------------------------------------------
# tab: costs
# ---------------------------------------------------------------------

def _render_costs(car_id, car_df, full_df):
    st.subheader(f"Costs — {car_id}")

    total_maint = car_df["Exact_Maintenance_Cost_INR"].sum()
    total_charge = car_df["Charging_Cost_INR"].sum()
    fleet_avg_maint = full_df.groupby("Car_ID")["Exact_Maintenance_Cost_INR"].sum().mean()
    fleet_avg_charge = full_df.groupby("Car_ID")["Charging_Cost_INR"].sum().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Maintenance Cost", f"₹{total_maint:,.0f}",
              delta=f"{total_maint - fleet_avg_maint:+,.0f} vs fleet avg", delta_color="inverse")
    c2.metric("Total Charging Cost", f"₹{total_charge:,.0f}",
              delta=f"{total_charge - fleet_avg_charge:+,.0f} vs fleet avg", delta_color="inverse")
    c3.metric("Total Revenue Generated", f"₹{car_df['Recognized_Revenue'].sum():,.0f}")
    net = car_df['Recognized_Revenue'].sum() - total_maint - total_charge
    c4.metric("Net (Revenue − Costs)", f"₹{net:,.0f}")

    monthly_costs = car_df.groupby("Month_Num").agg(
        Maintenance=("Exact_Maintenance_Cost_INR", "sum"),
        Charging=("Charging_Cost_INR", "sum"),
    ).reset_index().sort_values("Month_Num")

    if not monthly_costs.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_costs["Month_Num"], y=monthly_costs["Maintenance"],
                              name="Maintenance", marker_color="#d70015"))
        fig.add_trace(go.Bar(x=monthly_costs["Month_Num"], y=monthly_costs["Charging"],
                              name="Charging", marker_color="#0071e3"))
        fig.update_layout(
            barmode="group", title="Monthly Cost Breakdown", height=340,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif"}, xaxis_title="Month #", yaxis_title="₹",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Fleet average is computed across all 50 cars' full-history totals, for context on how this "
               "vehicle compares.")


# ---------------------------------------------------------------------
# tab: range prediction  (original logic kept intact)
# ---------------------------------------------------------------------

def _render_range_prediction(car_id, car_data):
    st.subheader("Range Prediction")

    manual_battery_pct = st.slider(
        "Battery Level (%)",
        min_value=0, max_value=100,
        value=int(round(car_data["Battery_Remaining_Percent"])),
        help="Adjust this to see predicted remaining range at any battery level — this value goes directly into the model, not into a formula.",
    )

    if st.button("Predict Range", type="primary"):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]

        row = pd.DataFrame([{
            "Battery_Remaining_Percent": manual_battery_pct,
            "Speed_kmph": car_data["Speed_kmph"],
            "Distance_km": car_data["Distance_km"],
            "Harsh_Events": car_data["Harsh_Events"],
            "Battery_Capacity_kWh": car_data["Battery_Capacity_kWh"],
            "Car_Max_Range_km": car_data["Car_Max_Range_km"],
            "Car_Weight_kg": car_data["Car_Weight_kg"],
            "Seating_Capacity": car_data["Seating_Capacity"],
            "Motor_Power_kW": car_data["Motor_Power_kW"],
            "Torque_Nm": car_data["Torque_Nm"],
            "Total_Odometer_km": car_data["Total_Odometer_km"],
        }])
        row["Driving_Mode_Highway"] = 1 if car_data["Driving_Mode"] == "Highway" else 0
        row = row.reindex(columns=feature_columns, fill_value=0)

        if manual_battery_pct <= 0:
            predicted_remaining_km = 0.0
        else:
            predicted_remaining_km = max(0.0, model.predict(row)[0])

        result_html = (
            '<div class="predict-result">'
            f'<div class="big-number">{predicted_remaining_km:.0f} km</div>'
            f'<div class="caption">ML-predicted remaining range for {car_id} at {manual_battery_pct}% battery</div>'
            '</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

        compare_df = pd.DataFrame({
            "Metric": ["Predicted (this battery %)", "Car Spec Max Range", "Last Known Actual Range"],
            "km": [predicted_remaining_km, car_data["Car_Max_Range_km"], car_data["Actual_Remaining_Range_km"]],
        })
        fig = px.bar(compare_df, x="Metric", y="km", color="Metric",
                     color_discrete_sequence=["#0071e3", "#86868b", "#34c759"],
                     title="Predicted vs Spec vs Last Actual Range")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"family": "Inter, sans-serif"})
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# tab: my profile
# ---------------------------------------------------------------------

def _render_profile(all_cars):
    username = st.session_state.get("username")
    user = db.get_user(username)
    if user is None:
        st.warning("Could not load your profile.")
        return

    st.subheader("My Profile")

    profile_html = (
        '<div class="profile-card">'
        f'<div class="profile-row"><span class="profile-label">Username</span><span class="profile-value">{user["username"]}</span></div>'
        f'<div class="profile-row"><span class="profile-label">Full name</span><span class="profile-value">{user["full_name"] or "—"}</span></div>'
        f'<div class="profile-row"><span class="profile-label">Email</span><span class="profile-value">{user["email"] or "—"}</span></div>'
        f'<div class="profile-row"><span class="profile-label">Role</span><span class="profile-value">{user["role"]}</span></div>'
        f'<div class="profile-row"><span class="profile-label">Assigned car</span><span class="profile-value">{user["assigned_car_id"] or "—"}</span></div>'
        f'<div class="profile-row"><span class="profile-label">Member since</span><span class="profile-value">{str(user["created_at"])[:19]}</span></div>'
        f'<div class="profile-row"><span class="profile-label">Last login</span><span class="profile-value">{user["last_login"] or "This is your first login"}</span></div>'
        '</div>'
    )
    st.markdown(profile_html, unsafe_allow_html=True)

    col_edit, col_pwd = st.columns(2)

    with col_edit:
        st.markdown("##### Edit Profile")
        with st.form("edit_profile_form"):
            new_name = st.text_input("Full name", value=user["full_name"])
            new_email = st.text_input("Email", value=user["email"])
            new_car = st.selectbox(
                "Assigned car", all_cars,
                index=all_cars.index(user["assigned_car_id"]) if user["assigned_car_id"] in all_cars else 0,
            )
            save = st.form_submit_button("Save changes", type="primary", use_container_width=True)
        if save:
            success, message = db.update_profile(username, full_name=new_name, email=new_email, assigned_car_id=new_car)
            if success:
                st.session_state.full_name = new_name
                st.session_state.assigned_car_id = new_car
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with col_pwd:
        st.markdown("##### Change Password")
        with st.form("change_password_form"):
            current_pwd = st.text_input("Current password", type="password")
            new_pwd = st.text_input("New password", type="password")
            confirm_pwd = st.text_input("Confirm new password", type="password")
            change = st.form_submit_button("Update password", type="primary", use_container_width=True)
        if change:
            success, message = db.change_password(username, current_pwd, new_pwd, confirm_pwd)
            if success:
                st.success(message)
            else:
                st.error(message)
