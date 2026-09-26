"""
admin_dashboard.py
-------------------
Fleet manager view, split into four tabs:

  1. Fleet Overview        : status cards, month filter, revenue &
                              maintenance leaderboards, driver-habit
                              breakdown (110 km/h rash-driving rule).
  2. Key Fleet Metrics      : the 4 must-have presentation views,
                              ready with one click.
  3. Driver Lookup          : pick any driver, see everything tied to
                              them — current status/location, safe-vs-
                              rash history, revenue earned, trends, and
                              the full record log.
  4. Car Lookup             : the same rich single-car "hero card" view
                              drivers see on their own dashboard, now
                              available to the admin for ANY car.
  5. Advanced Filters       : full multi-dimensional filtering across
                              drivers/cars, plus a rank/compare tool
                              (highest, lowest, top-N by any metric,
                              including net profit/loss) and CSV export.
  6. Alerts & Health        : a 0-100 fleet health score plus a rule-
                              based alerts feed (low battery, rash-
                              driving drivers, cars at a net loss,
                              maintenance backlog, harsh-event outliers).
  7. Predictive Maintenance : a 0-100 wear-risk score per car (odometer,
                              harsh events, battery discharge depth,
                              maintenance spend) with recommendations.
  8. Driver Leaderboard     : a composite 0-100 performance score per
                              driver (safety, revenue efficiency, harsh-
                              event rate, completion rate) with medals.
  9. Reports Center         : one-click multi-sheet Excel workbook
                              (fleet/driver/car summaries + alerts).
 10. User Management        : activate/deactivate accounts, change
                              roles, admin password reset, delete
                              accounts, create accounts directly.
 11. Audit Log              : read-only trail of every sign-in, sign-up,
                              and admin account action.
"""

import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import db

ACCENT = "#0071e3"
ACCENT_RED = "#FF3B30"
INK = "#1d1d1f"
GREEN = "#34c759"
RED = "#ff3b30"

STATUS_COLORS = {
    "Running": ("#e6f7ec", "#1e7e34"),
    "Charging": ("#e8f2ff", "#0071e3"),
    "Idle": ("#f2f2f4", "#6e6e73"),
    "Under Maintenance": ("#fdeceb", "#d70015"),
}


def _chart_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=INK, family="Inter", weight=700)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", color=INK),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


def _maintenance_total(df_subset: pd.DataFrame) -> float:
    """
    Exact_Maintenance_Cost_INR is a FIXED, one-time value per car — it's
    identical across every one of that car's records (it's a snapshot
    total, not a cost incurred per trip). Summing it across every row
    (like Recognized_Revenue, which genuinely IS earned per-trip) massively
    over-counts it: a car with 200 records would have its maintenance cost
    counted 200 times. This takes ONE value per unique car instead.
    """
    return df_subset.groupby("Car_ID")["Exact_Maintenance_Cost_INR"].first().sum()


def _maintenance_total_by_car(df_subset: pd.DataFrame) -> pd.Series:
    """Same fixed-cost-per-car logic as _maintenance_total, but returns the
    per-car breakdown (a Series) instead of a single grand total — used for
    car-wise leaderboard charts."""
    return df_subset.groupby("Car_ID")["Exact_Maintenance_Cost_INR"].first()


def _order_topn_controls(key_prefix: str, default_top_n: int = 10, max_n: int = 50):
    """
    Renders the Highest/Lowest + Top N controls used above every ranked bar
    chart in the dashboard. Returns (order, top_n) where order is
    "Highest" or "Lowest".
    """
    c1, c2 = st.columns([1, 2])
    with c1:
        order = st.radio("Order", ["Highest", "Lowest"], key=f"{key_prefix}_order", horizontal=True)
    with c2:
        top_n = st.number_input("Top N", min_value=3, max_value=max_n, value=default_top_n, step=1, key=f"{key_prefix}_topn")
    return order, top_n


def _rank_series(series: pd.Series, order: str, top_n: int) -> pd.Series:
    """Applies Highest/Lowest ordering and Top N limit to any Series
    (e.g. a groupby().sum() result) before it goes into a bar chart."""
    ascending = (order == "Lowest")
    return series.sort_values(ascending=ascending).head(top_n)


def _show_key_fleet_metrics(df: pd.DataFrame):
    """
    The 4 must-have views for presentations/reviews, each ready to show
    with one click — no live filter configuration needed:
      1. Vehicle status
      2. Driver behavior, ranked by violations (highest first)
      3. Revenue by driver
      4. Charging + maintenance expense, car-wise and brand-wise
    """
    st.subheader("1. Vehicle Status")
    latest_per_car = df.sort_values("Date").groupby("Car_ID").last()
    status_counts = latest_per_car["Fleet_Status"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cars", latest_per_car.shape[0])
    c2.metric("Running", int(status_counts.get("Running", 0)))
    c3.metric("Charging", int(status_counts.get("Charging", 0)))
    c4.metric("Idle / Maintenance",
              int(status_counts.get("Idle", 0) + status_counts.get("Under Maintenance", 0)))

    st.divider()

    st.subheader("2. Driver Behavior — Violations")
    st.caption("A 'violation' is any record where the car reached 110+ km/h.")
    order2, topn2 = _order_topn_controls("kfm_viol", default_top_n=15)
    violations_full = df[df["Driving_Style"] == "Rash"].groupby("Driver_Name").size()
    violations = _rank_series(violations_full, order2, topn2)
    if not violations.empty:
        fig_viol = px.bar(
            violations, x=violations.values, y=violations.index, orientation="h",
            labels={"x": "Violation Count (110+ km/h)", "y": "Driver"},
            color_discrete_sequence=[ACCENT_RED],
        )
        fig_viol = _chart_layout(fig_viol, f"{order2} {topn2} Drivers by Violations")
        fig_viol.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_viol, use_container_width=True)
        st.caption(f"{order2} violations: **{violations.index[0]}** — {violations.iloc[0]} times")
    else:
        st.info("No violations recorded.")

    st.divider()

    st.subheader("3. Revenue by Driver")
    order3, topn3 = _order_topn_controls("kfm_rev", default_top_n=15)
    revenue_by_driver_full = df.groupby("Driver_Name")["Recognized_Revenue"].sum()
    revenue_by_driver = _rank_series(revenue_by_driver_full, order3, topn3)
    fig_rev = px.bar(
        revenue_by_driver, x=revenue_by_driver.values, y=revenue_by_driver.index, orientation="h",
        labels={"x": "Recognized Revenue (₹)", "y": "Driver"},
        color_discrete_sequence=[ACCENT],
    )
    fig_rev = _chart_layout(fig_rev, f"{order3} {topn3} Drivers by Recognized Revenue")
    fig_rev.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_rev, use_container_width=True)
    st.caption(f"{order3}: **{revenue_by_driver.index[0]}** — ₹{revenue_by_driver.iloc[0]:,.0f}")

    st.divider()

    st.subheader("4. Charging & Maintenance Expense — Car-wise")
    order4, topn4 = _order_topn_controls("kfm_carwise", default_top_n=15)
    cw1, cw2 = st.columns(2)
    with cw1:
        charge_by_car_full = df.groupby("Car_ID")["Charging_Cost_INR"].sum()
        charge_by_car = _rank_series(charge_by_car_full, order4, topn4)
        fig_charge_car = px.bar(
            charge_by_car, x=charge_by_car.values, y=charge_by_car.index, orientation="h",
            labels={"x": "Charging Cost (₹)", "y": "Car"}, color_discrete_sequence=["#5AC8FA"],
        )
        fig_charge_car = _chart_layout(fig_charge_car, f"{order4} {topn4} Cars by Charging Cost")
        fig_charge_car.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_charge_car, use_container_width=True)
    with cw2:
        maint_by_car_full = _maintenance_total_by_car(df)
        maint_by_car = _rank_series(maint_by_car_full, order4, topn4)
        fig_maint_car = px.bar(
            maint_by_car, x=maint_by_car.values, y=maint_by_car.index, orientation="h",
            labels={"x": "Maintenance Cost (₹)", "y": "Car"}, color_discrete_sequence=[ACCENT_RED],
        )
        fig_maint_car = _chart_layout(fig_maint_car, f"{order4} {topn4} Cars by Maintenance Cost")
        fig_maint_car.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_maint_car, use_container_width=True)

    st.subheader("4. Charging & Maintenance Expense — Brand-wise")
    st.caption("Maintenance is a fixed one-time cost per car — summed once per car within each brand, "
               "not once per driving record (see Section 12.5 of the project document for why that matters).")
    order5, topn5 = _order_topn_controls("kfm_brandwise", default_top_n=10, max_n=10)
    bw1, bw2 = st.columns(2)
    with bw1:
        charge_by_brand_full = df.groupby("Company")["Charging_Cost_INR"].sum()
        charge_by_brand = _rank_series(charge_by_brand_full, order5, topn5)
        fig_charge_brand = px.bar(
            charge_by_brand, x=charge_by_brand.index, y=charge_by_brand.values,
            labels={"x": "Brand", "y": "Charging Cost (₹)"}, color_discrete_sequence=["#5AC8FA"],
        )
        fig_charge_brand = _chart_layout(fig_charge_brand, f"{order5} Brands by Charging Cost")
        st.plotly_chart(fig_charge_brand, use_container_width=True)
    with bw2:
        maint_by_brand_full = (
            df.groupby(["Company", "Car_ID"])["Exact_Maintenance_Cost_INR"].first()
            .groupby("Company").sum()
        )
        maint_by_brand = _rank_series(maint_by_brand_full, order5, topn5)
        fig_maint_brand = px.bar(
            maint_by_brand, x=maint_by_brand.index, y=maint_by_brand.values,
            labels={"x": "Brand", "y": "Maintenance Cost (₹)"}, color_discrete_sequence=[ACCENT_RED],
        )
        fig_maint_brand = _chart_layout(fig_maint_brand, f"{order5} Brands by Maintenance Cost")
        st.plotly_chart(fig_maint_brand, use_container_width=True)


def show_admin_dashboard(df: pd.DataFrame):
    st.title("Fleet Overview")
    st.caption("Live status and performance across the full EV fleet")

    (tab_fleet, tab_key, tab_driver, tab_car, tab_filters,
     tab_alerts, tab_predictive, tab_leaderboard, tab_reports,
     tab_users, tab_audit) = st.tabs(
        ["Fleet Overview", "Key Fleet Metrics", "Driver Lookup", "Car Lookup",
         "Advanced Filters", "Alerts & Health", "Predictive Maintenance",
         "Driver Leaderboard", "Reports Center", "User Management", "Audit Log"]
    )

    with tab_fleet:
        _show_fleet_overview(df)

    with tab_key:
        _show_key_fleet_metrics(df)

    with tab_driver:
        _show_driver_lookup(df)

    with tab_car:
        _show_car_lookup(df)

    with tab_filters:
        _show_advanced_filters(df)

    with tab_alerts:
        _show_alerts_health(df)

    with tab_predictive:
        _show_predictive_maintenance(df)

    with tab_leaderboard:
        _show_driver_leaderboard(df)

    with tab_reports:
        _show_reports_center(df)

    with tab_users:
        _show_user_management(df)

    with tab_audit:
        _show_audit_log()


# ============================================================
# TAB 1 — Fleet Overview (unchanged from before)
# ============================================================
def _show_fleet_overview(df: pd.DataFrame):
    latest_per_car = df.sort_values("Date").groupby("Car_ID").last()
    status_counts = latest_per_car["Fleet_Status"].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cars", latest_per_car.shape[0])
    col2.metric("Running", int(status_counts.get("Running", 0)))
    col3.metric("Charging", int(status_counts.get("Charging", 0)))
    col4.metric("Idle / Maintenance",
                int(status_counts.get("Idle", 0) + status_counts.get("Under Maintenance", 0)))

    st.divider()

    months_ordered = (
        df[["Month", "Month_Num"]].drop_duplicates().sort_values("Month_Num")["Month"].tolist()
    )
    selected_month = st.selectbox("Select Month", months_ordered, index=len(months_ordered) - 1)
    month_df = df[df["Month"] == selected_month]

    st.subheader(f"{selected_month} Performance")

    trip_col1, trip_col2, trip_col3 = st.columns(3)
    completed = int((month_df["Trip_Status"] == "Completed").sum())
    in_transit = int((month_df["Trip_Status"] == "In Transit").sum())
    total_recognized = month_df["Recognized_Revenue"].sum()
    trip_col1.metric("Completed Trips", completed)
    trip_col2.metric("In Transit (unbilled)", in_transit)
    trip_col3.metric("Recognized Revenue", f"₹{total_recognized:,.0f}")
    st.caption("Recognized Revenue only counts trips that hit 100% completion — "
               "in-progress trips contribute ₹0 until then.")

    st.divider()

    order1, topn1 = _order_topn_controls("fleet_overview", default_top_n=10)
    left, right = st.columns(2)

    with left:
        revenue_by_car_full = month_df.groupby("Car_ID")["Recognized_Revenue"].sum()
        revenue_by_car = _rank_series(revenue_by_car_full, order1, topn1)
        fig_rev = px.bar(
            revenue_by_car, x=revenue_by_car.values, y=revenue_by_car.index,
            orientation="h", labels={"x": "Recognized Revenue (INR)", "y": "Car"},
            color_discrete_sequence=[ACCENT],
        )
        fig_rev = _chart_layout(fig_rev, f"{order1} {topn1} Cars by Recognized Revenue")
        fig_rev.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_rev, use_container_width=True)
        if not revenue_by_car.empty:
            st.caption(f"{order1}: **{revenue_by_car.index[0]}** — ₹{revenue_by_car.iloc[0]:,.0f}")

    with right:
        maint_by_car_full = month_df.groupby("Car_ID")["Exact_Maintenance_Cost_INR"].first()
        maint_by_car = _rank_series(maint_by_car_full, order1, topn1)
        fig_maint = px.bar(
            maint_by_car, x=maint_by_car.values, y=maint_by_car.index,
            orientation="h", labels={"x": "Maintenance Cost (INR)", "y": "Car"},
            color_discrete_sequence=[RED],
        )
        fig_maint = _chart_layout(fig_maint, f"{order1} {topn1} Cars by Maintenance Cost")
        fig_maint.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_maint, use_container_width=True)
        if not maint_by_car.empty:
            st.caption(f"{order1}: **{maint_by_car.index[0]}** — ₹{maint_by_car.iloc[0]:,.0f}")

    st.divider()

    st.subheader("Driver Habits")
    st.caption("A driver is marked Rash on any record where speed reached 110 km/h or above.")
    style_counts = month_df["Driving_Style"].value_counts()

    hab_col1, hab_col2 = st.columns([1, 2])
    with hab_col1:
        st.metric("Safe Records", int(style_counts.get("Safe", 0)))
        st.metric("Rash Records", int(style_counts.get("Rash", 0)))

    with hab_col2:
        fig_style = px.pie(
            names=style_counts.index, values=style_counts.values, color=style_counts.index,
            color_discrete_map={"Safe": ACCENT, "Rash": RED}, hole=0.55,
        )
        fig_style = _chart_layout(fig_style, "Safe vs Rash Driving")
        st.plotly_chart(fig_style, use_container_width=True)

    st.divider()
    with st.expander("Raw data for this month"):
        st.dataframe(month_df, use_container_width=True)


# ============================================================
# TAB 2 — Driver Lookup
# ============================================================
def _driver_lookup_styles():
    st.markdown("""
        <style>
        .driver-lookup-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 28px 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }
        .dl-name { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; color: #1d1d1f; }
        .dl-sub { font-size: 14px; color: #86868b; margin-bottom: 14px; }
        .dl-status-pill { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; }
        .dl-style-pill { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; margin-left: 8px; }
        .dl-location-label { font-size: 12px; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 18px; }
        .dl-location-line { font-size: 15px; color: #1d1d1f; }
        </style>
    """, unsafe_allow_html=True)


def _show_driver_lookup(df: pd.DataFrame):
    _driver_lookup_styles()

    st.subheader("Look Up a Driver")
    st.caption("Everything tied to one driver — current status, location, "
               "driving history, and revenue — in one place.")

    driver_names = sorted(df["Driver_Name"].dropna().unique())
    if not driver_names:
        st.info("No driver data available.")
        return

    selected_driver = st.selectbox("Select Driver", driver_names, key="dl_driver_select")
    driver_df = df[df["Driver_Name"] == selected_driver].sort_values("Date")

    if driver_df.empty:
        st.info("No records found for this driver.")
        return

    latest = driver_df.sort_values("Date").iloc[-1]

    status = latest["Fleet_Status"]
    bg_color, text_color = STATUS_COLORS.get(status, ("#f2f2f4", "#6e6e73"))
    style_bg, style_color = ("#fdeceb", "#d70015") if latest["Driving_Style"] == "Rash" else ("#e6f7ec", "#1e7e34")

    cars_driven = driver_df["Car_ID"].unique()
    car_list_str = ", ".join(sorted(cars_driven))

    hero_html = (
        '<div class="driver-lookup-card">'
        f'<div class="dl-name">{selected_driver}</div>'
        f'<div class="dl-sub">Currently on {latest["Car_ID"]} &middot; '
        f'{latest["Company"]} {latest["Model"]}</div>'
        f'<span class="dl-status-pill" style="background:{bg_color}; color:{text_color};">{status}</span>'
        f'<span class="dl-style-pill" style="background:{style_bg}; color:{style_color};">{latest["Driving_Style"]}</span>'
        '<div class="dl-location-label">Current Location</div>'
        f'<div class="dl-location-line">{latest["Location"]}, {latest["City"]}</div>'
        '<div class="dl-location-label">Cars Driven (all-time)</div>'
        f'<div class="dl-location-line">{car_list_str}</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Battery Remaining", f"{latest['Battery_Remaining_Percent']:.0f}%")
    col2.metric("Last Recorded Speed", f"{latest['Speed_kmph']:.0f} km/h")
    col3.metric("Driving Mode", latest["Driving_Mode"])
    col4.metric("Total Records", int(driver_df.shape[0]))

    st.divider()

    st.subheader("Driving Behavior History")
    style_counts = driver_df["Driving_Style"].value_counts()
    rash_count = int(style_counts.get("Rash", 0))
    safe_count = int(style_counts.get("Safe", 0))
    rash_pct = (rash_count / driver_df.shape[0] * 100) if driver_df.shape[0] else 0

    beh_col1, beh_col2 = st.columns([1, 2])
    with beh_col1:
        st.metric("Safe Records", safe_count)
        st.metric("Rash Records", rash_count)
        st.metric("Rash Rate", f"{rash_pct:.1f}%")

    with beh_col2:
        fig_style = px.pie(
            names=style_counts.index, values=style_counts.values, color=style_counts.index,
            color_discrete_map={"Safe": ACCENT, "Rash": RED}, hole=0.55,
        )
        fig_style = _chart_layout(fig_style, f"{selected_driver} — Safe vs Rash")
        st.plotly_chart(fig_style, use_container_width=True)

    st.divider()

    st.subheader("Trip & Revenue Summary")
    completed = int((driver_df["Trip_Status"] == "Completed").sum())
    in_transit = int((driver_df["Trip_Status"] == "In Transit").sum())
    total_revenue = driver_df["Recognized_Revenue"].sum()
    total_maint = _maintenance_total(driver_df)

    rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
    rev_col1.metric("Completed Trips", completed)
    rev_col2.metric("In Transit", in_transit)
    rev_col3.metric("Total Revenue Generated", f"₹{total_revenue:,.0f}")
    rev_col4.metric("Total Maintenance Cost", f"₹{total_maint:,.0f}")

    st.divider()

    st.subheader("Trends Over Time")
    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        fig_speed = px.line(
            driver_df, x="Date", y="Speed_kmph", markers=True,
            color_discrete_sequence=[ACCENT],
        )
        fig_speed.add_hline(y=110, line_dash="dash", line_color=RED,
                             annotation_text="Rash threshold (110 km/h)")
        fig_speed = _chart_layout(fig_speed, "Speed Over Time")
        st.plotly_chart(fig_speed, use_container_width=True)

    with trend_col2:
        fig_batt = px.line(
            driver_df, x="Date", y="Battery_Remaining_Percent", markers=True,
            color_discrete_sequence=[GREEN],
        )
        fig_batt = _chart_layout(fig_batt, "Battery Level Over Time")
        st.plotly_chart(fig_batt, use_container_width=True)

    st.divider()

    st.subheader("Full Record Log")
    display_cols = [
        "Date", "Car_ID", "Company", "Model", "Location", "City",
        "Fleet_Status", "Speed_kmph", "Driving_Mode", "Driving_Style",
        "Distance_km", "Battery_Remaining_Percent", "Harsh_Events",
        "Trip_Status", "Recognized_Revenue", "Total_Odometer_km",
    ]
    display_cols = [c for c in display_cols if c in driver_df.columns]
    st.dataframe(
        driver_df[display_cols].sort_values("Date", ascending=False),
        use_container_width=True,
    )


# ============================================================
# TAB 3 — Car Lookup (NEW — mirrors the Driver Dashboard hero card)
# ============================================================
def _car_lookup_styles():
    st.markdown("""
        <style>
        .car-lookup-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 28px 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }
        .cl-name { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; color: #1d1d1f; }
        .cl-sub { font-size: 14px; color: #86868b; margin-bottom: 14px; }
        .cl-status-pill { display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: 13px; font-weight: 600; }
        .cl-battery-track { width: 100%; height: 14px; background: #f2f2f4; border-radius: 999px; overflow: hidden; margin-top: 8px; }
        .cl-battery-fill { height: 100%; border-radius: 999px; }
        .cl-location-line { font-size: 15px; color: #1d1d1f; margin-top: 16px; }
        .cl-location-label { font-size: 12px; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em; }
        </style>
    """, unsafe_allow_html=True)


def _battery_color(pct):
    if pct >= 50:
        return GREEN
    elif pct >= 20:
        return "#ff9f0a"
    return RED


def _show_car_lookup(df: pd.DataFrame):
    _car_lookup_styles()

    st.subheader("Look Up a Car")
    st.caption("The same rich single-car view drivers see on their own dashboard — "
               "available here for any car in the fleet.")

    car_ids = sorted(df["Car_ID"].dropna().unique())
    if not car_ids:
        st.info("No car data available.")
        return

    car_id = st.selectbox("Select Car", car_ids, key="cl_car_select")
    car_data = df[df["Car_ID"] == car_id].sort_values("Date", ascending=False).iloc[0]
    car_history = df[df["Car_ID"] == car_id].sort_values("Date")

    rash_event_count = int((car_history["Driving_Style"] == "Rash").sum())

    status = car_data["Fleet_Status"]
    bg_color, text_color = STATUS_COLORS.get(status, ("#f2f2f4", "#6e6e73"))
    battery_pct = float(car_data["Battery_Remaining_Percent"])
    battery_color = _battery_color(battery_pct)

    hero_html = (
        '<div class="car-lookup-card">'
        f'<div class="cl-name">{car_data["Company"]} {car_data["Model"]}</div>'
        f'<div class="cl-sub">{car_id} &middot; Driver: {car_data["Driver_Name"]} &middot; '
        f'{int(car_data["Seating_Capacity"])} seats</div>'
        f'<span class="cl-status-pill" style="background:{bg_color}; color:{text_color};">{status}</span>'
        '<div class="cl-location-label" style="margin-top:20px;">Battery Remaining</div>'
        f'<div style="font-size:22px; font-weight:700; color:#1d1d1f;">{battery_pct:.1f}%</div>'
        '<div class="cl-battery-track">'
        f'<div class="cl-battery-fill" style="width:{battery_pct}%; background:{battery_color};"></div>'
        '</div>'
        '<div class="cl-location-label" style="margin-top:20px;">Current Location</div>'
        f'<div class="cl-location-line">{car_data["Location"]}, {car_data["City"]}</div>'
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

    col10, col11, col12, col13 = st.columns(4)
    col10.metric("Times Reached 110+ km/h (all-time)", f"{rash_event_count}")
    col11.metric("Total Revenue Earned", f"₹{car_history['Recognized_Revenue'].sum():,.0f}")
    col12.metric("Total Charging Cost", f"₹{car_history['Charging_Cost_INR'].sum():,.0f}")
    col13.metric("Total Maintenance Cost", f"₹{_maintenance_total(car_history):,.0f}")

    st.caption(f"'Driving Style (now)' reflects only the latest reading "
               f"({car_data['Speed_kmph']:.0f} km/h) — a car can have past rash "
               f"events but be driving safely right now, or vice versa.")

    st.divider()

    st.subheader("Current Trip")
    trip_col1, trip_col2, trip_col3 = st.columns(3)
    trip_col1.metric("Trip Status", car_data["Trip_Status"])
    trip_col2.metric("Minutes Elapsed", f"{int(car_data['Minutes_Elapsed'])} / {int(car_data['Trip_Duration_Minutes'])} min")
    trip_col3.metric("Recognized Revenue", f"₹{car_data['Recognized_Revenue']:,.0f}")
    if car_data["Trip_Status"] == "In Transit":
        st.caption("Revenue is withheld until the trip reaches 100% completion.")

    st.divider()

    st.subheader("Trends Over Time")
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        fig_speed = px.line(car_history, x="Date", y="Speed_kmph", markers=True,
                             color_discrete_sequence=[ACCENT])
        fig_speed.add_hline(y=110, line_dash="dash", line_color=RED,
                             annotation_text="Rash threshold (110 km/h)")
        fig_speed = _chart_layout(fig_speed, "Speed Over Time")
        st.plotly_chart(fig_speed, use_container_width=True)
    with trend_col2:
        fig_batt = px.line(car_history, x="Date", y="Battery_Remaining_Percent", markers=True,
                            color_discrete_sequence=[GREEN])
        fig_batt = _chart_layout(fig_batt, "Battery Level Over Time")
        st.plotly_chart(fig_batt, use_container_width=True)

    st.divider()
    with st.expander("Full record log for this car"):
        display_cols = [
            "Date", "Driver_Name", "Location", "City", "Fleet_Status",
            "Speed_kmph", "Driving_Mode", "Driving_Style", "Distance_km",
            "Battery_Remaining_Percent", "Harsh_Events", "Trip_Status",
            "Recognized_Revenue", "Total_Odometer_km",
        ]
        display_cols = [c for c in display_cols if c in car_history.columns]
        st.dataframe(
            car_history[display_cols].sort_values("Date", ascending=False),
            use_container_width=True,
        )


# ============================================================
# TAB 4 — Advanced Filters (NEW)
# ============================================================
METRIC_LABELS = {
    "Recognized_Revenue": "Total Recognized Revenue (₹)",
    "Exact_Maintenance_Cost_INR": "Total Maintenance Cost (₹)",
    "Charging_Cost_INR": "Total Charging Cost (₹)",
    "Net_Profit": "Net Profit / Loss (₹)",
    "Battery_Remaining_Percent": "Avg Battery Remaining (%)",
    "Speed_kmph": "Avg Speed (km/h)",
    "Harsh_Events": "Total Harsh Events",
    "Distance_km": "Total Distance Driven (km)",
    "Total_Odometer_km": "Total Odometer (km)",
    "Car_Max_Range_km": "Avg Car Max Range Spec (km)",
    "Actual_Remaining_Range_km": "Avg Predicted Remaining Range (km)",
}


def _aggregate_by_group(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    if group_by == "Company":
        # Maintenance is a fixed cost per CAR, not per row — even when
        # grouping by brand, it must be summed once per unique car within
        # that brand, never once per driving record.
        maint = df.groupby(["Company", "Car_ID"])["Exact_Maintenance_Cost_INR"].first().groupby("Company").sum()
    else:
        maint = df.groupby(group_by)["Exact_Maintenance_Cost_INR"].first()

    agg = df.groupby(group_by).agg(
        Recognized_Revenue=("Recognized_Revenue", "sum"),
        Charging_Cost_INR=("Charging_Cost_INR", "sum"),
        Battery_Remaining_Percent=("Battery_Remaining_Percent", "mean"),
        Speed_kmph=("Speed_kmph", "mean"),
        Harsh_Events=("Harsh_Events", "sum"),
        Distance_km=("Distance_km", "sum"),
        Total_Odometer_km=("Total_Odometer_km", "max"),
        Car_Max_Range_km=("Car_Max_Range_km", "mean"),
        Actual_Remaining_Range_km=("Actual_Remaining_Range_km", "mean"),
        Records=("Date", "count"),
    ).reset_index()
    agg["Exact_Maintenance_Cost_INR"] = agg[group_by].map(maint)
    # Net Profit = Revenue - Maintenance - Charging. Charging cost was
    # previously left out of this calculation entirely — since it's a real
    # operating expense (and actually a bigger one than maintenance across
    # the fleet), leaving it out understated true cost significantly.
    agg["Net_Profit"] = agg["Recognized_Revenue"] - agg["Exact_Maintenance_Cost_INR"] - agg["Charging_Cost_INR"]
    return agg


def _show_advanced_filters(df: pd.DataFrame):
    st.subheader("Advanced Filters")
    st.caption("Slice the fleet by any driver, car, or number below — then rank "
               "everything by revenue, cost, range, battery, or net profit/loss.")

    # Bulletproof reset pattern: instead of trying to clear each widget's
    # session_state value (fragile — some Streamlit versions don't reliably
    # reset sliders/multiselects that have computed default values this
    # way), we change every widget's KEY on reset. A new key means
    # Streamlit treats it as a brand new widget with no memory of the old
    # selection at all — guaranteed fresh defaults, no timing edge cases.
    if "af_reset_counter" not in st.session_state:
        st.session_state["af_reset_counter"] = 0

    def _fk(base: str) -> str:
        """filter key, salted with the current reset counter"""
        return f"{base}_{st.session_state['af_reset_counter']}"

    reset_col, _ = st.columns([1, 4])
    with reset_col:
        if st.button("Reset All Filters"):
            st.session_state["af_reset_counter"] += 1
            st.rerun()

    # ---------- Filter controls ----------
    with st.expander("Filter Records", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_drivers = st.multiselect("Driver", sorted(df["Driver_Name"].dropna().unique()), key=_fk("af_driver"))
            sel_cars = st.multiselect("Car ID", sorted(df["Car_ID"].dropna().unique()), key=_fk("af_car"))
            sel_models = st.multiselect("Model", sorted(df["Model"].dropna().unique()), key=_fk("af_model"))
        with c2:
            sel_cities = st.multiselect("City", sorted(df["City"].dropna().unique()), key=_fk("af_city"))
            sel_status = st.multiselect("Fleet Status", sorted(df["Fleet_Status"].dropna().unique()), key=_fk("af_status"))
            sel_trip_status = st.multiselect("Trip Status", sorted(df["Trip_Status"].dropna().unique()), key=_fk("af_trip_status"))
        with c3:
            sel_style = st.multiselect("Driving Style", sorted(df["Driving_Style"].dropna().unique()), key=_fk("af_style"))
            months_ordered = df[["Month", "Month_Num"]].drop_duplicates().sort_values("Month_Num")["Month"].tolist()
            sel_months = st.multiselect("Month", months_ordered, key=_fk("af_month"))
            min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
            date_range = st.date_input("Date Range", (min_date, max_date), key=_fk("af_date"))

        st.markdown("**Numeric Ranges**")
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            battery_range = st.slider("Battery (%)", 0, 100, (0, 100), key=_fk("af_battery"))
            speed_min, speed_max = int(df["Speed_kmph"].min()), int(df["Speed_kmph"].max()) + 1
            speed_range = st.slider("Speed (km/h)", speed_min, speed_max, (speed_min, speed_max), key=_fk("af_speed"))
        with n2:
            rev_min, rev_max = float(df["Recognized_Revenue"].min()), float(df["Recognized_Revenue"].max()) + 1
            revenue_range = st.slider("Recognized Revenue (₹)", rev_min, rev_max, (rev_min, rev_max), key=_fk("af_revenue"))
            maint_min, maint_max = float(df["Exact_Maintenance_Cost_INR"].min()), float(df["Exact_Maintenance_Cost_INR"].max()) + 1
            maint_range = st.slider("Maintenance Cost (₹)", maint_min, maint_max, (maint_min, maint_max), key=_fk("af_maint"))
            charge_min, charge_max = float(df["Charging_Cost_INR"].min()), float(df["Charging_Cost_INR"].max()) + 1
            charge_range = st.slider("Charging Cost (₹)", charge_min, charge_max, (charge_min, charge_max), key=_fk("af_charge"))
        with n3:
            range_min, range_max = float(df["Car_Max_Range_km"].min()), float(df["Car_Max_Range_km"].max()) + 1
            range_spec_range = st.slider("Car Max Range Spec (km)", range_min, range_max, (range_min, range_max), key=_fk("af_range_spec"))
            odo_min, odo_max = float(df["Total_Odometer_km"].min()), float(df["Total_Odometer_km"].max()) + 1
            odometer_range = st.slider("Total Odometer (km)", odo_min, odo_max, (odo_min, odo_max), key=_fk("af_odometer"))
        with n4:
            dist_min, dist_max = float(df["Distance_km"].min()), float(df["Distance_km"].max()) + 1
            distance_range = st.slider("Trip Distance (km)", dist_min, dist_max, (dist_min, dist_max), key=_fk("af_distance"))
            harsh_min, harsh_max = int(df["Harsh_Events"].min()), int(df["Harsh_Events"].max()) + 1
            harsh_range = st.slider("Harsh Events", harsh_min, harsh_max, (harsh_min, harsh_max), key=_fk("af_harsh"))

    # ---------- Apply filters ----------
    filtered = df.copy()
    if sel_drivers:
        filtered = filtered[filtered["Driver_Name"].isin(sel_drivers)]
    if sel_cars:
        filtered = filtered[filtered["Car_ID"].isin(sel_cars)]
    if sel_models:
        filtered = filtered[filtered["Model"].isin(sel_models)]
    if sel_cities:
        filtered = filtered[filtered["City"].isin(sel_cities)]
    if sel_status:
        filtered = filtered[filtered["Fleet_Status"].isin(sel_status)]
    if sel_trip_status:
        filtered = filtered[filtered["Trip_Status"].isin(sel_trip_status)]
    if sel_style:
        filtered = filtered[filtered["Driving_Style"].isin(sel_style)]
    if sel_months:
        filtered = filtered[filtered["Month"].isin(sel_months)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered = filtered[(filtered["Date"].dt.date >= start_d) & (filtered["Date"].dt.date <= end_d)]

    filtered = filtered[filtered["Battery_Remaining_Percent"].between(*battery_range)]
    filtered = filtered[filtered["Speed_kmph"].between(*speed_range)]
    filtered = filtered[filtered["Recognized_Revenue"].between(*revenue_range)]
    filtered = filtered[filtered["Exact_Maintenance_Cost_INR"].between(*maint_range)]
    filtered = filtered[filtered["Charging_Cost_INR"].between(*charge_range)]
    filtered = filtered[filtered["Car_Max_Range_km"].between(*range_spec_range)]
    filtered = filtered[filtered["Total_Odometer_km"].between(*odometer_range)]
    filtered = filtered[filtered["Distance_km"].between(*distance_range)]
    filtered = filtered[filtered["Harsh_Events"].between(*harsh_range)]

    st.divider()

    if filtered.empty:
        st.warning("No records match the selected filters. Try widening a range or clearing a filter.")
        return

    # ---------- Summary metrics on the filtered set ----------
    st.subheader("Filtered Summary")
    total_revenue = filtered["Recognized_Revenue"].sum()
    total_maint = _maintenance_total(filtered)
    total_charging = filtered["Charging_Cost_INR"].sum()
    net_profit = total_revenue - total_maint - total_charging
    rash_pct = (filtered["Driving_Style"] == "Rash").mean() * 100

    s1, s2, s3 = st.columns(3)
    s1.metric("Records Matched", f"{filtered.shape[0]:,}")
    s2.metric("Cars Involved", filtered["Car_ID"].nunique())
    s3.metric("Total Revenue", f"₹{total_revenue:,.0f}")

    s4, s5, s6, s7 = st.columns(4)
    s4.metric("Total Maintenance", f"₹{total_maint:,.0f}")
    s5.metric("Total Charging Cost", f"₹{total_charging:,.0f}")
    profit_delta = f"{'+' if net_profit >= 0 else '-'}₹{abs(net_profit):,.0f} ({'Profit' if net_profit >= 0 else 'Loss'})"
    s6.metric("Net Profit / Loss", f"₹{net_profit:,.0f}", delta=profit_delta)
    s7.metric("Rash Record Rate", f"{rash_pct:.1f}%")
    st.caption("Net Profit / Loss = Revenue − Maintenance − Charging Cost.")

    st.divider()

    # ---------- Rank & Compare ----------
    st.subheader("Rank & Compare")
    st.caption("Find the highest, lowest, or top-N cars/drivers by any metric — including net profit/loss.")

    r1, r2, r3, r4 = st.columns([2, 2, 1, 1])
    with r1:
        group_by = st.selectbox("Group By", ["Car_ID", "Driver_Name", "Company"], key="af_group_by")
    with r2:
        metric = st.selectbox("Metric", list(METRIC_LABELS.keys()),
                               format_func=lambda k: METRIC_LABELS[k], key="af_metric")
    with r3:
        order = st.radio("Order", ["Highest", "Lowest"], key="af_order", horizontal=True)
    with r4:
        top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1, key="af_top_n")

    agg = _aggregate_by_group(filtered, group_by)
    ascending = (order == "Lowest")
    ranked = agg.sort_values(metric, ascending=ascending).head(int(top_n))

    bar_colors = None
    if metric == "Net_Profit":
        bar_colors = np.where(ranked[metric] >= 0, GREEN, RED)

    fig_rank = px.bar(
        ranked, x=metric, y=group_by, orientation="h",
        labels={metric: METRIC_LABELS[metric], group_by: group_by.replace("_", " ")},
    )
    if bar_colors is not None:
        fig_rank.update_traces(marker_color=bar_colors)
    else:
        fig_rank.update_traces(marker_color=ACCENT)
    fig_rank = _chart_layout(fig_rank, f"{order} {top_n} by {METRIC_LABELS[metric]}")
    fig_rank.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_rank, use_container_width=True)

    st.dataframe(ranked, use_container_width=True)

    st.divider()

    # ---------- Distribution visualizations ----------
    st.subheader("Distributions")
    d1, d2 = st.columns(2)
    with d1:
        fig_batt_hist = px.histogram(filtered, x="Battery_Remaining_Percent", nbins=20,
                                      color_discrete_sequence=[ACCENT])
        fig_batt_hist = _chart_layout(fig_batt_hist, "Battery Level Distribution")
        st.plotly_chart(fig_batt_hist, use_container_width=True)
    with d2:
        fig_speed_hist = px.histogram(filtered, x="Speed_kmph", nbins=20,
                                       color_discrete_sequence=[ACCENT])
        fig_speed_hist = _chart_layout(fig_speed_hist, "Speed Distribution")
        st.plotly_chart(fig_speed_hist, use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        status_counts = filtered["Fleet_Status"].value_counts()
        fig_status = px.bar(status_counts, x=status_counts.index, y=status_counts.values,
                             labels={"x": "Fleet Status", "y": "Records"},
                             color_discrete_sequence=[ACCENT])
        fig_status = _chart_layout(fig_status, "Fleet Status Breakdown")
        st.plotly_chart(fig_status, use_container_width=True)
    with d4:
        style_counts = filtered["Driving_Style"].value_counts()
        fig_style = px.pie(names=style_counts.index, values=style_counts.values, color=style_counts.index,
                            color_discrete_map={"Safe": ACCENT, "Rash": RED}, hole=0.55)
        fig_style = _chart_layout(fig_style, "Safe vs Rash (filtered)")
        st.plotly_chart(fig_style, use_container_width=True)

    st.divider()

    # ---------- Export ----------
    st.subheader("Filtered Data")
    st.caption(f"Showing {filtered.shape[0]:,} of {df.shape[0]:,} total records.")
    st.dataframe(filtered.sort_values("Date", ascending=False), use_container_width=True)
    st.download_button(
        "Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ev_fleet_data.csv",
        mime="text/csv",
    )

# ============================================================
# TAB 6 — Alerts & Fleet Health (NEW — pro upgrade)
# ============================================================
def _health_score_components(df: pd.DataFrame):
    """Builds a single 0-100 fleet health score from four weighted
    signals, plus returns the raw numbers so they can be shown alongside
    the gauge. Weights: battery health, safe-driving rate, uptime
    (cars not stuck in maintenance), and profitability."""
    latest_per_car = df.sort_values("Date").groupby("Car_ID").last()

    avg_battery = float(latest_per_car["Battery_Remaining_Percent"].mean())

    total_records = len(df)
    rash_records = int((df["Driving_Style"] == "Rash").sum())
    safe_rate = 100.0 * (1 - (rash_records / total_records)) if total_records else 100.0

    total_cars = latest_per_car.shape[0]
    in_maintenance = int((latest_per_car["Fleet_Status"] == "Under Maintenance").sum())
    uptime_rate = 100.0 * (1 - (in_maintenance / total_cars)) if total_cars else 100.0

    total_revenue = df["Recognized_Revenue"].sum()
    total_cost = _maintenance_total(df) + df["Charging_Cost_INR"].sum()
    if total_revenue > 0:
        margin = max(0.0, min(100.0, 100.0 * (total_revenue - total_cost) / total_revenue))
    else:
        margin = 0.0

    score = (0.30 * avg_battery) + (0.30 * safe_rate) + (0.20 * uptime_rate) + (0.20 * margin)
    return {
        "score": round(score, 1),
        "avg_battery": avg_battery,
        "safe_rate": safe_rate,
        "uptime_rate": uptime_rate,
        "margin": margin,
        "in_maintenance": in_maintenance,
        "total_cars": total_cars,
    }


def _build_alerts(df: pd.DataFrame):
    """Scans the fleet and returns a list of alert dicts, each with
    severity ('Critical' | 'Warning' | 'Info'), a title, and a detail
    string. Pure rule-based — no external dependency, so it always
    works on whatever CSV is loaded."""
    alerts = []
    latest_per_car = df.sort_values("Date").groupby("Car_ID").last().reset_index()

    # 1) Critically low battery while still on the road
    critical_batt = latest_per_car[
        (latest_per_car["Battery_Remaining_Percent"] < 15)
        & (latest_per_car["Fleet_Status"].isin(["Running", "Idle"]))
    ]
    for _, row in critical_batt.iterrows():
        alerts.append({
            "severity": "Critical",
            "title": f"{row['Car_ID']} — critically low battery",
            "detail": f"{row['Battery_Remaining_Percent']:.0f}% remaining, driven by "
                      f"{row['Driver_Name']} in {row['City']}. Needs charging soon.",
        })

    # 2) Low (not critical) battery
    low_batt = latest_per_car[
        (latest_per_car["Battery_Remaining_Percent"] >= 15)
        & (latest_per_car["Battery_Remaining_Percent"] < 30)
    ]
    for _, row in low_batt.iterrows():
        alerts.append({
            "severity": "Warning",
            "title": f"{row['Car_ID']} — low battery",
            "detail": f"{row['Battery_Remaining_Percent']:.0f}% remaining. "
                      f"Driver: {row['Driver_Name']}.",
        })

    # 3) Cars currently under maintenance
    for _, row in latest_per_car[latest_per_car["Fleet_Status"] == "Under Maintenance"].iterrows():
        alerts.append({
            "severity": "Info",
            "title": f"{row['Car_ID']} — under maintenance",
            "detail": f"{row['Company']} {row['Model']} is currently off the road in {row['City']}.",
        })

    # 4) Drivers with a high rash-driving rate (min. 5 records, so one
    #    fast reading doesn't unfairly flag a driver)
    per_driver = df.groupby("Driver_Name").agg(
        records=("Driving_Style", "size"),
        rash=("Driving_Style", lambda s: (s == "Rash").sum()),
    )
    per_driver = per_driver[per_driver["records"] >= 5]
    per_driver["rash_rate"] = per_driver["rash"] / per_driver["records"] * 100
    for name, row in per_driver[per_driver["rash_rate"] >= 50].sort_values("rash_rate", ascending=False).iterrows():
        alerts.append({
            "severity": "Warning" if row["rash_rate"] < 75 else "Critical",
            "title": f"{name} — high rash-driving rate",
            "detail": f"{row['rash_rate']:.0f}% of {int(row['records'])} recorded trips "
                      f"were flagged Rash (110+ km/h).",
        })

    # 5) Cars operating at a net loss (maintenance + charging > revenue)
    maint_by_car = _maintenance_total_by_car(df)
    charge_by_car = df.groupby("Car_ID")["Charging_Cost_INR"].sum()
    rev_by_car = df.groupby("Car_ID")["Recognized_Revenue"].sum()
    net_by_car = rev_by_car.reindex(maint_by_car.index, fill_value=0) \
        - maint_by_car - charge_by_car.reindex(maint_by_car.index, fill_value=0)
    for car_id, net in net_by_car[net_by_car < 0].sort_values().items():
        alerts.append({
            "severity": "Warning",
            "title": f"{car_id} — running at a net loss",
            "detail": f"Net profit is ₹{net:,.0f} (maintenance + charging currently "
                      f"exceed recognized revenue for this car).",
        })

    # 6) Cars with an unusually high harsh-events count (top 5%, at
    #    least 3 events, so it only fires on real outliers)
    harsh_by_car = df.groupby("Car_ID")["Harsh_Events"].sum()
    if not harsh_by_car.empty and harsh_by_car.max() > 0:
        threshold = max(3, harsh_by_car.quantile(0.95))
        for car_id, count in harsh_by_car[harsh_by_car >= threshold].sort_values(ascending=False).items():
            alerts.append({
                "severity": "Warning",
                "title": f"{car_id} — unusually high harsh-events count",
                "detail": f"{int(count)} harsh braking/acceleration events recorded — "
                          f"in the top 5% of the fleet.",
            })

    severity_order = {"Critical": 0, "Warning": 1, "Info": 2}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))
    return alerts


_SEVERITY_STYLE = {
    "Critical": ("#fdeceb", "#d70015", "🔴"),
    "Warning": ("#fff6e5", "#b25000", "🟠"),
    "Info": ("#e8f2ff", "#0071e3", "🔵"),
}


def _show_alerts_health(df: pd.DataFrame):
    st.subheader("Fleet Health Score")
    st.caption("A single 0-100 score combining battery health, safe-driving rate, "
               "fleet uptime, and profit margin.")

    health = _health_score_components(df)

    g1, g2 = st.columns([1, 2])
    with g1:
        score = health["score"]
        gauge_color = GREEN if score >= 75 else ("#ff9f0a" if score >= 50 else RED)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 50], "color": "#fdeceb"},
                    {"range": [50, 75], "color": "#fff6e5"},
                    {"range": [75, 100], "color": "#e6f7ec"},
                ],
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g2:
        st.metric("Avg. Battery Health", f"{health['avg_battery']:.0f}%")
        m1, m2, m3 = st.columns(3)
        m1.metric("Safe-Driving Rate", f"{health['safe_rate']:.0f}%")
        m2.metric("Fleet Uptime", f"{health['uptime_rate']:.0f}%",
                   help=f"{health['in_maintenance']} of {health['total_cars']} cars "
                        f"currently under maintenance.")
        m3.metric("Profit Margin", f"{health['margin']:.0f}%")

    st.divider()

    st.subheader("Active Alerts")
    alerts = _build_alerts(df)

    if not alerts:
        st.success("No active alerts — the fleet is operating within normal parameters.")
        return

    counts = pd.Series([a["severity"] for a in alerts]).value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical", int(counts.get("Critical", 0)))
    c2.metric("🟠 Warning", int(counts.get("Warning", 0)))
    c3.metric("🔵 Info", int(counts.get("Info", 0)))

    sev_filter = st.multiselect(
        "Filter by severity", ["Critical", "Warning", "Info"],
        default=["Critical", "Warning", "Info"], key="alerts_sev_filter",
    )

    st.write("")
    for alert in alerts:
        if alert["severity"] not in sev_filter:
            continue
        bg, color, icon = _SEVERITY_STYLE[alert["severity"]]
        st.markdown(
            f'<div style="background:{bg}; border-left: 4px solid {color}; '
            f'border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">'
            f'<div style="font-weight:700; color:{color}; font-size:14px;">'
            f'{icon} {alert["title"]}</div>'
            f'<div style="font-size:13px; color:#1d1d1f; margin-top:2px;">{alert["detail"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 10 — User Management (NEW — pro upgrade)
# ============================================================
def _show_user_management(df: pd.DataFrame):
    st.subheader("User Management")
    st.caption("Manage every account that can sign in — activate/deactivate, "
               "change roles, reset passwords, or remove accounts.")

    db.init_db()
    users = db.list_users()
    current_username = st.session_state.get("username")

    if not users:
        st.info("No user accounts found.")
        return

    users_df = pd.DataFrame(users)

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Total Accounts", len(users_df))
    u2.metric("Admins", int((users_df["role"] == "Admin").sum()))
    u3.metric("Drivers", int((users_df["role"] == "Driver").sum()))
    u4.metric("Disabled Accounts", int((users_df["is_active"] == 0).sum()))

    st.divider()

    f1, f2 = st.columns(2)
    with f1:
        role_filter = st.multiselect("Role", ["Admin", "Driver"], default=["Admin", "Driver"],
                                      key="um_role_filter")
    with f2:
        status_filter = st.multiselect("Status", ["Active", "Disabled"], default=["Active", "Disabled"],
                                        key="um_status_filter")

    view_df = users_df[users_df["role"].isin(role_filter)]
    active_mask = view_df["is_active"] == 1
    keep = pd.Series(False, index=view_df.index)
    if "Active" in status_filter:
        keep |= active_mask
    if "Disabled" in status_filter:
        keep |= ~active_mask
    view_df = view_df[keep].copy()
    view_df["Status"] = view_df["is_active"].map({1: "Active", 0: "Disabled"})

    display_cols = ["username", "full_name", "role", "email", "assigned_car_id",
                     "Status", "last_login", "created_at"]
    st.dataframe(
        view_df[display_cols].rename(columns={
            "username": "Username", "full_name": "Full Name", "role": "Role",
            "email": "Email", "assigned_car_id": "Assigned Car",
            "last_login": "Last Login", "created_at": "Created",
        }),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Manage an Account")
    usernames = sorted(users_df["username"].tolist())
    selected_user = st.selectbox("Select account", usernames, key="um_selected_user")
    user_row = users_df[users_df["username"] == selected_user].iloc[0]

    is_self = (selected_user == current_username)
    if is_self:
        st.info("This is your own account — role changes, deactivation, and deletion "
                "are disabled here to prevent accidentally locking yourself out.")

    status_label = "Active" if user_row["is_active"] else "Disabled"
    st.markdown(
        f"**{user_row['full_name'] or user_row['username']}** &middot; "
        f"`{user_row['username']}` &middot; {user_row['role']} &middot; "
        f"Status: **{status_label}**"
    )

    mc1, mc2, mc3 = st.columns(3)

    with mc1:
        st.markdown("**Access**")
        if user_row["is_active"]:
            if st.button("Deactivate account", key="um_deactivate", disabled=is_self,
                         use_container_width=True):
                db.set_active(selected_user, False)
                db.log_audit(current_username, "Deactivated account", target=selected_user)
                st.success(f"{selected_user} deactivated.")
                st.rerun()
        else:
            if st.button("Reactivate account", key="um_reactivate", type="primary",
                         use_container_width=True):
                db.set_active(selected_user, True)
                db.log_audit(current_username, "Reactivated account", target=selected_user)
                st.success(f"{selected_user} reactivated.")
                st.rerun()

        new_role = st.selectbox("Role", ["Admin", "Driver"],
                                 index=["Admin", "Driver"].index(user_row["role"]),
                                 key="um_new_role", disabled=is_self)
        if st.button("Update role", key="um_update_role", disabled=is_self, use_container_width=True):
            db.set_role(selected_user, new_role)
            db.log_audit(current_username, "Changed role", target=selected_user, details=f"new_role={new_role}")
            st.success(f"{selected_user} is now {new_role}.")
            st.rerun()

    with mc2:
        st.markdown("**Reset password**")
        with st.form("um_reset_pw_form"):
            new_pw = st.text_input("New password", type="password", key="um_new_pw")
            confirm_pw = st.text_input("Confirm new password", type="password", key="um_confirm_pw")
            reset_submitted = st.form_submit_button("Reset password", use_container_width=True)
        if reset_submitted:
            if not new_pw or len(new_pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_pw != confirm_pw:
                st.error("Passwords don't match.")
            else:
                db.admin_reset_password(selected_user, new_pw)
                db.log_audit(current_username, "Reset password", target=selected_user)
                st.success(f"Password reset for {selected_user}.")

    with mc3:
        st.markdown("**Delete account**")
        st.caption("This permanently removes the account. This cannot be undone.")
        confirm_delete = st.checkbox(f"I understand, delete {selected_user}",
                                      key="um_confirm_delete", disabled=is_self)
        if st.button("Delete account", key="um_delete", disabled=(is_self or not confirm_delete),
                     use_container_width=True):
            db.delete_user(selected_user)
            db.log_audit(current_username, "Deleted account", target=selected_user)
            st.success(f"{selected_user} deleted.")
            st.rerun()

    st.divider()

    st.subheader("Create a New Account")
    st.caption("Add a driver or admin account directly, without them signing up themselves.")
    car_options = sorted(df["Car_ID"].unique().tolist()) if df is not None and "Car_ID" in df.columns else []

    with st.form("um_create_user_form"):
        cu1, cu2 = st.columns(2)
        with cu1:
            cu_full_name = st.text_input("Full name", key="um_cu_full_name")
            cu_username = st.text_input("Username", key="um_cu_username")
            cu_email = st.text_input("Email (optional)", key="um_cu_email")
        with cu2:
            cu_role = st.selectbox("Role", ["Driver", "Admin"], key="um_cu_role")
            cu_car = None
            if cu_role == "Driver" and car_options:
                cu_car = st.selectbox("Assigned car", car_options, key="um_cu_car")
            cu_password = st.text_input("Temporary password", type="password", key="um_cu_password")
        create_submitted = st.form_submit_button("Create account", type="primary")

    if create_submitted:
        success, message = db.register_user(
            username=cu_username,
            password=cu_password,
            confirm_password=cu_password,
            role=cu_role,
            full_name=cu_full_name,
            email=cu_email,
            assigned_car_id=cu_car or "",
        )
        if success:
            db.log_audit(current_username, "Created account (by admin)", target=cu_username,
                         details=f"role={cu_role}")
            st.success(f"Account created for {cu_username}.")
            st.rerun()
        else:
            st.error(message)


# ============================================================
# TAB 7 — Predictive Maintenance (NEW — pro upgrade)
# ============================================================
def _minmax_norm(series: pd.Series) -> pd.Series:
    """Scales a series to 0-100. A flat series (all equal values) maps
    to 0 everywhere instead of dividing by zero."""
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def _predictive_maintenance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic (rule-based, no external model needed) risk score per
    car: higher total distance covered, more harsh-driving events, and
    deeper average battery discharge all correlate with faster wear —
    weighted and combined into one 0-100 Risk Score."""
    latest = df.sort_values("Date").groupby("Car_ID").last()

    per_car = df.groupby("Car_ID").agg(
        Company=("Company", "first"),
        Model=("Model", "first"),
        Total_Odometer_km=("Total_Odometer_km", "max"),
        Harsh_Events_Total=("Harsh_Events", "sum"),
        Avg_Battery_Pct_Used=("Battery_Pct_Used", "mean"),
        Records=("Date", "count"),
    )
    per_car["Maintenance_Cost_So_Far"] = _maintenance_total_by_car(df)
    per_car["Current_Status"] = latest["Fleet_Status"]

    per_car["_odo_n"] = _minmax_norm(per_car["Total_Odometer_km"])
    per_car["_harsh_n"] = _minmax_norm(per_car["Harsh_Events_Total"])
    per_car["_batt_n"] = _minmax_norm(per_car["Avg_Battery_Pct_Used"])
    per_car["_cost_n"] = _minmax_norm(per_car["Maintenance_Cost_So_Far"])

    per_car["Risk_Score"] = (
        0.35 * per_car["_odo_n"]
        + 0.30 * per_car["_harsh_n"]
        + 0.20 * per_car["_batt_n"]
        + 0.15 * per_car["_cost_n"]
    ).round(1)

    def _bucket(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"

    def _recommendation(row):
        if row["Risk_Score"] >= 70:
            return "Schedule an inspection soon — high wear indicators."
        elif row["Risk_Score"] >= 40:
            return "Monitor — keep an eye on upcoming service intervals."
        return "No action needed — wear indicators are within normal range."

    per_car["Risk_Level"] = per_car["Risk_Score"].apply(_bucket)
    per_car["Recommendation"] = per_car.apply(_recommendation, axis=1)
    per_car = per_car.drop(columns=["_odo_n", "_harsh_n", "_batt_n", "_cost_n"])
    return per_car.reset_index().sort_values("Risk_Score", ascending=False)


_RISK_COLORS = {"High": RED, "Medium": "#ff9f0a", "Low": GREEN}


def _show_predictive_maintenance(df: pd.DataFrame):
    st.subheader("Predictive Maintenance")
    st.caption("A rule-based risk score (0-100) per car, combining total distance covered, "
               "harsh-driving events, average battery discharge depth, and maintenance spend "
               "so far — to flag which cars are most likely to need attention next.")

    risk_df = _predictive_maintenance_table(df)

    counts = risk_df["Risk_Level"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Risk", int(counts.get("High", 0)))
    c2.metric("🟠 Medium Risk", int(counts.get("Medium", 0)))
    c3.metric("🟢 Low Risk", int(counts.get("Low", 0)))

    st.divider()

    top_risk = risk_df.head(10)
    fig_risk = px.bar(
        top_risk, x="Risk_Score", y="Car_ID", orientation="h",
        labels={"Risk_Score": "Risk Score (0-100)", "Car_ID": "Car"},
        hover_data=["Company", "Model", "Current_Status", "Total_Odometer_km"],
    )
    fig_risk.update_traces(marker_color=[_RISK_COLORS[lvl] for lvl in top_risk["Risk_Level"]])
    fig_risk = _chart_layout(fig_risk, "Top 10 Cars by Maintenance Risk")
    fig_risk.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()

    st.subheader("Full Risk Table")
    level_filter = st.multiselect("Risk level", ["High", "Medium", "Low"],
                                   default=["High", "Medium", "Low"], key="pm_level_filter")
    shown = risk_df[risk_df["Risk_Level"].isin(level_filter)]

    display_cols = ["Car_ID", "Company", "Model", "Current_Status", "Risk_Score", "Risk_Level",
                     "Total_Odometer_km", "Harsh_Events_Total", "Avg_Battery_Pct_Used",
                     "Maintenance_Cost_So_Far", "Recommendation"]
    st.dataframe(shown[display_cols], use_container_width=True)
    st.download_button(
        "Download Risk Report (CSV)",
        data=shown[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="predictive_maintenance_risk.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 8 — Driver Leaderboard (NEW — pro upgrade)
# ============================================================
def _driver_leaderboard_table(df: pd.DataFrame) -> pd.DataFrame:
    per_driver = df.groupby("Driver_Name").agg(
        Records=("Date", "count"),
        Safe_Records=("Driving_Style", lambda s: (s == "Safe").sum()),
        Rash_Records=("Driving_Style", lambda s: (s == "Rash").sum()),
        Harsh_Events_Total=("Harsh_Events", "sum"),
        Distance_km=("Distance_km", "sum"),
        Recognized_Revenue=("Recognized_Revenue", "sum"),
        Completed_Trips=("Trip_Status", lambda s: (s == "Completed").sum()),
    )
    per_driver["Safe_Rate"] = (per_driver["Safe_Records"] / per_driver["Records"] * 100).round(1)
    per_driver["Revenue_Per_Km"] = (
        per_driver["Recognized_Revenue"] / per_driver["Distance_km"].replace(0, np.nan)
    ).fillna(0)
    per_driver["Harsh_Rate"] = (per_driver["Harsh_Events_Total"] / per_driver["Records"]).fillna(0)
    per_driver["Completion_Rate"] = (per_driver["Completed_Trips"] / per_driver["Records"] * 100).round(1)

    safety_n = per_driver["Safe_Rate"]
    revenue_n = _minmax_norm(per_driver["Revenue_Per_Km"])
    harsh_n = 100 - _minmax_norm(per_driver["Harsh_Rate"])
    completion_n = per_driver["Completion_Rate"]

    per_driver["Performance_Score"] = (
        0.35 * safety_n + 0.30 * revenue_n + 0.20 * harsh_n + 0.15 * completion_n
    ).round(1)

    per_driver = per_driver.sort_values("Performance_Score", ascending=False).reset_index()
    per_driver.insert(0, "Rank", range(1, len(per_driver) + 1))

    def _badge(rank):
        return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")

    per_driver["Medal"] = per_driver["Rank"].apply(_badge)
    return per_driver


def _show_driver_leaderboard(df: pd.DataFrame):
    st.subheader("Driver Leaderboard")
    st.caption("A composite 0-100 performance score per driver — 35% safe-driving rate, "
               "30% revenue efficiency (₹ per km), 20% low harsh-events rate, "
               "15% trip-completion rate.")

    board = _driver_leaderboard_table(df)

    if board.empty:
        st.info("No driver data available.")
        return

    top3 = board.head(3)
    cols = st.columns(3)
    for col, (_, row) in zip(cols, top3.iterrows()):
        with col:
            st.markdown(
                f'<div style="background:#ffffff; border-radius:16px; padding:18px; '
                f'text-align:center; box-shadow:0 8px 24px rgba(0,0,0,0.05); '
                f'border:1px solid rgba(0,0,0,0.04);">'
                f'<div style="font-size:34px;">{row["Medal"]}</div>'
                f'<div style="font-weight:800; font-size:16px; color:#1d1d1f;">{row["Driver_Name"]}</div>'
                f'<div style="font-size:13px; color:#86868b;">Score: {row["Performance_Score"]:.1f}/100</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    top_n = st.slider("Show top N drivers", min_value=5, max_value=max(5, len(board)),
                       value=min(15, len(board)), key="dl_leaderboard_topn")
    chart_df = board.head(top_n)
    fig_board = px.bar(
        chart_df, x="Performance_Score", y="Driver_Name", orientation="h",
        labels={"Performance_Score": "Performance Score", "Driver_Name": "Driver"},
        color_discrete_sequence=[ACCENT],
    )
    fig_board = _chart_layout(fig_board, f"Top {top_n} Drivers by Performance Score")
    fig_board.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_board, use_container_width=True)

    st.divider()

    st.subheader("Full Leaderboard")
    display_cols = ["Rank", "Medal", "Driver_Name", "Performance_Score", "Safe_Rate",
                     "Revenue_Per_Km", "Harsh_Rate", "Completion_Rate",
                     "Recognized_Revenue", "Distance_km", "Records"]
    st.dataframe(board[display_cols], use_container_width=True)
    st.download_button(
        "Download Leaderboard (CSV)",
        data=board[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="driver_leaderboard.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 9 — Reports Center (NEW — pro upgrade)
# ============================================================
def _build_report_workbook(df: pd.DataFrame) -> bytes:
    """Builds a multi-sheet Excel workbook (Fleet Summary, Driver Summary,
    Car Summary, Predictive Maintenance, Alerts) in-memory and returns
    the raw bytes for a download button — nothing is written to disk."""
    buffer = io.BytesIO()

    health = _health_score_components(df)
    fleet_summary = pd.DataFrame([{
        "Fleet Health Score": health["score"],
        "Avg Battery Health (%)": round(health["avg_battery"], 1),
        "Safe-Driving Rate (%)": round(health["safe_rate"], 1),
        "Fleet Uptime (%)": round(health["uptime_rate"], 1),
        "Profit Margin (%)": round(health["margin"], 1),
        "Total Cars": health["total_cars"],
        "Cars Under Maintenance": health["in_maintenance"],
        "Total Recognized Revenue (INR)": df["Recognized_Revenue"].sum(),
        "Total Maintenance Cost (INR)": _maintenance_total(df),
        "Total Charging Cost (INR)": df["Charging_Cost_INR"].sum(),
        "Report Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }])

    driver_summary = _driver_leaderboard_table(df).drop(columns=["Medal"])
    car_summary = _predictive_maintenance_table(df)

    alerts = _build_alerts(df)
    alerts_df = pd.DataFrame(alerts) if alerts else pd.DataFrame(
        columns=["severity", "title", "detail"])

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        fleet_summary.to_excel(writer, sheet_name="Fleet Summary", index=False)
        driver_summary.to_excel(writer, sheet_name="Driver Summary", index=False)
        car_summary.to_excel(writer, sheet_name="Car Summary", index=False)
        alerts_df.to_excel(writer, sheet_name="Active Alerts", index=False)

    return buffer.getvalue()


def _show_reports_center(df: pd.DataFrame):
    st.subheader("Reports Center")
    st.caption("Generate a single Excel workbook with Fleet Summary, Driver Summary, "
               "Car Summary (predictive maintenance), and Active Alerts — ready to share "
               "in a review or a stakeholder update.")

    months_ordered = df[["Month", "Month_Num"]].drop_duplicates().sort_values("Month_Num")["Month"].tolist()
    scope = st.radio("Report scope", ["Entire dataset"] + months_ordered, horizontal=True, key="rc_scope")
    report_df = df if scope == "Entire dataset" else df[df["Month"] == scope]

    st.write("")
    if st.button("Generate Report", type="primary", key="rc_generate"):
        with st.spinner("Building workbook..."):
            workbook_bytes = _build_report_workbook(report_df)
        st.session_state["rc_last_report"] = workbook_bytes
        st.session_state["rc_last_scope"] = scope
        st.success(f"Report generated for: {scope}")

    if "rc_last_report" in st.session_state:
        file_scope = st.session_state.get("rc_last_scope", "report").replace(" ", "_")
        st.download_button(
            "Download Report (Excel)",
            data=st.session_state["rc_last_report"],
            file_name=f"ev_fleet_report_{file_scope}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.caption("Preview — Fleet Summary for the selected scope")
    st.dataframe(pd.DataFrame([{
        "Records in scope": len(report_df),
        "Cars in scope": report_df["Car_ID"].nunique(),
        "Drivers in scope": report_df["Driver_Name"].nunique(),
        "Recognized Revenue (INR)": f"{report_df['Recognized_Revenue'].sum():,.0f}",
    }]), use_container_width=True)


# ============================================================
# TAB 11 — Audit Log (NEW — pro upgrade)
# ============================================================
def _show_audit_log():
    st.subheader("Audit Log")
    st.caption("Every sign-in, sign-up, and admin account-management action, newest first. "
               "Read-only — nothing here can be edited or deleted.")

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        limit = st.selectbox("Show last", [50, 100, 200, 500], index=1, key="al_limit")
    with f2:
        actor_search = st.text_input("Filter by username", key="al_actor_search")
    with f3:
        action_search = st.text_input("Filter by action contains", key="al_action_search")

    logs = db.list_audit_logs(
        limit=limit,
        actor=actor_search or None,
        action_contains=action_search or None,
    )

    if not logs:
        st.info("No audit log entries yet — actions will appear here as they happen.")
        return

    logs_df = pd.DataFrame(logs).rename(columns={
        "actor": "User", "action": "Action", "target": "Target",
        "details": "Details", "created_at": "Timestamp (UTC)",
    })
    st.dataframe(logs_df, use_container_width=True)
    st.download_button(
        "Download Audit Log (CSV)",
        data=logs_df.to_csv(index=False).encode("utf-8"),
        file_name="audit_log.csv",
        mime="text/csv",
    )
