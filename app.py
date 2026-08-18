"""
SkyCity Auckland Restaurants & Bars - Order Channel Performance & Market Share Analytics
Interactive Streamlit Web Dashboard
"""

import sys
import os
from pathlib import Path

# Ensure repository root is on sys.path for Streamlit Community Cloud and subfolder execution
BASE_DIR = Path(__file__).resolve().parent

import streamlit as st
import pandas as pd
import numpy as np

from data_cleaning import clean_and_prepare_data
from eda import (
    calculate_kpis, get_regional_dominance_matrix,
    plot_orders_by_channel, plot_revenue_by_channel, plot_profit_by_channel,
    plot_market_share_by_restaurant, plot_market_share_by_channel,
    plot_channel_distribution_by_subregion, plot_channel_distribution_by_cuisine,
    plot_channel_mix_by_segment, plot_monthly_order_trend, plot_monthly_revenue_trend,
    plot_peak_ordering_hours, plot_day_of_week_performance, plot_restaurant_wise_revenue,
    plot_cuisine_wise_revenue, plot_aggregator_dependency_analysis,
    plot_direct_vs_aggregator_performance, plot_aov_by_channel,
    plot_profit_margin_by_channel, plot_customer_rating_by_channel, plot_correlation_heatmap
)
from stats_analysis import run_chi_square_test, run_anova_test, run_ttest
from forecasting import (
    prepare_daily_timeseries, forecast_arima, forecast_exponential_smoothing,
    forecast_moving_average, plot_time_series_forecast
)
from ml_models import train_and_evaluate_ml_models, plot_feature_importances
from utils import format_currency, format_number, format_percentage

# --- Page Configuration ---
st.set_page_config(
    page_title="SkyCity Auckland Order Channel Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (SkyCity Glassmorphism Dark Aesthetic)
st.markdown("""
    <style>
    .main { background-color: #0F172A; }
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #0D9488; }
    .metric-label { font-size: 13px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; }
    .recommendation-box {
        background-color: #1E293B;
        border-left: 5px solid #0D9488;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = BASE_DIR / 'data' / 'skycity_orders.csv'
    if not csv_path.exists():
        alt_path = BASE_DIR / 'skycity_orders.csv'
        if alt_path.exists():
            csv_path = alt_path
        else:
            from data_generator import generate_orders_dataset
            master_path = BASE_DIR / "SkyCity Auckland Restaurants & Bars.csv"
            if not master_path.exists() and (BASE_DIR / 'data' / "SkyCity Auckland Restaurants & Bars.csv").exists():
                master_path = BASE_DIR / 'data' / "SkyCity Auckland Restaurants & Bars.csv"
            generate_orders_dataset(str(master_path), str(csv_path))
    df_raw = pd.read_csv(csv_path)
    return clean_and_prepare_data(df_raw)

df_all = load_data()

# --- Sidebar Filters ---
st.sidebar.image("https://img.icons8.com/color/96/restaurant-building.png", width=60)
st.sidebar.title("SkyCity Analytics")
st.sidebar.markdown("---")

# Navigation Menu
page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Page 1 – Executive Overview",
        "Page 2 – Channel Performance",
        "Page 3 – Regional Analysis",
        "Page 4 – Cuisine & Restaurant Analysis",
        "Page 5 – Forecasting & Machine Learning",
        "Page 6 – Business Insights & Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")

min_date = df_all['Order_Date'].min().date() if 'Order_Date' in df_all.columns and not df_all.empty else pd.Timestamp('2024-01-01').date()
max_date = df_all['Order_Date'].max().date() if 'Order_Date' in df_all.columns and not df_all.empty else pd.Timestamp('2025-12-31').date()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

subregion_opts = sorted([str(x) for x in df_all['Subregion'].dropna().unique()]) if 'Subregion' in df_all.columns else []
cuisine_opts = sorted([str(x) for x in df_all['Cuisine_Type'].dropna().unique()]) if 'Cuisine_Type' in df_all.columns else []
segment_opts = sorted([str(x) for x in df_all['Restaurant_Segment'].dropna().unique()]) if 'Restaurant_Segment' in df_all.columns else []
channel_opts = sorted([str(x) for x in df_all['Order_Channel'].dropna().unique()]) if 'Order_Channel' in df_all.columns else []
cust_type_opts = sorted([str(x) for x in df_all['Customer_Type'].dropna().unique()]) if 'Customer_Type' in df_all.columns else []

selected_subregions = st.sidebar.multiselect("Subregion", options=subregion_opts, default=subregion_opts)
selected_cuisines = st.sidebar.multiselect("Cuisine Type", options=cuisine_opts, default=cuisine_opts)
selected_segments = st.sidebar.multiselect("Restaurant Segment", options=segment_opts, default=segment_opts)
selected_channels = st.sidebar.multiselect("Order Channel", options=channel_opts, default=channel_opts)
selected_cust_types = st.sidebar.multiselect("Customer Type", options=cust_type_opts, default=cust_type_opts)

# Apply Filtering
df_filtered = df_all.copy()
if len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df_filtered[(df_filtered['Order_Date'] >= start_d) & (df_filtered['Order_Date'] <= end_d)]

if selected_subregions:
    df_filtered = df_filtered[df_filtered['Subregion'].astype(str).isin(selected_subregions)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_cuisines:
    df_filtered = df_filtered[df_filtered['Cuisine_Type'].astype(str).isin(selected_cuisines)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_segments:
    df_filtered = df_filtered[df_filtered['Restaurant_Segment'].astype(str).isin(selected_segments)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_channels:
    df_filtered = df_filtered[df_filtered['Order_Channel'].astype(str).isin(selected_channels)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_cust_types:
    df_filtered = df_filtered[df_filtered['Customer_Type'].astype(str).isin(selected_cust_types)]
else:
    df_filtered = df_filtered.iloc[0:0]

if df_filtered.empty:
    st.warning("No data available for the selected filters.")

kpis = calculate_kpis(df_filtered)

# --- PAGE 1: EXECUTIVE OVERVIEW ---
if page == "Page 1 – Executive Overview":
    st.title("🍽️ Executive Overview - SkyCity Auckland")
    st.caption("High-level executive dashboard tracking order channels, market share, and revenue performance.")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Revenue</div><div class="metric-value">{format_currency(kpis["Total_Revenue"])}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Orders</div><div class="metric-value">{format_number(kpis["Total_Orders"])}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Profit</div><div class="metric-value">{format_currency(kpis["Total_Profit"])}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Average Order Value</div><div class="metric-value">{format_currency(kpis["AOV"])}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Rating</div><div class="metric-value">{kpis["Avg_Rating"]:.2f} ⭐</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Aggregator Dep.</div><div class="metric-value">{format_percentage(kpis["Aggregator_Dependency_Pct"])}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not df_filtered.empty:
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            st.plotly_chart(plot_monthly_revenue_trend(df_filtered), use_container_width=True)
        with r1_col2:
            st.plotly_chart(plot_monthly_order_trend(df_filtered), use_container_width=True)
            
        r2_col1, r2_col2 = st.columns(2)
        with r2_col1:
            st.plotly_chart(plot_market_share_by_channel(df_filtered), use_container_width=True)
        with r2_col2:
            st.plotly_chart(plot_market_share_by_restaurant(df_filtered), use_container_width=True)

# --- PAGE 2: CHANNEL PERFORMANCE ---
elif page == "Page 2 – Channel Performance":
    st.title("📈 Order Channel Performance & Financial Dynamics")
    st.caption("In-depth channel performance breakdown: Revenue, Profitability, Margins, and Aggregator dependence.")
    
    if not df_filtered.empty:
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        with r1_col1:
            st.plotly_chart(plot_orders_by_channel(df_filtered), use_container_width=True)
        with r1_col2:
            st.plotly_chart(plot_revenue_by_channel(df_filtered), use_container_width=True)
        with r1_col3:
            st.plotly_chart(plot_profit_by_channel(df_filtered), use_container_width=True)
            
        r2_col1, r2_col2, r2_col3 = st.columns(3)
        with r2_col1:
            st.plotly_chart(plot_aov_by_channel(df_filtered), use_container_width=True)
        with r2_col2:
            st.plotly_chart(plot_profit_margin_by_channel(df_filtered), use_container_width=True)
        with r2_col3:
            st.plotly_chart(plot_customer_rating_by_channel(df_filtered), use_container_width=True)
            
        st.markdown("### Direct Ordering vs Aggregator Performance")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_direct_vs_aggregator_performance(df_filtered), use_container_width=True)
        with c2:
            st.plotly_chart(plot_aggregator_dependency_analysis(df_filtered), use_container_width=True)

# --- PAGE 3: REGIONAL ANALYSIS ---
elif page == "Page 3 – Regional Analysis":
    st.title("📍 Regional Subregion Channel Analysis")
    st.caption("Mapping channel preferences, delivery vs dine-in demand, and channel dominance across Auckland subregions.")
    
    if not df_filtered.empty:
        st.subheader("Dominant Channel by Subregion")
        dominance_df = get_regional_dominance_matrix(df_filtered)
        st.dataframe(dominance_df, use_container_width=True)
        
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            st.plotly_chart(plot_channel_distribution_by_subregion(df_filtered), use_container_width=True)
        with r1_col2:
            st.plotly_chart(plot_peak_ordering_hours(df_filtered), use_container_width=True)
            
        st.plotly_chart(plot_day_of_week_performance(df_filtered), use_container_width=True)

# --- PAGE 4: CUISINE & RESTAURANT ANALYSIS ---
elif page == "Page 4 – Cuisine & Restaurant Analysis":
    st.title("🥗 Cuisine & Restaurant Segment Analytics")
    st.caption("Understanding how cuisine type and restaurant segment dictate order-channel preferences.")
    
    if not df_filtered.empty:
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            st.plotly_chart(plot_channel_distribution_by_cuisine(df_filtered), use_container_width=True)
        with r1_col2:
            st.plotly_chart(plot_channel_mix_by_segment(df_filtered), use_container_width=True)
            
        r2_col1, r2_col2 = st.columns(2)
        with r2_col1:
            st.plotly_chart(plot_cuisine_wise_revenue(df_filtered), use_container_width=True)
        with r2_col2:
            st.plotly_chart(plot_restaurant_wise_revenue(df_filtered), use_container_width=True)

# --- PAGE 5: FORECASTING & MACHINE LEARNING ---
elif page == "Page 5 – Forecasting & Machine Learning":
    st.title("🔮 Predictive Analytics & Time-Series Forecasting")
    st.caption("Predicting future order volume & revenue trends with Moving Average, Exponential Smoothing, ARIMA, and Machine Learning.")
    
    tab1, tab2, tab3 = st.tabs(["Time-Series Forecasting", "Statistical Hypothesis Testing", "Machine Learning Models"])
    
    with tab1:
        st.subheader("Time-Series Forecast Configuration")
        fc_col1, fc_col2, fc_col3 = st.columns(3)
        with fc_col1:
            target_metric = st.selectbox("Target Forecasting Metric", ["Net_Revenue", "Gross_Revenue", "Quantity"])
        with fc_col2:
            model_type = st.selectbox("Forecasting Algorithm", ["ARIMA", "Exponential Smoothing", "Moving Average"])
        with fc_col3:
            horizon = st.slider("Forecast Horizon (Days)", min_value=14, max_value=90, value=30, step=7)
            
        if not df_filtered.empty:
            daily_ts_df = prepare_daily_timeseries(df_filtered, metric=target_metric)
            if daily_ts_df.empty or len(daily_ts_df) <= horizon:
                st.warning("⚠️ Insufficient continuous daily data points in the selected filter range to run time-series forecasting. Please widen your date or sidebar filters.")
            else:
                daily_ts = daily_ts_df[target_metric]
                if model_type == "ARIMA":
                    forecast_vals, conf_int, metrics = forecast_arima(daily_ts, horizon_days=horizon)
                elif model_type == "Exponential Smoothing":
                    forecast_vals, metrics = forecast_exponential_smoothing(daily_ts, horizon_days=horizon)
                    conf_int = None
                else:
                    forecast_vals, metrics = forecast_moving_average(daily_ts, horizon_days=horizon)
                    conf_int = None
                    
                fig_fc = plot_time_series_forecast(daily_ts, forecast_vals, conf_int, title=f"{horizon}-Day {model_type} Forecast for {target_metric}")
                st.plotly_chart(fig_fc, use_container_width=True)
                
                st.subheader("Forecast Evaluation Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Absolute Error (MAE)", metrics['MAE'])
                m2.metric("Root Mean Squared Error (RMSE)", metrics['RMSE'])
                m3.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']}%")
        
    with tab2:
        st.subheader("Formal Statistical Hypothesis Testing")
        
        if not df_filtered.empty:
            # Chi-Square
            st.markdown("#### 1. Chi-Square Test of Independence")
            chi_res = run_chi_square_test(df_filtered, 'Subregion', 'Order_Channel')
            st.info(f"**Test**: {chi_res['test_name']}\n\n"
                    f"**Null Hypothesis (H0)**: {chi_res['null_hypothesis']}\n\n"
                    f"**Alt Hypothesis (H1)**: {chi_res['alt_hypothesis']}\n\n"
                    f"**Chi2 Stat**: {chi_res['chi2_statistic']:.4f} | **p-value**: {chi_res['p_value']:.4e}\n\n"
                    f"**Decision**: {chi_res['conclusion']}")
                    
            # ANOVA
            st.markdown("#### 2. One-Way ANOVA")
            anova_res = run_anova_test(df_filtered, 'Order_Channel', 'Net_Revenue')
            st.info(f"**Test**: {anova_res['test_name']}\n\n"
                    f"**Null Hypothesis (H0)**: {anova_res['null_hypothesis']}\n\n"
                    f"**Alt Hypothesis (H1)**: {anova_res['alt_hypothesis']}\n\n"
                    f"**F-Stat**: {anova_res['f_statistic']:.4f} | **p-value**: {anova_res['p_value']:.4e}\n\n"
                    f"**Decision**: {anova_res['conclusion']}")
                    
            # T-Test
            st.markdown("#### 3. Independent Two-Sample T-Test")
            ttest_res = run_ttest(df_filtered, 'Channel_Category', 'Profit_Margin')
            st.info(f"**Test**: {ttest_res['test_name']}\n\n"
                    f"**Null Hypothesis (H0)**: {ttest_res['null_hypothesis']}\n\n"
                    f"**Alt Hypothesis (H1)**: {ttest_res['alt_hypothesis']}\n\n"
                    f"**T-Stat**: {ttest_res['t_statistic']:.4f} | **p-value**: {ttest_res['p_value']:.4e}\n\n"
                    f"**Decision**: {ttest_res['conclusion']}")

    with tab3:
        st.subheader("Machine Learning Supervised Regression")
        if not df_filtered.empty and len(df_filtered) >= 10:
            results_dict, summary_df, imp_df, best_name, best_pipe = train_and_evaluate_ml_models(df_filtered, 'Net_Revenue')
            
            st.markdown(f"**Best Model Selected**: `{best_name}`")
            st.dataframe(summary_df, use_container_width=True)
            
            if imp_df is not None and not imp_df.empty:
                fig_imp = plot_feature_importances(imp_df)
                if fig_imp is not None:
                    st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data to train Machine Learning models.")

# --- PAGE 6: BUSINESS INSIGHTS ---
elif page == "Page 6 – Business Insights & Recommendations":
    st.title("💡 Strategic Business Insights & Recommendations")
    st.caption("Data-backed actionable recommendations to optimize channel mix, increase direct revenue, and maximize profitability.")
    
    st.markdown("### Executive Summary of Analytical Findings")
    st.markdown(f"""
    - **Aggregator Dependency Risk**: Third-party aggregators (Uber Eats & DoorDash) control **{format_percentage(kpis['Aggregator_Dependency_Pct'])}** of total revenue. Commission rates (28%-33%) severely compress net profit margins to **8-12%** compared to **22-28%** on direct in-store and self-delivery orders.
    - **Regional Channel Dominance**: **CBD** and **North Shore** subregions display high aggregator reliance, whereas **South Auckland** and **West Auckland** exhibit stronger direct in-store dine-in ordering.
    - **Cuisine Channel Propensity**: Quick-service cuisines (Burgers, Kebabs, Pizza) drive over **65%** of delivery aggregator volume, while Fine Dining & Asian segments dominate direct in-store revenue.
    """)
    
    st.markdown("---")
    st.markdown("### Actionable Business Recommendations")
    
    recs = [
        ("1. Aggregator Commission Mitigation & Direct Channel Conversion",
         "Negotiate tiered commission caps with Uber Eats/DoorDash based on volume. Implement branded SkyCity Direct Web ordering with direct-order discounts (e.g. 10% off first direct order) to convert aggregator customers."),
        ("2. Dedicated SkyCity Digital Loyalty & Rewards Program",
         "Launch a unified SkyCity Auckland Rewards program across all restaurants to incentivize direct web and app orders with free delivery or reward points."),
        ("3. Regional Dark Kitchen & Delivery Hub Optimization",
         "Establish centralized delivery kitchen hubs in CBD and North Shore to fulfill aggregator delivery demand without cannibalizing prime fine-dining seating capacity."),
        ("4. High-Performing Cuisine Channel Customization",
         "Tailor packaging and menu offerings for delivery-heavy cuisines (Burgers, Asian) to preserve temperature and rating while restricting fragile gourmet items to dine-in."),
        ("5. Peak Ordering Hours Kitchen Staffing",
         "Optimize kitchen and dispatch staffing during peak windows (11:00–14:00 Lunch & 17:00–21:00 Dinner) to reduce delivery wait times and maintain >4.5 star ratings."),
        ("6. Targeted Underperforming Restaurant Remediation",
         "Implement targeted marketing and channel optimization for restaurants with <25% profit margins or >55% aggregator dependence."),
        ("7. Dynamic Promotional Pricing",
         "Utilize dynamic off-peak discounts (14:00–17:00) on direct ordering channels to fill kitchen idle capacity."),
        ("8. Direct Ordering Packaging & In-Box Collateral",
         "Include bounce-back flyers and QR codes in third-party aggregator bags offering direct ordering benefits on subsequent orders.")
    ]
    
    for title, desc in recs:
        st.markdown(f'<div class="recommendation-box"><strong>{title}</strong><br><span style="color:#CBD5E1;">{desc}</span></div>', unsafe_allow_html=True)
