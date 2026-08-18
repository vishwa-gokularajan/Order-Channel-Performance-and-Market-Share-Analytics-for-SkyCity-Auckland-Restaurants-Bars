"""
SkyCity Auckland Analytics - Data Cleaning & Preprocessing Pipeline
Handles missing values, data validation, outlier detection, and derived feature engineering.
"""

import pandas as pd
import numpy as np

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes full data cleaning, validation, type conversions, and feature engineering.
    """
    df = df.copy()
    
    # 1. Missing-value Analysis & Imputation
    numeric_cols = ['Quantity', 'Gross_Revenue', 'Discount', 'Net_Revenue', 'Cost', 'Profit', 'Customer_Rating']
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    cat_cols = ['Restaurant_Name', 'Restaurant_Segment', 'Cuisine_Type', 'Order_Channel', 'Subregion', 'Customer_Type', 'Order_Status']
    for col in cat_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 2. Duplicate Detection
    df = df.drop_duplicates()
    
    # 3. Data-Type Correction & DateTime Conversion
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
    if 'Order_Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(12).astype(int)
    else:
        df['Hour'] = 12
        
    # 4. Feature Engineering - Date/Time
    df['Year'] = df['Order_Date'].dt.year
    df['Month'] = df['Order_Date'].dt.month
    df['Month_Name'] = df['Order_Date'].dt.strftime('%b')
    df['Quarter'] = df['Order_Date'].dt.quarter
    df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
    df['Day_Name'] = df['Order_Date'].dt.strftime('%a')
    
    # Peak vs Off-Peak hours (Lunch: 11-14, Dinner: 17-21)
    df['Is_Peak_Hour'] = df['Hour'].apply(lambda h: 'Peak' if (11 <= h <= 14 or 17 <= h <= 21) else 'Off-Peak')
    
    # 5. Financial & Channel Derived Features
    # Net Revenue fallback if missing
    if 'Net_Revenue' not in df.columns or df['Net_Revenue'].isnull().all():
        df['Net_Revenue'] = df['Gross_Revenue'] - df.get('Discount', 0)
        
    df['Profit_Margin'] = np.where(df['Net_Revenue'] > 0, df['Profit'] / df['Net_Revenue'], 0.0)
    df['Profit_Margin'] = df['Profit_Margin'].clip(-1.0, 1.0)
    
    # Direct vs Aggregator classification
    aggregator_channels = ['Uber Eats', 'DoorDash']
    df['Channel_Category'] = df['Order_Channel'].apply(lambda ch: 'Aggregator' if ch in aggregator_channels else 'Direct')
    
    # 6. Restaurant-Level Aggregator Dependency %
    rest_channel_rev = df.groupby(['Restaurant_Name', 'Channel_Category'])['Net_Revenue'].sum().unstack(fill_value=0)
    if 'Aggregator' not in rest_channel_rev.columns:
        rest_channel_rev['Aggregator'] = 0
    if 'Direct' not in rest_channel_rev.columns:
        rest_channel_rev['Direct'] = 0
        
    rest_channel_rev['Total_Revenue'] = rest_channel_rev['Aggregator'] + rest_channel_rev['Direct']
    rest_channel_rev['Channel_Dependency_Pct'] = np.where(
        rest_channel_rev['Total_Revenue'] > 0,
        (rest_channel_rev['Aggregator'] / rest_channel_rev['Total_Revenue']) * 100,
        0.0
    )
    
    def classify_dependency(pct):
        if pct < 30.0:
            return 'Low Dependency'
        elif pct <= 50.0:
            return 'Medium Dependency'
        else:
            return 'High Dependency'
            
    rest_channel_rev['Dependency_Class'] = rest_channel_rev['Channel_Dependency_Pct'].apply(classify_dependency)
    
    # Map dependency back to order-level dataframe
    df['Channel_Dependency_Pct'] = df['Restaurant_Name'].map(rest_channel_rev['Channel_Dependency_Pct'])
    df['Dependency_Class'] = df['Restaurant_Name'].map(rest_channel_rev['Dependency_Class'])
    
    # 7. Market Share Derivations
    total_market_rev = df['Net_Revenue'].sum()
    rest_market_share = (df.groupby('Restaurant_Name')['Net_Revenue'].sum() / total_market_rev * 100).to_dict()
    channel_market_share = (df.groupby('Order_Channel')['Net_Revenue'].sum() / total_market_rev * 100).to_dict()
    
    df['Restaurant_Market_Share_Pct'] = df['Restaurant_Name'].map(rest_market_share)
    df['Channel_Market_Share_Pct'] = df['Order_Channel'].map(channel_market_share)
    
    return df

def get_outlier_summary(df: pd.DataFrame, col: str) -> dict:
    """Detect outliers using the Interquartile Range (IQR) method."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return {
        'column': col,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': len(outliers),
        'outlier_pct': len(outliers) / len(df) * 100
    }
