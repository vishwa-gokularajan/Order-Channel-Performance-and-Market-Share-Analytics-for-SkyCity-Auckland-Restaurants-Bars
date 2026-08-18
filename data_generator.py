"""
SkyCity Auckland Analytics - Synthetic Data Generator
Generates granular transaction-level order dataset (skycity_orders.csv)
anchored to master restaurant profiles (SkyCity Auckland Restaurants & Bars.csv).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_orders_dataset(master_csv_path: str, output_csv_path: str, num_orders: int = 35000, seed: int = 42):
    """
    Generate realistic order transactions using master restaurant metrics as probability anchors.
    """
    np.random.seed(seed)
    
    if not os.path.exists(master_csv_path):
        raise FileNotFoundError(f"Master CSV file not found at: {master_csv_path}")
        
    master_df = pd.read_csv(master_csv_path)
    
    # Pre-calculate sampling weights based on restaurant monthly orders
    weights = master_df['MonthlyOrders'].values / master_df['MonthlyOrders'].sum()
    
    # Sample restaurant indices
    sampled_indices = np.random.choice(len(master_df), size=num_orders, p=weights)
    sampled_restaurants = master_df.iloc[sampled_indices].reset_index(drop=True)
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days
    
    orders = []
    channels = ['In-Store', 'Uber Eats', 'DoorDash', 'Self-Delivery']
    cust_types = ['Regular', 'VIP', 'New', 'Corporate']
    cust_type_p = [0.45, 0.15, 0.25, 0.15]
    statuses = ['Completed', 'Cancelled', 'Refunded']
    status_p = [0.94, 0.04, 0.02]
    
    for i in range(num_orders):
        row = sampled_restaurants.iloc[i]
        
        # Determine channel based on restaurant channel shares
        ch_shares = np.array([
            float(row['InStoreShare']),
            float(row['UE_share']),
            float(row['DD_share']),
            float(row['SD_share'])
        ])
        ch_shares = ch_shares / ch_shares.sum()
        chosen_channel = np.random.choice(channels, p=ch_shares)
        
        # Aggregator mapping
        if chosen_channel == 'In-Store':
            agg_name = 'In-Store Direct'
        elif chosen_channel == 'Uber Eats':
            agg_name = 'Uber Eats'
        elif chosen_channel == 'DoorDash':
            agg_name = 'DoorDash'
        else:
            agg_name = 'SkyCity Direct Web'
            
        # Date & Time sampling with peak-hour weightings (11-14, 17-21)
        rand_days = np.random.randint(0, date_range_days + 1)
        ord_date = start_date + timedelta(days=int(rand_days))
        
        # Hourly weights (higher during lunch & dinner)
        hour_weights = [
            0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.05, 0.04, 0.05,  # 0-10
            0.10, 0.12, 0.08, 0.04, 0.03, 0.05, 0.11, 0.10, 0.06, 0.03, 0.02, 0.015, 0.01 # 11-23
        ]
        hour_weights = np.array(hour_weights) / sum(hour_weights)
        ord_hour = np.random.choice(24, p=hour_weights)
        ord_minute = np.random.randint(0, 60)
        ord_second = np.random.randint(0, 60)
        ord_time_str = f"{ord_hour:02d}:{ord_minute:02d}:{ord_second:02d}"
        
        # Order Economics
        aov_base = float(row['AOV'])
        quantity = int(np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.18, 0.08, 0.04]))
        gross_rev = round(np.random.normal(loc=aov_base, scale=aov_base * 0.25), 2)
        gross_rev = max(12.0, gross_rev)
        
        # Discount (10% chance of promo discount)
        discount = round(gross_rev * np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.75, 0.12, 0.08, 0.05]), 2)
        net_rev = max(5.0, round(gross_rev - discount, 2))
        
        # Cost structure (COGS + OPEX + Commission)
        cogs_rate = float(row['COGSRate'])
        opex_rate = float(row['OPEXRate'])
        comm_rate = float(row['CommissionRate']) if chosen_channel in ['Uber Eats', 'DoorDash'] else 0.03
        
        cogs = net_rev * cogs_rate
        opex = net_rev * opex_rate
        comm = net_rev * comm_rate
        total_cost = round(cogs + opex + comm, 2)
        profit = round(net_rev - total_cost, 2)
        
        # Customer rating (higher for direct/in-store, slight noise)
        base_rating = 4.3 if chosen_channel in ['In-Store', 'Self-Delivery'] else 4.0
        rating = round(min(5.0, max(1.0, np.random.normal(loc=base_rating, scale=0.6))), 1)
        
        cust_type = np.random.choice(cust_types, p=cust_type_p)
        status = np.random.choice(statuses, p=status_p)
        
        orders.append({
            'Order_ID': f"ORD-{ord_date.strftime('%Y')}-{i+1:05d}",
            'Order_Date': ord_date.strftime('%Y-%m-%d'),
            'Order_Time': ord_time_str,
            'Restaurant_ID': int(row['RestaurantID']),
            'Restaurant_Name': str(row['RestaurantName']),
            'Restaurant_Segment': str(row['Segment']),
            'Cuisine_Type': str(row['CuisineType']),
            'Order_Channel': chosen_channel,
            'Subregion': str(row['Subregion']),
            'Customer_Type': cust_type,
            'Order_Status': status,
            'Quantity': quantity,
            'Gross_Revenue': gross_rev,
            'Discount': discount,
            'Net_Revenue': net_rev,
            'Cost': total_cost,
            'Profit': profit,
            'Customer_Rating': rating,
            'Market_Share': float(row['GrowthFactor']),
            'Aggregator_Name': agg_name
        })
        
    df_orders = pd.DataFrame(orders)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_orders.to_csv(output_csv_path, index=False)
    print(f"Successfully generated {len(df_orders)} order records at {output_csv_path}")
    return df_orders

if __name__ == '__main__':
    master_path = r"SkyCity Auckland Restaurants & Bars.csv"
    out_path = r"data/skycity_orders.csv"
    generate_orders_dataset(master_path, out_path)
