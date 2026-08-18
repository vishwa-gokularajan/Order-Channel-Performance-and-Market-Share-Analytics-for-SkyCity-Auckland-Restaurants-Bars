"""
SkyCity Auckland Analytics - Statistical Analysis & Hypothesis Testing Engine
Executes Chi-Square Tests of Independence, One-Way ANOVA, and Independent T-Tests.
"""

import pandas as pd
import numpy as np
from scipy import stats

def run_chi_square_test(df: pd.DataFrame, col1: str = 'Subregion', col2: str = 'Order_Channel', alpha: float = 0.05) -> dict:
    """
    Performs Chi-Square Test of Independence between two categorical variables.
    """
    if df.empty or col1 not in df.columns or col2 not in df.columns or len(df) < 5:
        return {
            'test_name': f'Chi-Square Test of Independence ({col1} vs {col2})',
            'null_hypothesis': f"{col1} and {col2} are independent.",
            'alt_hypothesis': f"{col1} and {col2} are significantly associated.",
            'chi2_statistic': 0.0,
            'dof': 0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': "Insufficient data to perform Chi-Square test.",
            'contingency_table': pd.DataFrame()
        }
    contingency_table = pd.crosstab(df[col1], df[col2])
    if contingency_table.empty or contingency_table.shape[0] < 1 or contingency_table.shape[1] < 1:
        return {
            'test_name': f'Chi-Square Test of Independence ({col1} vs {col2})',
            'null_hypothesis': f"{col1} and {col2} are independent.",
            'alt_hypothesis': f"{col1} and {col2} are significantly associated.",
            'chi2_statistic': 0.0,
            'dof': 0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': "Insufficient variation in data for Chi-Square test.",
            'contingency_table': contingency_table
        }
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    
    h0 = f"{col1} and {col2} are independent."
    h1 = f"{col1} and {col2} are significantly associated."
    reject_h0 = bool(p < alpha)
    
    conclusion = (
        f"Reject Null Hypothesis (p={p:.4e} < {alpha}). There is a statistically significant "
        f"relationship between {col1} and {col2}."
        if reject_h0 else
        f"Fail to Reject Null Hypothesis (p={p:.4f} >= {alpha}). No statistically significant "
        f"relationship found between {col1} and {col2}."
    )
    
    return {
        'test_name': f'Chi-Square Test of Independence ({col1} vs {col2})',
        'null_hypothesis': h0,
        'alt_hypothesis': h1,
        'chi2_statistic': float(chi2),
        'dof': int(dof),
        'p_value': float(p),
        'alpha': alpha,
        'reject_null': reject_h0,
        'conclusion': conclusion,
        'contingency_table': contingency_table
    }

def run_anova_test(df: pd.DataFrame, group_col: str = 'Order_Channel', val_col: str = 'Net_Revenue', alpha: float = 0.05) -> dict:
    """
    Performs One-Way ANOVA across categorical groups for a continuous metric.
    """
    if df.empty or group_col not in df.columns or val_col not in df.columns:
        return {
            'test_name': f'One-Way ANOVA ({val_col} across {group_col})',
            'null_hypothesis': f"Mean {val_col} is equal across all {group_col} groups.",
            'alt_hypothesis': f"At least one {group_col} group has a significantly different mean {val_col}.",
            'f_statistic': 0.0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': "Insufficient data for ANOVA test."
        }
    groups = [group[val_col].dropna().values for _, group in df.groupby(group_col) if len(group[val_col].dropna()) > 0]
    if len(groups) < 2:
        return {
            'test_name': f'One-Way ANOVA ({val_col} across {group_col})',
            'null_hypothesis': f"Mean {val_col} is equal across all {group_col} groups.",
            'alt_hypothesis': f"At least one {group_col} group has a significantly different mean {val_col}.",
            'f_statistic': 0.0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': f"Need at least 2 non-empty groups in {group_col} to perform ANOVA."
        }
    f_stat, p = stats.f_oneway(*groups)
    
    h0 = f"Mean {val_col} is equal across all {group_col} groups."
    h1 = f"At least one {group_col} group has a significantly different mean {val_col}."
    reject_h0 = bool(p < alpha)
    
    conclusion = (
        f"Reject Null Hypothesis (F={f_stat:.2f}, p={p:.4e} < {alpha}). Significant difference "
        f"exists in mean {val_col} across {group_col} categories."
        if reject_h0 else
        f"Fail to Reject Null Hypothesis (F={f_stat:.2f}, p={p:.4f} >= {alpha}). No significant difference "
        f"detected in mean {val_col} across {group_col} categories."
    )
    
    return {
        'test_name': f'One-Way ANOVA ({val_col} across {group_col})',
        'null_hypothesis': h0,
        'alt_hypothesis': h1,
        'f_statistic': float(f_stat),
        'p_value': float(p),
        'alpha': alpha,
        'reject_null': reject_h0,
        'conclusion': conclusion
    }

def run_ttest(df: pd.DataFrame, group_col: str = 'Channel_Category', val_col: str = 'Profit_Margin', alpha: float = 0.05) -> dict:
    """
    Performs Independent Two-Sample T-Test between two categories (e.g. Direct vs Aggregator).
    """
    if df.empty or group_col not in df.columns or val_col not in df.columns:
        return {
            'test_name': f'Independent T-Test ({val_col} across {group_col})',
            'null_hypothesis': f"Mean {val_col} is equal across groups.",
            'alt_hypothesis': f"Mean {val_col} differs significantly across groups.",
            't_statistic': 0.0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': "Insufficient data for T-Test.",
            'cat1_mean': 0.0,
            'cat2_mean': 0.0
        }
    categories = df[group_col].dropna().unique()
    if len(categories) < 2:
        return {
            'test_name': f'Independent T-Test ({val_col} across {group_col})',
            'null_hypothesis': f"Mean {val_col} is equal across groups.",
            'alt_hypothesis': f"Mean {val_col} differs significantly across groups.",
            't_statistic': 0.0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': f"Column {group_col} contains only {len(categories)} group(s). Minimum 2 required for T-Test.",
            'cat1_mean': 0.0,
            'cat2_mean': 0.0
        }
        
    cat1_vals = df[df[group_col] == categories[0]][val_col].dropna().values
    cat2_vals = df[df[group_col] == categories[1]][val_col].dropna().values
    
    if len(cat1_vals) == 0 or len(cat2_vals) == 0:
        return {
            'test_name': f'Independent T-Test ({val_col}: {categories[0]} vs {categories[1]})',
            'null_hypothesis': f"Mean {val_col} is equal for {categories[0]} and {categories[1]}.",
            'alt_hypothesis': f"Mean {val_col} differs significantly between {categories[0]} and {categories[1]}.",
            't_statistic': 0.0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_null': False,
            'conclusion': "One or both comparison categories have no valid numeric data.",
            'cat1_mean': float(np.mean(cat1_vals)) if len(cat1_vals) > 0 else 0.0,
            'cat2_mean': float(np.mean(cat2_vals)) if len(cat2_vals) > 0 else 0.0
        }

    t_stat, p = stats.ttest_ind(cat1_vals, cat2_vals, equal_var=False)
    
    h0 = f"Mean {val_col} is equal for {categories[0]} and {categories[1]}."
    h1 = f"Mean {val_col} differs significantly between {categories[0]} and {categories[1]}."
    reject_h0 = bool(p < alpha)
    
    conclusion = (
        f"Reject Null Hypothesis (t={t_stat:.2f}, p={p:.4e} < {alpha}). Significant difference "
        f"in {val_col} between {categories[0]} (mean: {np.mean(cat1_vals):.2f}) and {categories[1]} (mean: {np.mean(cat2_vals):.2f})."
        if reject_h0 else
        f"Fail to Reject Null Hypothesis (t={t_stat:.2f}, p={p:.4f} >= {alpha}). No significant difference "
        f"in {val_col} between {categories[0]} and {categories[1]}."
    )
    
    return {
        'test_name': f'Independent T-Test ({val_col}: {categories[0]} vs {categories[1]})',
        'null_hypothesis': h0,
        'alt_hypothesis': h1,
        't_statistic': float(t_stat),
        'p_value': float(p),
        'alpha': alpha,
        'reject_null': reject_h0,
        'conclusion': conclusion,
        'cat1_mean': float(np.mean(cat1_vals)),
        'cat2_mean': float(np.mean(cat2_vals))
    }
