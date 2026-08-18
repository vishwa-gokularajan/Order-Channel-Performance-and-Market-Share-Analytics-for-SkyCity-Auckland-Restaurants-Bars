import json
import os

def generate_notebook(output_path: str = "notebooks/analysis.ipynb"):
    nb = {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '# Order Channel Performance and Market Share Analytics for SkyCity Auckland Restaurants & Bars\n',
                    '**Comprehensive End-to-End Analytics Notebook**\n',
                    'This notebook executes data preparation, feature engineering, exploratory data analysis, KPI calculation, statistical hypothesis testing, time-series forecasting, and machine learning.'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'import os, sys\n',
                    'sys.path.append("..")\n',
                    'import pandas as pd\n',
                    'import numpy as np\n',
                    'import plotly.express as px\n',
                    'from src.data_cleaning import clean_and_prepare_data\n',
                    'from src.eda import calculate_kpis, get_regional_dominance_matrix\n',
                    'from src.stats_analysis import run_chi_square_test, run_anova_test, run_ttest\n',
                    'from src.forecasting import prepare_daily_timeseries, forecast_arima\n',
                    'from src.ml_models import train_and_evaluate_ml_models\n',
                    'print("Libraries successfully imported!")'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 1. Data Loading & Preprocessing\n',
                    'We load the granular transactional order dataset and run the data cleaning pipeline.'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'df_raw = pd.read_csv("data/skycity_orders.csv")\n',
                    'df = clean_and_prepare_data(df_raw)\n',
                    'print("Dataset Shape:", df.shape)\n',
                    'df.head(3)'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 2. Key Performance Indicators (KPIs)\n',
                    'Calculating core performance metrics for SkyCity Auckland.'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'kpis = calculate_kpis(df)\n',
                    'for k, v in kpis.items():\n',
                    '    print(f"{k}: {v}")'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 3. Regional Channel Dominance Matrix'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'dom_df = get_regional_dominance_matrix(df)\n',
                    'dom_df'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 4. Statistical Hypothesis Testing\n',
                    'Testing relationship between Subregion & Channel (Chi2), ANOVA across Channels, and T-test for Direct vs Aggregators.'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'print("Chi-Square Test:", run_chi_square_test(df))\n',
                    'print("ANOVA Test:", run_anova_test(df))\n',
                    'print("T-Test:", run_ttest(df))'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 5. Time-Series Forecasting (ARIMA)'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'daily_ts = prepare_daily_timeseries(df, metric="Net_Revenue")["Net_Revenue"]\n',
                    'fc_vals, conf_int, metrics = forecast_arima(daily_ts, horizon_days=30)\n',
                    'print("ARIMA Forecast Metrics:", metrics)'
                ]
            },
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '## 6. Machine Learning Regression'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'results, summary_df, imp_df, best_name, best_pipe = train_and_evaluate_ml_models(df, target_col="Net_Revenue")\n',
                    'summary_df'
                ]
            }
        ],
        'metadata': {
            'language_info': {'name': 'python'}
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    generate_notebook("notebooks/analysis.ipynb")
