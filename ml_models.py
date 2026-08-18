"""
SkyCity Auckland Analytics - Machine Learning Predictive Modeling Engine
Trains Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors.
"""

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_evaluate_ml_models(df: pd.DataFrame, target_col: str = 'Net_Revenue') -> tuple[dict, pd.DataFrame, pd.DataFrame, str, Pipeline]:
    """
    Trains multiple ML regression models to predict revenue or order metrics.
    """
    if df.empty or len(df) < 10 or target_col not in df.columns:
        empty_res = {'R2 Score': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
        empty_df = pd.DataFrame([{'Model': 'N/A', 'R2 Score': 0.0, 'MAE': 0.0, 'RMSE': 0.0}])
        return {'N/A': empty_res}, empty_df, pd.DataFrame(), 'N/A', None

    df_ml = df.dropna(subset=[target_col]).copy()
    
    feature_cols_cat = ['Restaurant_Segment', 'Cuisine_Type', 'Order_Channel', 'Subregion', 'Customer_Type', 'Is_Peak_Hour', 'Day_Name']
    feature_cols_num = ['Quantity', 'Discount', 'Customer_Rating', 'Hour', 'Market_Share']
    
    # Filter available columns
    feature_cols_cat = [c for c in feature_cols_cat if c in df_ml.columns]
    feature_cols_num = [c for c in feature_cols_num if c in df_ml.columns]
    
    if not feature_cols_cat and not feature_cols_num:
        empty_res = {'R2 Score': 0.0, 'MAE': 0.0, 'RMSE': 0.0}
        empty_df = pd.DataFrame([{'Model': 'N/A', 'R2 Score': 0.0, 'MAE': 0.0, 'RMSE': 0.0}])
        return {'N/A': empty_res}, empty_df, pd.DataFrame(), 'N/A', None

    X = df_ml[feature_cols_cat + feature_cols_num]
    y = df_ml[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), feature_cols_cat),
            ('num', 'passthrough', feature_cols_num)
        ]
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -1e9
    best_pipeline = None
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'R2 Score': round(float(r2), 4),
            'MAE': round(float(mae), 2),
            'RMSE': round(float(rmse), 2)
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    summary_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    
    # Feature Importance for best model (Random Forest / Gradient Boosting)
    feature_names = list(best_pipeline.named_steps['prep'].get_feature_names_out())
    if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
        importances = best_pipeline.named_steps['model'].feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df['Feature'] = imp_df['Feature'].str.replace('cat__', '').str.replace('num__', '')
        imp_df = imp_df.sort_values('Importance', ascending=False).head(15)
    else:
        imp_df = pd.DataFrame()
        
    return results, summary_df, imp_df, best_model_name, best_pipeline

def plot_feature_importances(imp_df: pd.DataFrame):
    """Plotly horizontal bar chart of top feature importances."""
    if imp_df.empty:
        return None
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                 title='Top 15 Machine Learning Feature Importances',
                 color='Importance', color_continuous_scale='Teal')
    fig.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'})
    return fig
