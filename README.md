# Order Channel Performance and Market Share Analytics for SkyCity Auckland Restaurants & Bars

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, professional, internship-ready Data Analytics and Machine Learning project designed to analyze **order-channel performance, aggregator dependency, regional dominance, cuisine/segment mix, statistical hypotheses, time-series forecasting, and market share** for **SkyCity Auckland Restaurants & Bars**.

---

## 📌 Executive Summary

Restaurant operators and hospitality analysts at **SkyCity Auckland** face significant operational challenges regarding order channel mix:
* High dependence on third-party aggregators (**Uber Eats**, **DoorDash**) with commission rates reaching **28%–33%**, compressing net margins to **8%–12%**.
* Lack of visibility into regional ordering behavior across Auckland subregions (**CBD, North Shore, South Auckland, West Auckland**).
* Ambiguity surrounding which cuisines perform best through direct in-store dining vs. third-party delivery channels.

This project delivers a complete business intelligence solution—from raw transaction preparation to statistical validation, predictive time-series forecasting, and an interactive 6-page **Streamlit Web Application**.

---

## 📁 Project Structure

```
project/
├── data/
│   ├── SkyCity Auckland Restaurants & Bars.csv  # Provided master dataset (1,696 restaurant profiles)
│   └── skycity_orders.csv                       # Granular order transactional dataset (35,000 records)
├── notebooks/
│   └── analysis.ipynb                           # Comprehensive end-to-end analytical Jupyter notebook
├── src/
│   ├── __init__.py                              # Package marker
│   ├── data_generator.py                        # Synthetic transaction generator anchored to master CSV
│   ├── data_cleaning.py                         # Data cleaning, type conversion & feature engineering
│   ├── eda.py                                   # 20+ Visualization functions & KPI calculation engine
│   ├── stats_analysis.py                        # Chi-Square, One-Way ANOVA, and Independent T-Test engine
│   ├── forecasting.py                           # Time-series forecasting (SMA, Holt-Winters, ARIMA)
│   ├── ml_models.py                             # ML Regression models (Random Forest, Gradient Boosting)
│   ├── create_notebook.py                       # Script to build analysis.ipynb
│   └── utils.py                                 # Color palettes, currency formatters & UI tokens
├── app.py                                       # 6-Page Interactive Streamlit Web Application
├── requirements.txt                             # Python dependencies
├── README.md                                    # Comprehensive setup & project guide
├── Project_Report.md                            # Detailed 22-section internship report
└── presentation_content.md                      # 15-Slide internship viva presentation content
```

---

## 📊 Key Analytics Modules

### 1. Data Cleaning & Feature Engineering (`src/data_cleaning.py`)
- Missing value imputation & outlier detection using the IQR method.
- Derived temporal features: `Month`, `Quarter`, `Year`, `DayOfWeek`, `Hour`, `Is_Peak_Hour` (11-14 Lunch, 17-21 Dinner).
- Financial metrics: `Profit_Margin` ($\text{Profit} / \text{Net Revenue}$), `Average_Order_Value` (AOV), `Channel_Dependency_Pct` ($\text{Aggregator Revenue} / \text{Total Revenue}$).
- Classification of Aggregator Dependency (`Low`: <30%, `Medium`: 30-50%, `High`: >50%).

### 2. Exploratory Data Analysis & Visualizations (`src/eda.py`)
Calculates **14 Core KPIs** and renders **20 Standard EDA Charts**:
1. Total orders by channel
2. Revenue by channel
3. Profit by channel
4. Market share by restaurant
5. Market share by channel
6. Channel distribution by subregion
7. Channel distribution by cuisine
8. Channel mix by restaurant segment
9. Monthly order trend
10. Monthly revenue trend
11. Peak ordering hours
12. Day-of-week revenue performance
13. Restaurant-wise revenue ranking
14. Cuisine-wise revenue ranking
15. Aggregator dependency distribution
16. Direct vs aggregator performance comparison
17. Average Order Value by channel
18. Profit margin by channel
19. Customer rating by channel
20. Multi-feature correlation heatmap

### 3. Statistical Analysis (`src/stats_analysis.py`)
* **Chi-Square Test of Independence**: Confirms significant association between Subregion and Order Channel ($p < 0.001$).
* **One-Way ANOVA**: Validates significant differences in mean revenue across order channels ($p < 0.001$).
* **Independent T-Test**: Proves Direct channels yield significantly higher profit margins than Aggregators ($p < 0.001$).

### 4. Forecasting & Machine Learning (`src/forecasting.py` & `src/ml_models.py`)
* **Time-Series Forecasting**: Evaluates Moving Average, Exponential Smoothing, and ARIMA for 30/60/90-day horizons.
* **Supervised Machine Learning**: Compares Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors ($R^2$, MAE, RMSE).

---

## 🖥️ Streamlit Web Application Guide (`app.py`)

The application features a modern dark glassmorphism theme and 6 interactive pages:

1. **Page 1 – Executive Overview**: High-level KPI cards, revenue & order trends, channel market shares.
2. **Page 2 – Channel Performance**: Revenue, orders, profit, AOVs, profit margins, and Direct vs. Aggregator metrics.
3. **Page 3 – Regional Analysis**: Subregion filter, Subregion x Channel heatmap, dominant channel matrix.
4. **Page 4 – Cuisine & Restaurant Analysis**: Cuisine and segment cross-filtering, delivery vs. dine-in propensity.
5. **Page 5 – Forecasting & Machine Learning**: Interactive horizon slider, ARIMA confidence bounds, statistical tests, ML model comparisons.
6. **Page 6 – Business Insights & Recommendations**: 8 data-driven strategic recommendation cards.

---

## ⚙️ How to Run the Project

### 1. Prerequisites
Make sure Python 3.10+ is installed on your machine.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Granular Dataset (Optional)
```bash
python src/data_generator.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💡 Strategic Recommendations

1. **Mitigate Aggregator Commissions**: Negotiate tiered commission caps with Uber Eats/DoorDash and incentivize direct orders through exclusive direct web pricing.
2. **Launch SkyCity Digital Loyalty**: Implement a unified loyalty program to convert 3rd-party delivery users to SkyCity Direct Web.
3. **Regional Hub Optimization**: Establish dark kitchens in high-demand delivery zones (CBD, North Shore) to preserve fine-dining seating capacity.
4. **Peak Staffing Alignment**: Align kitchen dispatch staff with peak ordering hours (11:00–14:00 & 17:00–21:00).
