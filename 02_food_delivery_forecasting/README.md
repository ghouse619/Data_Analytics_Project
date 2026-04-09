# Project 2 — Food Delivery Operations & Demand Forecasting

**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · MySQL · Matplotlib  
**Dataset:** 50,000+ synthetic Swiggy/Zomato-style delivery orders

## Folder Structure
```
02_food_delivery_forecasting/
├── food_delivery_analysis.py      ← Main analysis script
├── data/
│   └── food_delivery_clean.csv    ← Cleaned delivery dataset
├── outputs/
│   ├── restaurant_scorecards.csv  ← Performance scores per restaurant
│   ├── hourly_demand.csv          ← Orders & avg value by hour
│   ├── cancellation_by_city.csv   ← Cancellation rates per city
│   └── analysis_report.txt        ← Full findings & recommendations
└── charts/
    ├── 01_demand_heatmap.png
    ├── 02_hourly_demand_curve.png
    ├── 03_dow_demand.png
    ├── 04_restaurant_scorecard.png
    └── 05_forecast_vs_actual.png
```

## Key Findings
- Peak demand window: 7–10 PM on weekdays
- Linear Regression model (R²=0.38) forecasts hourly zone-level demand
- 23 underperforming restaurants flagged for SLA review
- ETL pipeline resolved 12% data quality issues (nulls + duplicates)

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python food_delivery_analysis.py
```
