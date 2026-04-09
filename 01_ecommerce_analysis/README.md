# Project 1 — E-Commerce Sales & Customer Segmentation Analysis

**Tech Stack:** Python · Pandas · Matplotlib · Seaborn · SQL · Tableau  
**Dataset:** 100,000+ synthetic e-commerce transactions (Olist-style)

## Folder Structure
```
01_ecommerce_analysis/
├── ecommerce_analysis.py      ← Main analysis script
├── data/
│   └── ecommerce_clean.csv    ← Cleaned transaction dataset
├── outputs/
│   ├── rfm_segments.csv       ← RFM scores per customer
│   ├── segment_summary.csv    ← Revenue & customer % by segment
│   ├── churn_by_channel.csv   ← Churn rate by acquisition channel
│   └── analysis_report.txt   ← Full findings & recommendations
└── charts/
    ├── 01_monthly_gmv_trend.png
    ├── 02_revenue_by_category.png
    ├── 03_rfm_segmentation.png
    ├── 04_churn_by_channel.png
    └── 05_city_category_heatmap.png
```

## Key Findings
- Top 15% of customers (Champions) drive ~33% of total revenue
- Voucher-acquired customers churn at a higher rate than organic users
- Electronics & Fashion are the top 2 revenue categories
- Mumbai & Delhi drive the largest share of GMV

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python ecommerce_analysis.py
```
