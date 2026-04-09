# Project 4 — HR Employee Attrition & People Analytics

**Tech Stack:** Python · Pandas · Scikit-learn · Logistic Regression · Tableau  
**Dataset:** 1,470 synthetic IBM HR-style employee records (35 variables)

## Folder Structure
```
04_hr_attrition_analysis/
├── hr_attrition_analysis.py       ← Main analysis script
├── data/
│   └── hr_data_clean.csv          ← Cleaned employee dataset
├── outputs/
│   ├── feature_importance.csv     ← Feature importance ranking
│   ├── employee_risk_scores.csv   ← Per-employee attrition risk scores
│   ├── attrition_by_dept.csv      ← Attrition rates by department
│   └── analysis_report.txt        ← Full findings & recommendations
└── charts/
    ├── 01_attrition_by_dept.png
    ├── 02_feature_importance.png
    ├── 03_roc_curves.png
    ├── 04_dept_income_heatmap.png
    └── 05_satisfaction_vs_attrition.png
```

## Key Findings
- High-risk group (overtime + low satisfaction + <2 yrs tenure) = 3.0× baseline attrition
- Random Forest: 87.1% accuracy, AUC: 0.85
- Logistic Regression AUC: 0.86 (best performer)
- Monthly income is the top predictor of attrition

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
python hr_attrition_analysis.py
```
