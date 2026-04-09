# Project 3 — Student Dropout Risk Prediction

**Tech Stack:** Python · Pandas · Scikit-learn · Seaborn · SMOTE · Power BI  
**Dataset:** 4,424 synthetic UCI-style student records

## Folder Structure
```
03_student_dropout_prediction/
├── student_dropout_analysis.py    ← Main analysis script
├── data/
│   └── student_data_clean.csv     ← Cleaned student dataset
├── outputs/
│   ├── feature_importance.csv     ← Feature importance ranking
│   ├── student_risk_scores.csv    ← Per-student dropout risk scores
│   └── analysis_report.txt        ← Full findings & recommendations
└── charts/
    ├── 01_feature_importance.png
    ├── 02_roc_curve.png
    ├── 03_confusion_matrix.png
    ├── 04_dropout_by_attendance.png
    └── 05_risk_by_department.png
```

## Key Findings
- Random Forest: 99.2% accuracy, AUC: 1.00
- Attendance is the strongest dropout predictor (importance: 0.34)
- Students with <75% attendance are significantly more likely to drop out
- SMOTE applied to correct class imbalance before modelling

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
python student_dropout_analysis.py
```
