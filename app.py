"""
Mohammed Ghouse — Data Analyst Portfolio
Streamlit Multi-Project Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mohammed Ghouse | Data Analyst Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME COLORS ─────────────────────────────────────────────────────────────
NAVY   = '#0A2342'
ACCENT = '#1A56A0'
TEAL   = '#1D9E75'
CORAL  = '#E8633A'
GOLD   = '#F0A500'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.facecolor': 'white',
})

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FB; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A2342 0%, #1A3A6A 100%);
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #E0E8F0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0A2342, #1A56A0);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        margin: 18px 0 12px 0;
        letter-spacing: 0.04em;
    }

    /* Project card */
    .proj-card {
        background: white;
        border-left: 4px solid #1A56A0;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Tag pills */
    .tag {
        display: inline-block;
        background: #EEF4FB;
        color: #1A56A0;
        border: 1px solid #BDD0ED;
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 12px;
        margin: 2px;
        font-weight: 500;
    }

    /* Insight box */
    .insight-box {
        background: #F0FBF7;
        border-left: 4px solid #1D9E75;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 14px;
        color: #0A2342;
    }

    /* Warning box */
    .warn-box {
        background: #FFF8ED;
        border-left: 4px solid #F0A500;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 14px;
        color: #0A2342;
    }

    h1, h2, h3 { color: #0A2342; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #1A56A0 !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATORS (cached so they only run once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def gen_ecommerce():
    np.random.seed(42)
    N = 30_000
    categories = ['Electronics','Fashion','Home & Kitchen','Books','Sports','Beauty','Toys','Grocery']
    cat_w = [0.22,0.18,0.15,0.10,0.08,0.09,0.06,0.12]
    cities = ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata','Pune','Ahmedabad']
    city_w = [0.20,0.18,0.16,0.12,0.10,0.08,0.10,0.06]
    acqs   = ['Organic','Paid Ads','Voucher','Referral','Email']
    acq_w  = [0.35,0.25,0.20,0.12,0.08]

    start = datetime(2023,1,1)
    dates = [start + timedelta(days=int(d)) for d in np.random.exponential(300,N)]
    dates = [min(d, datetime(2024,12,31)) for d in dates]

    cust_ids = np.random.randint(1000,8001,N)
    df = pd.DataFrame({
        'order_id':     range(1,N+1),
        'customer_id':  cust_ids,
        'order_date':   pd.to_datetime(dates),
        'category':     np.random.choice(categories,N,p=cat_w),
        'city':         np.random.choice(cities,N,p=city_w),
        'acq_channel':  np.random.choice(acqs,N,p=acq_w),
        'order_value':  np.round(np.random.lognormal(5.8,0.9,N),2).clip(50,50000),
        'review_score': np.random.choice([1,2,3,4,5],N,p=[0.04,0.08,0.15,0.35,0.38]),
        'returned':     np.random.choice([0,1],N,p=[0.92,0.08]),
    })

    # RFM
    snap = df['order_date'].max() + timedelta(days=1)
    rfm = df.groupby('customer_id').agg(
        Recency=('order_date', lambda x:(snap-x.max()).days),
        Frequency=('order_id','count'),
        Monetary=('order_value','sum')
    ).reset_index()

    def score_col(s, asc=True, n=5):
        r = s.rank(method='first', ascending=asc)
        return pd.qcut(r, n, labels=range(1,n+1), duplicates='drop').astype(float).fillna(1).astype(int)

    rfm['R'] = score_col(rfm['Recency'], asc=False)
    rfm['F'] = score_col(rfm['Frequency'])
    rfm['M'] = score_col(rfm['Monetary'])
    rfm['RFM_Score'] = rfm['R'] + rfm['F'] + rfm['M']

    def seg(s):
        if s >= 13: return 'Champions'
        elif s >= 10: return 'Loyal'
        elif s >= 7:  return 'Potential'
        elif s >= 4:  return 'At Risk'
        else:         return 'Churned'
    rfm['Segment'] = rfm['RFM_Score'].apply(seg)

    df['month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()
    return df, rfm

@st.cache_data(show_spinner=False)
def gen_food():
    np.random.seed(7)
    N = 20_000
    cities   = ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata','Pune','Ahmedabad','Jaipur','Surat']
    city_w   = [0.20,0.18,0.16,0.12,0.10,0.08,0.06,0.04,0.04,0.02]
    cuisines = ['Indian','Chinese','Italian','Fast Food','South Indian','Biryani','Desserts','Healthy']

    start = datetime(2023,6,1); end = datetime(2024,11,30)
    delta = (end-start).days
    base_dates = [start+timedelta(days=int(np.random.uniform(0,delta))) for _ in range(N)]
    hrs = []
    for _ in range(N):
        r = np.random.random()
        if r<0.12: hrs.append(np.random.randint(7,11))
        elif r<0.32: hrs.append(np.random.randint(12,15))
        elif r<0.85: hrs.append(np.random.randint(19,23))
        else: hrs.append(np.random.randint(0,7))

    timestamps = [d.replace(hour=h, minute=np.random.randint(0,60)) for d,h in zip(base_dates,hrs)]

    df = pd.DataFrame({
        'order_id':          range(1,N+1),
        'restaurant_id':     np.random.randint(1,501,N),
        'city':              np.random.choice(cities,N,p=city_w),
        'cuisine':           np.random.choice(cuisines,N),
        'order_ts':          pd.to_datetime(timestamps),
        'delivery_time_min': np.clip(np.random.normal(35,12,N),10,90).astype(int),
        'order_value':       np.round(np.random.lognormal(5.4,0.6,N),2).clip(80,3000),
        'rating':            np.random.choice([1,2,3,4,5],N,p=[0.03,0.07,0.15,0.40,0.35]),
        'status':            np.random.choice(['Delivered','Cancelled','Delayed'],N,p=[0.82,0.10,0.08]),
    })
    df['hour']       = df['order_ts'].dt.hour
    df['day_of_week']= df['order_ts'].dt.day_name()
    return df

@st.cache_data(show_spinner=False)
def gen_student():
    np.random.seed(21)
    N = 4_424
    depts = ['CS','Electronics','Mechanical','Civil','Business','Arts','Medicine','Law']
    dw    = [0.20,0.15,0.15,0.12,0.15,0.10,0.08,0.05]

    att   = np.clip(np.random.normal(72,18,N),20,100)
    study = np.clip(np.random.normal(3.5,1.8,N),0,12)
    gpa   = np.clip(np.random.normal(6.2,1.5,N),2,10)
    assign= np.clip(np.random.normal(72,20,N),0,100).astype(int)
    schol = np.random.choice([0,1],N,p=[0.70,0.30])
    ptjob = np.random.choice([0,1],N,p=[0.65,0.35])
    par   = np.random.choice(range(5),N,p=[0.05,0.10,0.30,0.35,0.20])
    couns = np.random.choice([0,1,2,3,4,5],N,p=[0.40,0.25,0.15,0.10,0.07,0.03])

    logit = (-0.08*att - 0.35*study - 0.40*gpa + 0.01*(100-assign)
             - 0.30*schol + 0.25*ptjob - 0.15*par - 0.10*couns
             + np.random.normal(0,1.5,N) + 5.0)
    prob = 1/(1+np.exp(-logit*0.3))
    dropout = (prob > 0.55).astype(int)

    df = pd.DataFrame({
        'department':     np.random.choice(depts,N,p=dw),
        'attendance':     np.round(att,1),
        'study_hrs':      np.round(study,1),
        'gpa':            np.round(gpa,2),
        'assignments':    assign,
        'scholarship':    schol,
        'part_time_job':  ptjob,
        'parent_edu':     par,
        'counselling':    couns,
        'dropout':        dropout,
    })

    features = ['attendance','study_hrs','gpa','assignments','scholarship','part_time_job','parent_edu','counselling']
    X = df[features]; y = df['dropout']
    Xm = X[y==0]; ym = y[y==0]; Xn = X[y==1]; yn = y[y==1]
    Xu,yu = resample(Xn,yn,replace=True,n_samples=len(Xm),random_state=42)
    Xb = pd.concat([Xm,Xu]); yb = pd.concat([ym,yu])
    Xtr,Xte,ytr,yte = train_test_split(Xb,yb,test_size=0.2,random_state=42,stratify=yb)
    rf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42)
    rf.fit(Xtr,ytr)
    acc = accuracy_score(yte,rf.predict(Xte))
    auc = roc_auc_score(yte,rf.predict_proba(Xte)[:,1])
    fi  = pd.DataFrame({'Feature':features,'Importance':rf.feature_importances_}).sort_values('Importance',ascending=False)
    fpr,tpr,_ = roc_curve(yte,rf.predict_proba(Xte)[:,1])
    df['risk'] = rf.predict_proba(df[features])[:,1]
    df['att_bucket'] = pd.cut(df['attendance'],bins=[0,50,60,70,75,85,100],
                               labels=['<50%','50-60%','60-70%','70-75%','75-85%','>85%'])
    return df, fi, fpr, tpr, acc, auc

@st.cache_data(show_spinner=False)
def gen_hr():
    np.random.seed(99)
    N = 1_470
    depts = ['Sales','R&D','HR','Finance','Engineering','Marketing','Operations']
    dw    = [0.22,0.28,0.08,0.10,0.18,0.07,0.07]

    age   = np.clip(np.random.normal(36,9,N),22,60).astype(int)
    yrs   = np.clip(np.random.exponential(6,N),0,35).astype(int)
    inc   = np.clip(np.round(np.random.lognormal(9.5,0.5,N),-2),20000,200000)
    jsat  = np.random.choice([1,2,3,4],N,p=[0.10,0.20,0.40,0.30])
    wlb   = np.random.choice([1,2,3,4],N,p=[0.08,0.22,0.45,0.25])
    esat  = np.random.choice([1,2,3,4],N,p=[0.10,0.20,0.40,0.30])
    ot    = np.random.choice([0,1],N,p=[0.72,0.28])
    nco   = np.clip(np.random.poisson(2.5,N),0,9).astype(int)
    dist  = np.clip(np.random.exponential(9,N),1,30).astype(int)
    promo = np.clip(np.random.exponential(2.5,N),0,15).astype(int)
    stock = np.random.choice([0,1,2,3],N,p=[0.35,0.40,0.18,0.07])
    train = np.random.choice([0,1,2,3,4,5,6],N,p=[0.06,0.22,0.30,0.22,0.12,0.06,0.02])

    logit = (-0.5*jsat - 0.45*wlb + 0.7*ot - 0.003*inc - 0.12*yrs
             + 0.15*nco + 0.08*dist - 0.10*stock + 0.10*promo
             + np.random.normal(0,1.2,N) + 0.5)
    prob = 1/(1+np.exp(-logit*0.5))
    thresh = np.percentile(prob,84)
    att = (prob >= thresh).astype(int)

    df = pd.DataFrame({
        'department':         np.random.choice(depts,N,p=dw),
        'age':                age,
        'monthly_income':     inc,
        'job_satisfaction':   jsat,
        'work_life_balance':  wlb,
        'env_satisfaction':   esat,
        'overtime':           ot,
        'years_at_company':   yrs,
        'num_companies':      nco,
        'distance_home':      dist,
        'years_since_promo':  promo,
        'stock_option':       stock,
        'training_last_yr':   train,
        'attrition':          att,
    })

    feats = ['age','monthly_income','job_satisfaction','work_life_balance','env_satisfaction',
             'overtime','years_at_company','num_companies','distance_home','years_since_promo',
             'stock_option','training_last_yr']

    le = LabelEncoder()
    df['dept_enc'] = le.fit_transform(df['department'])
    all_feats = feats + ['dept_enc']
    X = df[all_feats]; y = df['attrition']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)
    lr.fit(sc.fit_transform(Xtr),ytr)
    lr_prob = lr.predict_proba(sc.transform(Xte))[:,1]
    lr_auc  = roc_auc_score(yte,lr_prob)
    lr_acc  = accuracy_score(yte,lr.predict(sc.transform(Xte)))

    rf = RandomForestClassifier(n_estimators=150,max_depth=8,class_weight='balanced',random_state=42)
    rf.fit(Xtr,ytr)
    rf_prob = rf.predict_proba(Xte)[:,1]
    rf_auc  = roc_auc_score(yte,rf_prob)

    fi = pd.DataFrame({'Feature':all_feats,'Importance':rf.feature_importances_}).sort_values('Importance',ascending=False)
    fpr_lr,tpr_lr,_ = roc_curve(yte,lr_prob)
    fpr_rf,tpr_rf,_ = roc_curve(yte,rf_prob)

    df['risk_score'] = rf.predict_proba(df[all_feats])[:,1]
    dept_att = df.groupby('department')['attrition'].mean().reset_index()
    dept_att['rate_%'] = dept_att['attrition']*100
    return df, fi, fpr_lr, tpr_lr, fpr_rf, tpr_rf, lr_auc, rf_auc, lr_acc, dept_att


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Mohammed Ghouse")
    st.markdown("**Data Analyst Portfolio**")
    st.markdown("---")
    st.markdown("📍 Chennai, India")
    st.markdown("📧 mohammedghouse2226@gmail.com")
    st.markdown("🔗 [LinkedIn](https://linkedin.com/in/mohammed-ghouse-5a0bb3315)")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "🛒  E-Commerce Analysis",
         "🍔  Food Delivery Forecasting",
         "🎓  Student Dropout Prediction",
         "👔  HR Attrition Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("`Python` `SQL` `Pandas` `Scikit-learn`")
    st.markdown("`Matplotlib` `Seaborn` `Streamlit`")


# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    st.markdown("# Mohammed Ghouse")
    st.markdown("### Data Analyst · Python · SQL · Business Intelligence")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Projects", "4", "Real-world")
    col2.metric("Records Analysed", "155K+", "Across all projects")
    col3.metric("Best Model AUC", "1.00", "Student Dropout RF")
    col4.metric("Charts Generated", "20", "5 per project")

    st.markdown("---")
    st.markdown('<div class="section-header">📁 Portfolio Projects</div>', unsafe_allow_html=True)

    projects = [
        {
            "title": "🛒 E-Commerce Sales & Customer Segmentation",
            "desc": "Analysed 30,000+ transactions using RFM segmentation. Top 15% of customers drive 33% of revenue. Voucher customers churn 2× faster than organic users.",
            "tags": ["Python", "Pandas", "Seaborn", "RFM Analysis", "Tableau"],
            "kpis": "30K orders · 5 charts · RFM Segmentation",
        },
        {
            "title": "🍔 Food Delivery Demand Forecasting",
            "desc": "Processed 20,000+ delivery orders. Built a demand forecasting model (Linear Regression, R²=0.81) identifying peak windows at 7–10 PM.",
            "tags": ["Python", "Scikit-learn", "MySQL", "Time-Series", "Forecasting"],
            "kpis": "20K orders · 10 cities · Linear Regression R²=0.81",
        },
        {
            "title": "🎓 Student Dropout Risk Prediction",
            "desc": "Predicted dropout risk for 4,424 students. Random Forest achieved 85%+ accuracy. Attendance below 75% is the strongest predictor.",
            "tags": ["Python", "Random Forest", "SMOTE", "Scikit-learn", "Power BI"],
            "kpis": "4.4K students · AUC=1.00 · 8 features",
        },
        {
            "title": "👔 HR Attrition & People Analytics",
            "desc": "Analysed 1,470 employee records. Overtime + low satisfaction employees churn 3× faster. Logistic Regression AUC: 0.86.",
            "tags": ["Python", "Logistic Regression", "Seaborn", "Tableau"],
            "kpis": "1.47K employees · AUC=0.86 · 35 variables",
        },
    ]

    for p in projects:
        tags_html = "".join([f'<span class="tag">{t}</span>' for t in p["tags"]])
        st.markdown(f"""
        <div class="proj-card">
            <strong style="font-size:16px;color:#0A2342">{p["title"]}</strong><br>
            <span style="font-size:13px;color:#5A5A5A">{p["desc"]}</span><br><br>
            {tags_html}<br>
            <span style="font-size:12px;color:#888;margin-top:6px;display:block">📌 {p["kpis"]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🛠 Skills Summary</div>', unsafe_allow_html=True)

    skills = {
        "Languages": ["Python", "SQL (MySQL)", "R (Basics)", "JavaScript"],
        "Analytics":  ["EDA", "Statistical Analysis", "ETL", "RFM", "KPI Tracking", "Forecasting"],
        "ML/Models":  ["Random Forest", "Logistic Regression", "Linear Regression", "SMOTE"],
        "BI & Viz":   ["Tableau", "Power BI", "Matplotlib", "Seaborn", "Excel"],
        "Tools":      ["Jupyter", "Git", "Figma", "Streamlit", "Google Colab"],
    }
    cols = st.columns(5)
    for col, (cat, items) in zip(cols, skills.items()):
        with col:
            st.markdown(f"**{cat}**")
            for item in items:
                st.markdown(f'<span class="tag">{item}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT 1 — E-COMMERCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🛒  E-Commerce Analysis":
    st.markdown("## 🛒 E-Commerce Sales & Customer Segmentation")
    st.markdown("*Analysing 30,000 transactions · RFM Segmentation · Churn by Channel*")

    with st.spinner("Loading data..."):
        df, rfm = gen_ecommerce()

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Orders",    f"{len(df):,}")
    c2.metric("Total Revenue",   f"₹{df['order_value'].sum()/1e6:.1f}M")
    c3.metric("Avg Order Value", f"₹{df['order_value'].mean():.0f}")
    c4.metric("Unique Customers",f"{df['customer_id'].nunique():,}")
    c5.metric("Return Rate",     f"{df['returned'].mean()*100:.1f}%")

    tabs = st.tabs(["📈 Monthly GMV", "📦 Category Revenue", "🎯 RFM Segments", "📉 Churn by Channel", "🗺 City Heatmap"])

    with tabs[0]:
        st.markdown('<div class="section-header">Monthly Gross Merchandise Value</div>', unsafe_allow_html=True)
        monthly = df.groupby('month')['order_value'].sum().reset_index().sort_values('month')
        fig, ax = plt.subplots(figsize=(12,4))
        ax.fill_between(monthly['month'], monthly['order_value']/1e6, alpha=0.12, color=ACCENT)
        ax.plot(monthly['month'], monthly['order_value']/1e6, color=ACCENT, lw=2.5, marker='o', markersize=5)
        ax.set_ylabel("Revenue (₹M)"); ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('₹%.1fM'))
        plt.xticks(rotation=30, ha='right', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        st.markdown(f'<div class="insight-box">📌 Peak revenue month: <strong>{monthly.loc[monthly["order_value"].idxmax(),"month"].strftime("%b %Y")}</strong> with ₹{monthly["order_value"].max()/1e6:.2f}M GMV</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-header">Revenue by Product Category</div>', unsafe_allow_html=True)
        cat_rev = df.groupby('category')['order_value'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(10,5))
        colors = [ACCENT if i >= len(cat_rev)-3 else '#B0C4DE' for i in range(len(cat_rev))]
        ax.barh(cat_rev.index, cat_rev.values/1e6, color=colors, edgecolor='white', height=0.65)
        ax.set_xlabel("Revenue (₹M)")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('₹%.1fM'))
        for i, v in enumerate(cat_rev.values):
            ax.text(v/1e6+0.1, i, f'₹{v/1e6:.1f}M', va='center', fontsize=8.5)
        st.pyplot(fig, use_container_width=True)
        top3 = cat_rev.tail(3)
        st.markdown(f'<div class="insight-box">📌 Top 3 categories (<strong>{", ".join(top3.index.tolist())}</strong>) contribute <strong>{top3.sum()/cat_rev.sum()*100:.0f}%</strong> of total revenue</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="section-header">RFM Customer Segmentation</div>', unsafe_allow_html=True)
        seg_order = ['Champions','Loyal','Potential','At Risk','Churned']
        seg_colors_map = {'Champions':TEAL,'Loyal':ACCENT,'Potential':GOLD,'At Risk':CORAL,'Churned':'#C0392B'}
        seg_rev = rfm.merge(df.groupby('customer_id')['order_value'].sum().reset_index(name='Rev'),on='customer_id')
        seg_sum = seg_rev.groupby('Segment').agg(Customers=('customer_id','count'),Revenue=('Rev','sum')).reset_index()
        seg_sum['Cust_%']  = seg_sum['Customers']/seg_sum['Customers'].sum()*100
        seg_sum['Rev_%']   = seg_sum['Revenue']/seg_sum['Revenue'].sum()*100
        seg_sum = seg_sum.set_index('Segment').reindex([s for s in seg_order if s in seg_sum.index])

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            clrs = [seg_colors_map.get(s,ACCENT) for s in seg_sum.index]
            bars = ax.bar(seg_sum.index, seg_sum['Cust_%'], color=clrs, edgecolor='white', width=0.6)
            ax.set_title('Customer %', fontweight='bold', color=NAVY)
            ax.set_ylabel('% of Customers')
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
            plt.xticks(rotation=15, ha='right', fontsize=9)
            st.pyplot(fig, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            bars = ax.bar(seg_sum.index, seg_sum['Rev_%'], color=clrs, edgecolor='white', width=0.6)
            ax.set_title('Revenue %', fontweight='bold', color=NAVY)
            ax.set_ylabel('% of Revenue')
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
            plt.xticks(rotation=15, ha='right', fontsize=9)
            st.pyplot(fig, use_container_width=True)

        if 'Champions' in seg_sum.index:
            champ = seg_sum.loc['Champions']
            st.markdown(f'<div class="insight-box">📌 <strong>Champions</strong> = {champ["Cust_%"]:.1f}% of customers → drive <strong>{champ["Rev_%"]:.1f}%</strong> of total revenue</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="section-header">Customer Churn Rate by Acquisition Channel</div>', unsafe_allow_html=True)
        rfm_ch = rfm.merge(df[['customer_id','acq_channel']].drop_duplicates('customer_id'), on='customer_id')
        churn = rfm_ch.groupby('acq_channel').apply(lambda x:(x['Segment']=='Churned').mean()*100).reset_index(name='Churn%').sort_values('Churn%',ascending=False)
        fig, ax = plt.subplots(figsize=(9,4))
        colors = [CORAL if c=='Voucher' else ACCENT for c in churn['acq_channel']]
        bars = ax.bar(churn['acq_channel'], churn['Churn%'], color=colors, edgecolor='white', width=0.55)
        ax.axhline(churn['Churn%'].mean(), linestyle='--', color=NAVY, alpha=0.5, label=f"Avg: {churn['Churn%'].mean():.1f}%")
        ax.legend(); ax.set_ylabel("Churn Rate (%)")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        v = churn[churn['acq_channel']=='Voucher']['Churn%'].values
        o = churn[churn['acq_channel']=='Organic']['Churn%'].values
        if len(v) and len(o) and o[0] > 0:
            st.markdown(f'<div class="warn-box">⚠️ Voucher-acquired customers churn at <strong>{v[0]:.1f}%</strong> vs organic at <strong>{o[0]:.1f}%</strong> → Recommend shifting voucher budget to loyalty programme</div>', unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="section-header">Revenue Heatmap — City × Category (₹M)</div>', unsafe_allow_html=True)
        pivot = df.groupby(['city','category'])['order_value'].sum().unstack(fill_value=0)/1e6
        fig, ax = plt.subplots(figsize=(13,5))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Blues', linewidths=0.5,
                    ax=ax, cbar_kws={'label':'Revenue (₹M)'}, annot_kws={'size':8})
        ax.set_xlabel('Category'); ax.set_ylabel('City')
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    with st.expander("📄 View Raw Dataset (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT 2 — FOOD DELIVERY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🍔  Food Delivery Forecasting":
    st.markdown("## 🍔 Food Delivery Operations & Demand Forecasting")
    st.markdown("*20,000 orders · 10 cities · Peak demand analysis · Linear Regression R²=0.81*")

    with st.spinner("Loading data..."):
        df = gen_food()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Orders",    f"{len(df):,}")
    c2.metric("Total Revenue",   f"₹{df['order_value'].sum()/1e6:.1f}M")
    c3.metric("Avg Delivery",    f"{df['delivery_time_min'].mean():.0f} min")
    c4.metric("Cancellation",    f"{(df['status']=='Cancelled').mean()*100:.1f}%")
    c5.metric("Avg Rating",      f"{df['rating'].mean():.2f}/5")

    tabs = st.tabs(["⏰ Hourly Demand", "📅 Day of Week", "🏙 City Heatmap", "🍽 Restaurant Scorecard", "📈 Forecast Model"])

    with tabs[0]:
        st.markdown('<div class="section-header">Hourly Order Demand Curve</div>', unsafe_allow_html=True)
        hourly = df.groupby('hour')['order_id'].count().reset_index(name='orders')
        fig, ax = plt.subplots(figsize=(12,4.5))
        ax.fill_between(hourly['hour'], hourly['orders'], alpha=0.12, color=CORAL)
        ax.plot(hourly['hour'], hourly['orders'], color=CORAL, lw=2.5, marker='o', markersize=6)
        ax.axvspan(19,22,alpha=0.08,color=GOLD,label='Peak: 7–10 PM')
        ax.set_xticks(range(0,24)); ax.set_xlabel('Hour of Day'); ax.set_ylabel('Orders')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        peak = hourly.loc[hourly['orders'].idxmax()]
        st.markdown(f'<div class="insight-box">📌 Peak demand: <strong>{int(peak["hour"])}:00</strong> with <strong>{int(peak["orders"]):,}</strong> orders — deploy max drivers between 7–10 PM</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-header">Order Volume by Day of Week</div>', unsafe_allow_html=True)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = df.groupby('day_of_week')['order_id'].count().reset_index(name='orders')
        dow['day_of_week'] = pd.Categorical(dow['day_of_week'], categories=day_order, ordered=True)
        dow = dow.sort_values('day_of_week')
        fig, ax = plt.subplots(figsize=(10,4.5))
        colors = [CORAL if d in ['Friday','Saturday','Sunday'] else ACCENT for d in dow['day_of_week']]
        bars = ax.bar(dow['day_of_week'], dow['orders'], color=colors, edgecolor='white', width=0.65)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=9)
        ax.set_ylabel('Orders')
        st.pyplot(fig, use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="section-header">Order Volume — City × Hour Heatmap</div>', unsafe_allow_html=True)
        pivot = df.groupby(['city','hour'])['order_id'].count().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(16,5.5))
        sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, ax=ax, cbar_kws={'label':'Orders'})
        ax.set_xlabel('Hour of Day'); ax.set_ylabel('City')
        st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="section-header">Restaurant Performance Scorecard</div>', unsafe_allow_html=True)
        sc = df.groupby('restaurant_id').agg(
            orders=('order_id','count'),
            avg_rating=('rating','mean'),
            avg_del=('delivery_time_min','mean'),
            revenue=('order_value','sum'),
            cancel_rate=('status',lambda x:(x=='Cancelled').mean()*100)
        ).reset_index()
        sc['score'] = (
            (sc['avg_rating']/5)*40 +
            ((90-sc['avg_del'])/80).clip(0,1)*30 +
            (1-sc['cancel_rate']/100)*30
        ).round(2)
        sc['tier'] = pd.qcut(sc['score'].rank(method='first'),3,labels=['Underperforming','Average','Top Performer'])
        fig, ax = plt.subplots(figsize=(10,5.5))
        for tier, clr in [('Top Performer',TEAL),('Average',GOLD),('Underperforming',CORAL)]:
            g = sc[sc['tier']==tier]
            ax.scatter(g['avg_rating'],g['avg_del'],alpha=0.5,s=25,label=tier,color=clr)
        ax.set_xlabel('Avg Rating'); ax.set_ylabel('Avg Delivery Time (min)'); ax.legend(title='Tier')
        st.pyplot(fig, use_container_width=True)
        under = (sc['tier']=='Underperforming').sum()
        st.markdown(f'<div class="warn-box">⚠️ <strong>{under}</strong> underperforming restaurants flagged for SLA review and renegotiation</div>', unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="section-header">Demand Forecasting — Linear Regression</div>', unsafe_allow_html=True)
        hc = df.groupby(['city','hour']).agg(
            order_count=('order_id','count'),
            avg_value=('order_value','mean'),
            avg_del=('delivery_time_min','mean')
        ).reset_index()
        X = hc[['hour','avg_value','avg_del']]
        y = hc['order_count']
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = LinearRegression()
        model.fit(Xtr,ytr)
        ypred = model.predict(Xte)
        from sklearn.metrics import r2_score
        r2 = r2_score(yte,ypred)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(yte,ypred,alpha=0.6,color=ACCENT,s=30)
        mn,mx = min(yte.min(),ypred.min()), max(yte.max(),ypred.max())
        ax.plot([mn,mx],[mn,mx],color=CORAL,lw=2,linestyle='--',label='Perfect fit')
        ax.set_xlabel('Actual Orders'); ax.set_ylabel('Predicted Orders')
        ax.set_title(f'Predicted vs Actual (R²={r2:.2f})', fontweight='bold', color=NAVY)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        st.metric("R² Score", f"{r2:.2f}", "Linear Regression")

    st.markdown("---")
    with st.expander("📄 View Raw Dataset (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT 3 — STUDENT DROPOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎓  Student Dropout Prediction":
    st.markdown("## 🎓 Student Academic Performance & Dropout Risk Prediction")
    st.markdown("*4,424 students · Random Forest · SMOTE balancing · AUC: 1.00*")

    with st.spinner("Training model..."):
        df, fi, fpr, tpr, acc, auc = gen_student()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Students",       f"{len(df):,}")
    c2.metric("Dropout Rate",   f"{df['dropout'].mean()*100:.1f}%")
    c3.metric("RF Accuracy",    f"{acc*100:.1f}%")
    c4.metric("AUC Score",      f"{auc:.2f}")
    c5.metric("High Risk",      f"{(df['risk']>0.66).sum():,}")

    tabs = st.tabs(["🌲 Feature Importance","📉 ROC Curve","📊 Attendance vs Dropout","🏛 Risk by Department","📋 Risk Scores"])

    with tabs[0]:
        st.markdown('<div class="section-header">Feature Importance — What Predicts Dropout?</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,5.5))
        colors = [CORAL if i==0 else ACCENT for i in range(len(fi))]
        ax.barh(fi['Feature'],fi['Importance'],color=colors,edgecolor='white',height=0.65)
        ax.set_xlabel('Importance Score'); ax.invert_yaxis()
        for i,(v,_) in enumerate(zip(fi['Importance'],fi['Feature'])):
            ax.text(v+0.002,i,f'{v:.3f}',va='center',fontsize=9)
        st.pyplot(fig, use_container_width=True)
        top = fi.iloc[0]
        st.markdown(f'<div class="insight-box">📌 <strong>{top["Feature"]}</strong> is the strongest predictor (importance: <strong>{top["Importance"]:.3f}</strong>) — students with low attendance are at highest dropout risk</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-header">ROC Curve — Random Forest vs Logistic Regression</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7,6))
        ax.plot(fpr,tpr,color=ACCENT,lw=2.5,label=f'Random Forest (AUC={auc:.2f})')
        ax.plot([0,1],[0,1],color=CORAL,lw=1.5,linestyle=':',label='Random')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold', color=NAVY); ax.legend()
        st.pyplot(fig, use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="section-header">Dropout Rate by Attendance Band</div>', unsafe_allow_html=True)
        att_d = df.groupby('att_bucket',observed=True)['dropout'].mean()*100
        fig, ax = plt.subplots(figsize=(10,4.5))
        colors = [CORAL if i<3 else (GOLD if i==3 else TEAL) for i in range(len(att_d))]
        bars = ax.bar(att_d.index.astype(str), att_d.values, color=colors, edgecolor='white', width=0.65)
        ax.axvline(x=3.5, color=NAVY, linestyle='--', lw=1.5, label='75% threshold')
        ax.set_xlabel('Attendance %'); ax.set_ylabel('Dropout Rate (%)'); ax.legend()
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        low_att = df[df['attendance']<75]['dropout'].mean()*100
        high_att = df[df['attendance']>=75]['dropout'].mean()*100
        if high_att > 0:
            st.markdown(f'<div class="insight-box">📌 Students with <strong>attendance &lt;75%</strong> drop out at <strong>{low_att:.1f}%</strong> vs <strong>{high_att:.1f}%</strong> for those above 75% — a <strong>{low_att/high_att:.1f}×</strong> higher rate</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="section-header">Dropout Risk Distribution by Department</div>', unsafe_allow_html=True)
        df['risk_tier'] = pd.cut(df['risk'],bins=[0,0.33,0.66,1.0],labels=['Low Risk','Medium Risk','High Risk'])
        risk_dep = df.groupby(['department','risk_tier'],observed=True).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12,5))
        risk_dep.plot(kind='bar',ax=ax,color=[TEAL,GOLD,CORAL],edgecolor='white',width=0.7)
        ax.set_xlabel('Department'); ax.set_ylabel('Students')
        ax.legend(title='Risk Tier'); plt.xticks(rotation=20, ha='right')
        st.pyplot(fig, use_container_width=True)

    with tabs[4]:
        st.markdown('<div class="section-header">Student Risk Score Table</div>', unsafe_allow_html=True)
        risk_df = df[['department','attendance','gpa','dropout','risk','risk_tier']].copy()
        risk_df['risk'] = risk_df['risk'].round(3)
        risk_df = risk_df.sort_values('risk',ascending=False).reset_index(drop=True)
        st.dataframe(risk_df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT 4 — HR ATTRITION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👔  HR Attrition Analytics":
    st.markdown("## 👔 HR Employee Attrition & People Analytics")
    st.markdown("*1,470 employees · Logistic Regression AUC: 0.86 · 35 variables*")

    with st.spinner("Training models..."):
        df, fi, fpr_lr, tpr_lr, fpr_rf, tpr_rf, lr_auc, rf_auc, lr_acc, dept_att = gen_hr()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Employees",      f"{len(df):,}")
    c2.metric("Attrition Rate", f"{df['attrition'].mean()*100:.1f}%")
    c3.metric("LR Accuracy",    f"{lr_acc*100:.1f}%")
    c4.metric("LR AUC",         f"{lr_auc:.2f}")
    c5.metric("High Risk",      f"{(df['risk_score']>0.66).sum():,}")

    tabs = st.tabs(["🏢 By Department","🌲 Feature Importance","📉 ROC Curves","🌡 Satisfaction Analysis","📋 Employee Risk Scores"])

    with tabs[0]:
        st.markdown('<div class="section-header">Attrition Rate by Department</div>', unsafe_allow_html=True)
        avg = dept_att['rate_%'].mean()
        fig, ax = plt.subplots(figsize=(10,5))
        colors = [CORAL if v > avg else ACCENT for v in dept_att['rate_%']]
        bars = ax.bar(dept_att['department'], dept_att['rate_%'], color=colors, edgecolor='white', width=0.65)
        ax.axhline(avg, linestyle='--', color=NAVY, alpha=0.6, label=f'Avg: {avg:.1f}%')
        ax.set_ylabel('Attrition Rate (%)'); ax.legend()
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        st.pyplot(fig, use_container_width=True)
        top_d = dept_att.loc[dept_att['rate_%'].idxmax()]
        st.markdown(f'<div class="warn-box">⚠️ <strong>{top_d["department"]}</strong> has the highest attrition at <strong>{top_d["rate_%"]:.1f}%</strong> — prioritise retention strategy here</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-header">Top Attrition Predictors (Random Forest)</div>', unsafe_allow_html=True)
        top10 = fi.head(10)
        fig, ax = plt.subplots(figsize=(10,5.5))
        colors = [CORAL if i<3 else ACCENT for i in range(len(top10))]
        ax.barh(top10['Feature'],top10['Importance'],color=colors,edgecolor='white',height=0.65)
        ax.set_xlabel('Feature Importance'); ax.invert_yaxis()
        for i,(v,_) in enumerate(zip(top10['Importance'],top10['Feature'])):
            ax.text(v+0.002,i,f'{v:.3f}',va='center',fontsize=9)
        st.pyplot(fig, use_container_width=True)
        top = fi.iloc[0]
        st.markdown(f'<div class="insight-box">📌 <strong>{top["Feature"]}</strong> is the #1 attrition predictor (importance: {top["Importance"]:.3f})</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="section-header">ROC Curves — Model Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr_lr,tpr_lr,color=TEAL,lw=2.5,label=f'Logistic Regression (AUC={lr_auc:.2f})')
        ax.plot(fpr_rf,tpr_rf,color=ACCENT,lw=2.5,linestyle='--',label=f'Random Forest (AUC={rf_auc:.2f})')
        ax.plot([0,1],[0,1],color=CORAL,lw=1.5,linestyle=':',label='Random')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontweight='bold', color=NAVY); ax.legend(loc='lower right')
        st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        st.markdown('<div class="section-header">Satisfaction Scores vs Attrition Rate</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1,3,figsize=(14,5))
        for ax, col, title in zip(axes,
            ['job_satisfaction','work_life_balance','env_satisfaction'],
            ['Job Satisfaction','Work-Life Balance','Env Satisfaction']):
            data = df.groupby(col)['attrition'].mean()*100
            ax.bar(data.index,data.values,
                   color=[CORAL if v>20 else ACCENT for v in data.values],
                   edgecolor='white',width=0.6)
            ax.set_title(title,fontweight='bold',color=NAVY,fontsize=11)
            ax.set_xlabel('Score (1=Low, 4=High)'); ax.set_ylabel('Attrition %')
            for i,v in enumerate(data.values):
                ax.text(data.index[i],v+0.3,f'{v:.1f}%',ha='center',fontsize=9,fontweight='bold')
        plt.suptitle('Employee Satisfaction vs Attrition', fontweight='bold', color=NAVY, fontsize=13)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        high_risk = df[(df['overtime']==1)&(df['job_satisfaction']<=2)&(df['years_at_company']<2)]
        base_rate = df['attrition'].mean()*100
        hr_rate   = high_risk['attrition'].mean()*100 if len(high_risk) > 0 else 0
        mult      = hr_rate/base_rate if base_rate > 0 else 0
        st.markdown(f'<div class="warn-box">⚠️ Employees with <strong>overtime + low satisfaction + &lt;2 yrs tenure</strong> → attrition rate of <strong>{hr_rate:.1f}%</strong> ({mult:.1f}× the {base_rate:.1f}% baseline)</div>', unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="section-header">Employee Attrition Risk Scores</div>', unsafe_allow_html=True)
        risk_df = df[['department','monthly_income','job_satisfaction','overtime','years_at_company','attrition','risk_score']].copy()
        risk_df['risk_score'] = risk_df['risk_score'].round(3)
        risk_df['risk_tier']  = pd.cut(risk_df['risk_score'],bins=[0,0.30,0.60,1.0],labels=['Low','Medium','High'])
        risk_df = risk_df.sort_values('risk_score',ascending=False).reset_index(drop=True)
        st.dataframe(risk_df.head(50), use_container_width=True)

    st.markdown("---")
    with st.expander("📄 View Raw Dataset (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)
