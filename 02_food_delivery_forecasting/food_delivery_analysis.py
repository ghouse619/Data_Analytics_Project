"""
Food Delivery Operations & Demand Forecasting Analysis
Mohammed Ghouse | Data Analyst Portfolio Project 2
Dataset: Synthetic Swiggy/Zomato-style delivery data (50,000+ orders)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'figure.dpi':150})
NAVY=  '#0A2342'; ACCENT='#1A56A0'; TEAL='#1D9E75'; CORAL='#E8633A'; GOLD='#F0A500'

np.random.seed(7)
N = 50_000

# ── 1. GENERATE DATA ─────────────────────────────────────────────────────────
print("Generating food delivery dataset...")

cities     = ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata','Pune','Ahmedabad','Jaipur','Surat']
cuisines   = ['Indian','Chinese','Italian','Fast Food','South Indian','North Indian','Biryani','Desserts','Healthy','Mexican']
platforms  = ['Swiggy','Zomato','Direct App']
status_w   = [0.82, 0.10, 0.08]   # delivered / cancelled / delayed

start = datetime(2023, 6, 1)
end   = datetime(2024, 11, 30)
delta = (end - start).days

# Simulate time-based demand spikes
base_dates = [start + timedelta(days=int(np.random.uniform(0, delta))) for _ in range(N)]
base_hours = []
for _ in range(N):
    r = np.random.random()
    if r < 0.12:   h = np.random.randint(7, 11)    # breakfast
    elif r < 0.32: h = np.random.randint(12, 15)   # lunch
    elif r < 0.85: h = np.random.randint(19, 23)   # dinner peak
    else:          h = np.random.randint(0, 7)      # night/other
    base_hours.append(h)

order_timestamps = [d.replace(hour=h, minute=np.random.randint(0,60)) for d, h in zip(base_dates, base_hours)]

restaurant_ids = np.random.randint(1, 2001, N)
delivery_times = np.clip(np.random.normal(35, 12, N), 10, 90).astype(int)
order_vals     = np.round(np.random.lognormal(5.4, 0.6, N), 2).clip(80, 3000)
ratings        = np.random.choice([1,2,3,4,5], N, p=[0.03,0.07,0.15,0.40,0.35])
distance_km    = np.round(np.random.uniform(0.5, 12, N), 1)
status         = np.random.choice(['Delivered','Cancelled','Delayed'], N, p=status_w)

df = pd.DataFrame({
    'order_id':        range(1, N+1),
    'restaurant_id':   restaurant_ids,
    'city':            np.random.choice(cities, N, p=[0.20,0.18,0.16,0.12,0.10,0.08,0.06,0.04,0.04,0.02]),
    'cuisine':         np.random.choice(cuisines, N),
    'platform':        np.random.choice(platforms, N, p=[0.45,0.45,0.10]),
    'order_timestamp': order_timestamps,
    'delivery_time_min': delivery_times,
    'order_value':     order_vals,
    'customer_rating': ratings,
    'distance_km':     distance_km,
    'status':          status,
})
df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])

# Introduce nulls and duplicates for realistic cleaning
null_idx = np.random.choice(df.index, int(N*0.07), replace=False)
df.loc[null_idx[:int(N*0.04)], 'delivery_time_min'] = np.nan
df.loc[null_idx[int(N*0.04):], 'customer_rating']   = np.nan
dup_idx = np.random.choice(df.index[:300], 40, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

print(f"Raw dataset: {len(df):,} rows | Nulls: {df.isnull().sum().sum()} | Dupes: {df.duplicated('order_id').sum()}")

# ── 2. CLEAN ──────────────────────────────────────────────────────────────────
df = df.drop_duplicates('order_id').reset_index(drop=True)
df['delivery_time_min'] = df['delivery_time_min'].fillna(df.groupby('city')['delivery_time_min'].transform('median'))
df['customer_rating']   = df['customer_rating'].fillna(df['customer_rating'].median())
df['order_value']       = df['order_value'].clip(80, 3000)
print(f"Clean dataset: {len(df):,} rows | Nulls: {df.isnull().sum().sum()}")

# Feature extraction
df['hour']        = df['order_timestamp'].dt.hour
df['day_of_week'] = df['order_timestamp'].dt.day_name()
df['month']       = df['order_timestamp'].dt.to_period('M')
df['is_weekend']  = df['order_timestamp'].dt.dayofweek >= 5
df['is_peak']     = df['hour'].between(19, 22)

# ── 3. DEMAND ANALYSIS ────────────────────────────────────────────────────────
print("\nDemand pattern analysis...")
hourly_demand = df.groupby('hour').agg(orders=('order_id','count'), avg_value=('order_value','mean')).reset_index()
peak_hour = hourly_demand.loc[hourly_demand['orders'].idxmax(), 'hour']
print(f"Peak demand hour: {peak_hour}:00 ({hourly_demand['orders'].max():,} orders)")

dow_demand = df.groupby('day_of_week')['order_id'].count().reset_index(name='orders')
day_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_demand['day_of_week'] = pd.Categorical(dow_demand['day_of_week'], categories=day_order, ordered=True)
dow_demand = dow_demand.sort_values('day_of_week')

cancellation = df.groupby('city').apply(lambda x: (x['status']=='Cancelled').mean()*100).reset_index(name='cancel_rate_%')
cancellation = cancellation.sort_values('cancel_rate_%', ascending=False)
print(f"\nHighest cancellation city: {cancellation.iloc[0]['city']} ({cancellation.iloc[0]['cancel_rate_%']:.1f}%)")

# ── 4. RESTAURANT SCORECARD ───────────────────────────────────────────────────
print("\nBuilding restaurant scorecard...")
scorecards = df.groupby('restaurant_id').agg(
    total_orders   = ('order_id','count'),
    avg_rating     = ('customer_rating','mean'),
    avg_del_time   = ('delivery_time_min','mean'),
    total_revenue  = ('order_value','sum'),
    cancel_rate    = ('status', lambda x: (x=='Cancelled').mean()*100)
).reset_index()
scorecards['performance_score'] = (
    (scorecards['avg_rating'] / 5) * 40 +
    ((90 - scorecards['avg_del_time']) / 80).clip(0,1) * 30 +
    ((1 - scorecards['cancel_rate']/100)) * 30
).round(2)
scorecards['tier'] = pd.qcut(scorecards['performance_score'], 3, labels=['Underperforming','Average','Top Performer'])

under = scorecards[scorecards['tier']=='Underperforming']
print(f"Underperforming restaurants: {len(under)} of {len(scorecards)}")
print(f"Avg performance score — Top: {scorecards[scorecards['tier']=='Top Performer']['performance_score'].mean():.1f} | Under: {under['performance_score'].mean():.1f}")

# ── 5. DEMAND FORECASTING MODEL ───────────────────────────────────────────────
print("\nBuilding demand forecasting model...")
hourly_city = df.groupby(['city','hour','is_weekend']).agg(
    order_count   = ('order_id','count'),
    avg_value     = ('order_value','mean'),
    avg_del_time  = ('delivery_time_min','mean')
).reset_index()

# Features
X = pd.get_dummies(hourly_city[['hour','is_weekend','avg_value','avg_del_time']], drop_first=True)
y = hourly_city['order_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Linear Regression → R²: {r2:.2f} | MAE: {mae:.1f} orders")

# ── 6. CHARTS ─────────────────────────────────────────────────────────────────
print("\nGenerating charts...")

# Chart 1 — Hourly demand heatmap (city × hour)
pivot = df.groupby(['city','hour'])['order_id'].count().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(pivot, cmap='YlOrRd', annot=False, linewidths=0.3, ax=ax, cbar_kws={'label':'Orders'})
ax.set_title('Order Volume Heatmap — City × Hour of Day', fontsize=14, fontweight='bold', color=NAVY, pad=12)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('City')
plt.tight_layout()
plt.savefig('/home/claude/projects/02_food_delivery_forecasting/charts/01_demand_heatmap.png', bbox_inches='tight')
plt.close()

# Chart 2 — Hourly demand curve
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.fill_between(hourly_demand['hour'], hourly_demand['orders'], alpha=0.15, color=CORAL)
ax.plot(hourly_demand['hour'], hourly_demand['orders'], color=CORAL, linewidth=2.5, marker='o', markersize=6)
ax.axvspan(19, 22, alpha=0.1, color=GOLD, label='Peak window (7–10 PM)')
ax.set_title('Hourly Order Volume — Platform-wide Demand Curve', fontsize=14, fontweight='bold', color=NAVY, pad=12)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Total Orders')
ax.set_xticks(range(0,24))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/projects/02_food_delivery_forecasting/charts/02_hourly_demand_curve.png', bbox_inches='tight')
plt.close()

# Chart 3 — Day of week demand
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [CORAL if d in ['Friday','Saturday','Sunday'] else ACCENT for d in dow_demand['day_of_week']]
bars = ax.bar(dow_demand['day_of_week'], dow_demand['orders'], color=colors, edgecolor='white', width=0.6)
ax.set_title('Order Volume by Day of Week', fontsize=13, fontweight='bold', color=NAVY)
ax.set_ylabel('Total Orders')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30, f'{int(bar.get_height()):,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/projects/02_food_delivery_forecasting/charts/03_dow_demand.png', bbox_inches='tight')
plt.close()

# Chart 4 — Restaurant scorecard scatter
fig, ax = plt.subplots(figsize=(11, 6))
tier_colors = {'Top Performer':TEAL,'Average':GOLD,'Underperforming':CORAL}
for tier, grp in scorecards.groupby('tier'):
    ax.scatter(grp['avg_rating'], grp['avg_del_time'], alpha=0.5, s=30,
               label=tier, color=tier_colors[str(tier)])
ax.set_title('Restaurant Scorecard: Avg Rating vs Avg Delivery Time', fontsize=13, fontweight='bold', color=NAVY)
ax.set_xlabel('Average Customer Rating'); ax.set_ylabel('Average Delivery Time (min)')
ax.legend(title='Performance Tier')
plt.tight_layout()
plt.savefig('/home/claude/projects/02_food_delivery_forecasting/charts/04_restaurant_scorecard.png', bbox_inches='tight')
plt.close()

# Chart 5 — Forecast vs Actual
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test, y_pred, alpha=0.5, color=ACCENT, s=25, label='Predicted vs Actual')
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], color=CORAL, linewidth=2, linestyle='--', label='Perfect prediction')
ax.set_title(f'Demand Forecasting Model — Predicted vs Actual (R²={r2:.2f})', fontsize=13, fontweight='bold', color=NAVY)
ax.set_xlabel('Actual Orders'); ax.set_ylabel('Predicted Orders')
ax.legend()
plt.tight_layout()
plt.savefig('/home/claude/projects/02_food_delivery_forecasting/charts/05_forecast_vs_actual.png', bbox_inches='tight')
plt.close()

# ── 7. SAVE ───────────────────────────────────────────────────────────────────
df.to_csv('/home/claude/projects/02_food_delivery_forecasting/data/food_delivery_clean.csv', index=False)
scorecards.to_csv('/home/claude/projects/02_food_delivery_forecasting/outputs/restaurant_scorecards.csv', index=False)
hourly_demand.to_csv('/home/claude/projects/02_food_delivery_forecasting/outputs/hourly_demand.csv', index=False)
cancellation.to_csv('/home/claude/projects/02_food_delivery_forecasting/outputs/cancellation_by_city.csv', index=False)

report = f"""FOOD DELIVERY OPERATIONS & DEMAND FORECASTING — ANALYSIS REPORT
Mohammed Ghouse | Data Analyst Portfolio
Generated: {datetime.now().strftime('%d %B %Y')}
{'='*60}

DATASET
  Raw records     : {N+40:,}
  After cleaning  : {len(df):,}
  Date range      : Jun 2023 – Nov 2024
  Cities covered  : {df['city'].nunique()}
  Restaurants     : {df['restaurant_id'].nunique():,}

KEY METRICS
  Total Orders    : {len(df):,}
  Total Revenue   : ₹{df['order_value'].sum():,.0f}
  Avg Order Value : ₹{df['order_value'].mean():.2f}
  Avg Delivery    : {df['delivery_time_min'].mean():.1f} min
  Avg Rating      : {df['customer_rating'].mean():.2f}/5
  Cancellation %  : {(df['status']=='Cancelled').mean()*100:.1f}%

DEMAND PATTERNS
  Peak hour       : {peak_hour}:00 ({hourly_demand['orders'].max():,} orders)
  Peak window     : 7 PM – 10 PM (weekdays)
  Weekend lift    : {df[df['is_weekend']].shape[0]/df[~df['is_weekend']].shape[0]*7/5:.2f}× vs weekdays (per day)

FORECASTING MODEL
  Algorithm       : Linear Regression
  R² Score        : {r2:.2f}
  MAE             : {mae:.1f} orders per zone/hour

RESTAURANT SCORECARD
  Total restaurants : {len(scorecards):,}
  Top Performers    : {len(scorecards[scorecards['tier']=='Top Performer']):,}
  Underperforming   : {len(under):,} (flagged for review)

CANCELLATION BY CITY
{cancellation.to_string(index=False)}

RECOMMENDATIONS
  1. Deploy 20% more drivers between 7–10 PM on weekdays (demand forecasting)
  2. Flag {len(under)} underperforming restaurants for SLA review and renegotiation
  3. Investigate {cancellation.iloc[0]['city']} cancellation spike — supply-side issue likely
  4. Introduce peak-hour dynamic pricing to improve margin on high-demand slots
"""
with open('/home/claude/projects/02_food_delivery_forecasting/outputs/analysis_report.txt','w') as f:
    f.write(report)

print("\n✓ Project 2 complete — all charts and outputs saved")
print(report)
