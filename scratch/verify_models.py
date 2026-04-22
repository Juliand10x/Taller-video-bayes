import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('data/BaseUBER.xlsx')
df['wait_time_num'] = df['wait_time'].str.extract('(\d+)').astype(float)
df['commute'] = df['commute'].astype(int)

def check(cols, name):
    y = df['trips_pool'].values
    X = df[cols].values
    with pm.Model() as model:
        ic = pm.Normal('i', 1400, 500)
        b = pm.Normal('b', 0, 10, shape=len(cols))
        s = pm.HalfNormal('s', 300)
        mu = ic + pm.math.dot(X, b)
        pm.Normal('o', mu, s, observed=y)
        trace = pm.sample(500, tune=500, chains=2, progressbar=False, random_seed=42)
    
    summary = az.summary(trace, var_names=['b'])
    print(f"\n--- {name} ({cols}) ---")
    print(summary)
    for i, col in enumerate(cols):
        low = summary.iloc[i]['hdi_3%']
        high = summary.iloc[i]['hdi_97%']
        sig = "SOLID (Away from 0)" if (low > 0 or high < 0) else "WEAK (Contains 0)"
        print(f"  {col}: {sig} [{low:.2f}, {high:.2f}]")

check(['wait_time_num', 'commute'], "M1")
check(['wait_time_num', 'rider_cancellations'], "M2")
check(['total_driver_payout', 'wait_time_num'], "M3")
check(['total_driver_payout', 'rider_cancellations'], "M4")
