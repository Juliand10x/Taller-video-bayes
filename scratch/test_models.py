import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import os

# Filter warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('data/BaseUBER.xlsx')
df['wait_time_num'] = df['wait_time'].str.extract('(\d+)').astype(float)
df['commute'] = df['commute'].astype(int)

def test_model(name, cols):
    print(f"\n--- Testing Model: {name} ({cols}) ---")
    y = df['trips_pool'].values
    X = df[cols].values
    
    with pm.Model() as m:
        intercept = pm.Normal('intercept', 0, 100)
        betas = pm.Normal('betas', 0, 10, shape=len(cols))
        sigma = pm.HalfNormal('sigma', 300)
        
        mu = intercept + pm.math.dot(X, betas)
        likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Reduced samples for quick check
        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, random_seed=42, progressbar=False)
        
    summary = az.summary(trace, var_names=['betas'])
    print(summary)
    
    # Check if HDIs are away from 0
    for i, col in enumerate(cols):
        hdi_lower = summary.iloc[i]['hdi_3%']
        hdi_upper = summary.iloc[i]['hdi_97%']
        is_significant = not (hdi_lower <= 0 <= hdi_upper)
        print(f"Variable '{col}': Significant? {is_significant} (HDI: [{hdi_lower:.2f}, {hdi_upper:.2f}])")

# Run models
test_model("M1", ['wait_time_num', 'commute'])
test_model("M2", ['wait_time_num', 'rider_cancellations'])
test_model("M3", ['total_matches', 'wait_time_num'])
test_model("M4", ['total_matches', 'total_double_matches'])
