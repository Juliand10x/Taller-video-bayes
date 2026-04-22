
import pandas as pd
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler
import os

# Cargar datos
DATA_PATH = '/home/fabri/code/Semestre 4/EstadisticaBayesiana/TallerVideoBayes/data/BaseUBER.xlsx'
df = pd.read_excel(DATA_PATH)

# Preprocesamiento (mismo del notebook)
predictores_num = ['total_matches', 'trips_express', 'rider_cancellations']
predictores_bin = ['treat', 'commute']

df['treat_int'] = df['treat'].astype(int)
df['commute_int'] = df['commute'].astype(int)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(df[predictores_num])
X_num_df = pd.DataFrame(X_num_scaled, columns=[f'{c}_z' for c in predictores_num])
X_df = pd.concat([X_num_df, df[['treat_int', 'commute_int']].reset_index(drop=True)], axis=1)
y = df['total_driver_payout'].values

# Preparar vectores para PyMC
total_matches_z = X_df['total_matches_z'].values
trips_express_z = X_df['trips_express_z'].values
rider_cancellations_z = X_df['rider_cancellations_z'].values
treat_x = X_df['treat_int'].values
commute_x = X_df['commute_int'].values

print("Definiendo el modelo...")
with pm.Model() as modelo_uber:
    beta0 = pm.Normal('beta0', mu=0, sigma=10000)
    beta1 = pm.Normal('beta1_matches', mu=0, sigma=1000)
    beta2 = pm.Normal('beta2_express', mu=0, sigma=1000)
    beta3 = pm.Normal('beta3_cancel', mu=0, sigma=1000)
    beta4 = pm.Normal('beta4_treat', mu=0, sigma=500)
    beta5 = pm.Normal('beta5_commute', mu=0, sigma=500)
    sigma = pm.HalfNormal('sigma', sigma=1000)

    mu = (
        beta0
        + beta1 * total_matches_z
        + beta2 * trips_express_z
        + beta3 * rider_cancellations_z
        + beta4 * treat_x
        + beta5 * commute_x
    )

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print("Probando muestreo corto (10 draws)...")
with modelo_uber:
    trace = pm.sample(draws=10, tune=50, chains=2, random_seed=42, return_inferencedata=True)

print("✓ El modelo y el entorno están funcionando correctamente.")
