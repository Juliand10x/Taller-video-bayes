import pandas as pd
import pymc as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("Iniciando generación de PPC totalmente customizado...")
df = pd.read_excel('BaseUBER.xlsx')
df['wait_time_num'] = df['wait_time'].str.extract(r'(\d+)').astype(float)
X_wait = df['wait_time_num'].values
X_commute = df['commute'].values
Y = df['trips_pool'].values

with pm.Model() as m:
    beta_0 = pm.Normal('beta_0', mu=1400, sigma=500)
    beta_wait = pm.Normal('beta_wait', mu=0, sigma=10)
    beta_commute = pm.Normal('beta_commute', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=300)
    mu = beta_0 + beta_wait * X_wait + beta_commute * X_commute
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=Y)
    
    # Muestreo súper ágil solo para tener forma distributiva
    trace = pm.sample(draws=1000, tune=1000, chains=2, random_seed=42, progressbar=False)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42, progressbar=False)

# Extraer simulaciones
y_sim = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(Y))

# Diseño custom usando Seaborn puro
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Graficar un puñado de simulaciones individuales en el fondo
for sim in y_sim[:100]:
    sns.kdeplot(sim, color='#5c92c8', alpha=0.05, linewidth=1)

# Plot de la media simulada
mean_sim = y_sim.mean(axis=0)
sns.kdeplot(mean_sim, color='#0f4c81', linewidth=4, label='Predicción Promedio del Modelo')

# Plot de la vida real
sns.kdeplot(Y, color='#d9534f', linewidth=3, linestyle='--', label='Datos Reales Observados (Uber)')

# Textos
plt.title("Chequeo Predictivo a Posteriori (PPC)\n¿Logró el algoritmo imitar la vida real de la empresa?", fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Cantidad total de Viajes Compartidos por Periodo Muestreado", fontsize=13, fontweight='bold')
plt.ylabel("Densidad Probabilística de Ocurrencia", fontsize=13, fontweight='bold')

plt.xlim(500, 2500) # Mantenerlo centrado
plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.savefig('uber_ppc_plot.png', dpi=300)
print("¡Archivo uber_ppc_plot.png sobreescrito con EXITO TOTAL!")
