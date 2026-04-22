import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

print("Iniciando corrección técnica del gráfico PPC...")
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
    
    trace = pm.sample(draws=2000, tune=1000, chains=2, random_seed=42, progressbar=False)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42, progressbar=False)

# GRAFICA CORREGIDA: Comparando dispersión contra datos observados
az.style.use("arviz-darkgrid")
fig, ax = plt.subplots(figsize=(10, 6))

# El parámetro 'num_pp_samples' genera las nubes de simulación para comparar con la realidad
az.plot_ppc(ppc, ax=ax, num_pp_samples=50)

ax.set_title("Validación: ¿Qué tan bien imita el modelo la demanda de Uber?", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Cantidad de Viajes", fontsize=12, fontweight='bold')
ax.set_ylabel("Densidad (Frecuencia)", fontsize=12, fontweight='bold')

# Renombrar leyenda a español
handles, _ = ax.get_legend_handles_labels()
# En az.plot_ppc el orden suele ser: 0=Observed, 1=Posterior predictive, 2=Posterior predictive mean
leg_labels = ["Datos Reales (Uber)", "Simulaciones del Modelo", "Predicción Promedio"]
ax.legend(handles, leg_labels, loc='upper right', fontsize=11, frameon=True, facecolor='white')

plt.tight_layout()
plt.savefig('uber_ppc_plot.png', dpi=300)
print("¡Imagen uber_ppc_plot.png corregida y guardada con éxito!")
