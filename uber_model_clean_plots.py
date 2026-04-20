import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

print("Cargando datos...")
df = pd.read_excel('BaseUBER.xlsx')

df['wait_time_num'] = df['wait_time'].str.extract(r'(\d+)').astype(float)
X_wait = df['wait_time_num'].values
X_commute = df['commute'].values
Y = df['trips_pool'].values

print("Entrenando el modelo...")
with pm.Model() as uber_model:
    beta_0 = pm.Normal('beta_0', mu=1400, sigma=500)
    beta_wait = pm.Normal('beta_wait', mu=0, sigma=10)
    beta_commute = pm.Normal('beta_commute', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=300)
    
    mu = beta_0 + beta_wait * X_wait + beta_commute * X_commute
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=Y)
    
    trace = pm.sample(draws=10000, tune=2000, chains=4, return_inferencedata=True, random_seed=42)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

summary = az.summary(trace)
summary.to_csv('uber_summary.csv')

# ==========================================
# 1. TRACEPLOT PROFESIONAL (Solo variables clave)
# ==========================================
print("Generando Traceplot limpio...")
az.style.use("arviz-darkgrid")
axes = az.plot_trace(
    trace, 
    var_names=['beta_wait', 'beta_commute'],
    figsize=(12, 6),
    lines=[("beta_wait", {}, 0), ("beta_commute", {}, 0)] # Línea en cero
)
# Ajustar títulos
fig = plt.gcf()
fig.suptitle("Diagnóstico MCMC de las Variables de Interés (10,000 Iteraciones)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('uber_trace_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. FOREST PLOT PROFESIONAL (Sin superposiciones)
# ==========================================
print("Generando Forest Plot profesional...")
params = ['beta_wait', 'beta_commute']
labels = ['Tiempo de Espera\n(beta_wait)', 'Efecto Hora Pico\n(beta_commute)']
means = summary.loc[params, 'mean'].values
hdi_lower = summary.loc[params, 'hdi_3%'].values
hdi_upper = summary.loc[params, 'hdi_97%'].values
errors = [means - hdi_lower, hdi_upper - means]

plt.figure(figsize=(10, 4.5))
plt.style.use('seaborn-v0_8-whitegrid')

plt.errorbar(means, range(len(params)), xerr=errors, 
             fmt='o', color='#0f4c81', ecolor='#5c92c8', 
             elinewidth=5, capsize=8, capthick=2, markersize=12)

plt.yticks(range(len(params)), labels, fontsize=12, fontweight='bold', color='#333333')
plt.gca().invert_yaxis()
plt.axvline(x=0, color='#d9534f', linestyle='--', linewidth=2.5, label='Efecto Nulo (0)')

plt.title('Intervalos de Credibilidad al 95% (H.D.I.)\nEvaluación de Significancia', 
          pad=15, fontsize=14, fontweight='bold', color='#111111')
plt.xlabel('Magnitud del Efecto en Cantidad de Viajes', fontsize=12)

# Escribir los números a la DERECHA del bigote para NUNCA superponerse
for i, (m, l, u) in enumerate(zip(means, hdi_lower, hdi_upper)):
    plt.text(u + 2, i, f'Media: {m:.1f}  |  HDI: [{l:.1f}, {u:.1f}]', 
             ha='left', va='center', fontsize=11, fontweight='bold', color='#222222')

plt.xlim(min(hdi_lower)-5, max(hdi_upper)+25) # Espacio extra para el texto
plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.savefig('uber_forest_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. PPC PLOT PROFESIONAL
# ==========================================
print("Generando PPC Plot profesional...")
az.style.use("arviz-darkgrid")
ax = az.plot_ppc(ppc, figsize=(10, 6), num_pp_samples=100, mean_kwargs={"linewidth": 3}, observed_kwargs={"linewidth": 3})
plt.title("Chequeo Predictivo a Posteriori (PPC)\n¿Qué tanto se parecen los datos simulados a la realidad?", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('uber_ppc_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("¡Todas las gráficas generadas y sobreescritas con éxito!")
