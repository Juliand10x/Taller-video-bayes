import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

print("Cargando modelo para refinar PPC...")
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
    
    # 2000 draws es perfecto para graficar KDEs suaves en un PPC
    trace = pm.sample(draws=2000, tune=1000, chains=4, random_seed=42)
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

print("Generando Gráfica PPC altamente personalizada...")
az.style.use("arviz-darkgrid")
fig, ax = plt.subplots(figsize=(10, 6))

az.plot_ppc(ppc, num_pp_samples=100, ax=ax, 
            mean_kwargs={"linewidth": 3}, 
            observed_kwargs={"linewidth": 3})

# Configuración ultra descriptiva para los profesores
ax.set_title("Chequeo Predictivo a Posteriori (PPC)\nDemostración Visual de Efectividad del Modelo Bayesiano", 
             fontsize=14, fontweight='bold', pad=20, color='#111111')

ax.set_xlabel("Cantidad de Viajes (trips_pool) logrados en Uber", fontsize=12, fontweight='bold', color='#333333')
ax.set_ylabel("Frecuencia Estadística (Densidad)", fontsize=12, fontweight='bold', color='#333333')

# Extraer y traducir la leyenda automática de Arviz
handles, labels = ax.get_legend_handles_labels()
new_labels = []
for label in labels:
    if 'observed' in label.lower():
        new_labels.append('Datos Reales (Observados por Uber)')
    elif 'mean' in label.lower():
        new_labels.append('Predicción Promedio del Modelo')
    elif 'posterior' in label.lower() or 'ppc' in label.lower() or 'predictive' in label.lower():
        new_labels.append('Incertidumbre: Múltiples Simulaciones del Modelo')
    else:
        new_labels.append(label)

if handles:
    ax.legend(handles, new_labels, fontsize=11, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
plt.savefig('uber_ppc_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("¡PPC actualizado con ejes y leyendas descriptivas!")
