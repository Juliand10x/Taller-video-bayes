import json
from pathlib import Path


ROOT = Path("/home/fabri/code/Semestre 4/EstadisticaBayesiana/TallerVideoBayes")
OUT = ROOT / "Cuadernos Jupyter" / "RegresionBayesiana_TotalMatches.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line for line in text.splitlines(keepends=True)],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line for line in text.splitlines(keepends=True)],
    }


cells = [
    md(
        """# Regresión Bayesiana para Conteos — Total Matches de Uber Bogotá

Este cuaderno desarrolla un análisis completo de regresión bayesiana con **PyMC** sobre el experimento de Uber en Bogotá. Se mantiene la misma estructura del cuaderno de `DriverPayout`, pero ahora el objetivo es modelar la eficiencia de emparejamiento medida por el **número total de matches** (`total_matches`).

**Variable objetivo elegida:** `total_matches`

**¿Por qué `total_matches` y no `total_double_matches`?**
- `total_matches` representa mejor la eficiencia operativa global del algoritmo de emparejamiento.
- Tiene una señal más fuerte frente a variables del experimento como `commute` y `treat`.
- `total_double_matches` es más específico, más subordinado al proceso de match general y menos interpretable como KPI principal.

**Distribución elegida:** `NegativeBinomial`

**Justificación de la verosimilitud:**
- La variable objetivo es un **conteo positivo**, así que `Beta` no aplica.
- La distribución empírica es asimétrica y no normal, por lo que una `Normal` simple no es adecuada.
- La varianza de `total_matches` es muy superior a su media, así que una `Poisson` también queda corta por **sobredispersión**.
- La `NegativeBinomial` permite modelar conteos con enlace log y dispersión extra-Poisson.

**Predictores seleccionados:**
- `trips_pool` — volumen de viajes POOL completados
- `rider_cancellations` — fricción operativa por cancelaciones
- `treat` — 1 si el tiempo de espera es 5 min, 0 si es 2 min
- `commute` — 1 si es hora pico
- `treat × commute` — interacción para capturar si el efecto del tratamiento cambia en hora pico

**Variables excluidas:**
- `trips_express` — tiene correlación muy alta con `total_matches`, casi mecánica
- `total_double_matches` — es un subconjunto del objetivo y generaría redundancia
- `total_driver_payout` — es más razonable tratarlo como resultado económico downstream que como predictor primario de matches
"""
    ),
    md("## 0. Instalación de dependencias (Google Colab)\n"),
    code(
        """# Descomenta si estás en Google Colab
# !pip install pymc arviz openpyxl scipy -q
"""
    ),
    md("## 1. Importaciones y configuración\n"),
    code(
        """import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

# Carpeta de salida para figuras
IMG_DIR = '../img/resultados_total_matches'
os.makedirs(IMG_DIR, exist_ok=True)

print(f'PyMC versión: {pm.__version__}')
print(f'ArviZ versión: {az.__version__}')
"""
    ),
    md(
        """## 1. Carga y EDA

Leemos la base del experimento de Uber en Bogotá. Cada fila corresponde a un período de 2 horas 40 minutos. En este cuaderno el foco ya no es el pago a conductores, sino la **cantidad total de matches**, es decir, cuántos viajes lograron emparejarse con al menos otro pasajero.
"""
    ),
    code(
        """# Ajusta la ruta si ejecutas desde Google Colab
DATA_PATH = '../data/BaseUBER.xlsx'

df = pd.read_excel(DATA_PATH)
print(f'Dimensiones: {df.shape}')
print(f'\\nTipos de variables:')
print(df.dtypes)
df.head()
"""
    ),
    md("### 1.1 Estadísticas descriptivas\n"),
    code(
        """# Vista general de todas las variables numéricas
desc = df.describe().T
desc['cv'] = desc['std'] / desc['mean']
display(desc.round(2))
"""
    ),
    code(
        """# Distribución de la variable objetivo
mean_matches = df['total_matches'].mean()
var_matches = df['total_matches'].var()
dispersion_ratio = var_matches / mean_matches

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['total_matches'], bins=25, color='steelblue', edgecolor='white')
axes[0].set_title('Distribución de total_matches')
axes[0].set_xlabel('Total matches')
axes[0].set_ylabel('Frecuencia')

# QQ-plot sobre log(1 + y) para mostrar que ni siquiera la transformación logra plena normalidad
stats.probplot(np.log1p(df['total_matches']), plot=axes[1])
axes[1].set_title('Q-Q Plot de log(1 + total_matches)')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/01_distribucion_objetivo.png', bbox_inches='tight')
plt.show()
print('Guardado: 01_distribucion_objetivo.png')

print(f'Media de total_matches: {mean_matches:,.2f}')
print(f'Varianza de total_matches: {var_matches:,.2f}')
print(f'Razón varianza/media: {dispersion_ratio:,.2f}')
print(f'Asimetría: {stats.skew(df["total_matches"]):.3f}')
print(f'Normalidad (D’Agostino) p-value: {stats.normaltest(df["total_matches"]).pvalue:.3e}')

if dispersion_ratio > 1.5:
    print('\\nConclusión: existe sobredispersión clara, lo que descarta una Poisson simple como primera opción.')
else:
    print('\\nConclusión: la relación varianza/media no muestra sobredispersión severa.')

print('Beta no aplica porque la variable no está acotada entre 0 y 1.')
"""
    ),
    md("### 1.2 Matriz de correlaciones\n"),
    code(
        """# Preparamos variables numéricas para la correlación
df['treat_int'] = df['treat'].astype(int)
df['commute_int'] = df['commute'].astype(int)
df['interaction_tc'] = df['treat_int'] * df['commute_int']

cols_corr = [
    'total_matches', 'trips_pool', 'trips_express',
    'rider_cancellations', 'treat_int', 'commute_int',
    'interaction_tc', 'total_double_matches'
]

corr = df[cols_corr].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, square=True,
    linewidths=0.5, cbar_kws={'shrink': 0.8}
)
plt.title('Matriz de Correlación de Pearson', fontsize=13)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/02_heatmap_correlacion.png', bbox_inches='tight')
plt.show()
print('Guardado: 02_heatmap_correlacion.png')

print('\\nCorrelaciones clave con total_matches:')
print(corr['total_matches'][['trips_express', 'total_double_matches', 'rider_cancellations', 'commute_int']].round(3))
"""
    ),
    md(
        """La correlación extremadamente alta entre `total_matches` y `trips_express`, junto con el hecho de que `total_double_matches` es un subconjunto directo del objetivo, hace preferible excluir ambas variables del modelo principal. Conservamos variables con interpretación operativa más clara: tratamiento, hora pico, cancelaciones, volumen POOL e interacción."""
    ),
    md("### 1.3 Boxplots por tratamiento y horario\n"),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Por tratamiento (wait_time)
df['wait_label'] = df['treat'].map({True: '5 mins (treat=1)', False: '2 mins (treat=0)'})
sns.boxplot(
    x='wait_label', y='total_matches', data=df,
    palette='Set2', ax=axes[0]
)
axes[0].set_title('Total matches por tiempo de espera (treat)')
axes[0].set_xlabel('Grupo')
axes[0].set_ylabel('total_matches')

# Por hora pico
df['commute_label'] = df['commute'].map({True: 'Hora pico (commute=1)', False: 'No pico (commute=0)'})
sns.boxplot(
    x='commute_label', y='total_matches', data=df,
    palette='muted', ax=axes[1]
)
axes[1].set_title('Total matches por horario (commute)')
axes[1].set_xlabel('Grupo')
axes[1].set_ylabel('total_matches')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/03_boxplots_treat_commute.png', bbox_inches='tight')
plt.show()
print('Guardado: 03_boxplots_treat_commute.png')
"""
    ),
    md("### 1.4 Scatter plots de la variable objetivo vs predictores clave\n"),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# total_matches vs rider_cancellations
sns.regplot(
    x='rider_cancellations', y='total_matches', data=df,
    scatter_kws={'alpha': 0.55, 'color': 'steelblue'},
    line_kws={'color': 'red'}, ax=axes[0]
)
axes[0].set_title('Total Matches vs Cancelaciones')
axes[0].set_xlabel('rider_cancellations')
axes[0].set_ylabel('total_matches')

# total_matches vs trips_pool
sns.regplot(
    x='trips_pool', y='total_matches', data=df,
    scatter_kws={'alpha': 0.55, 'color': 'darkorange'},
    line_kws={'color': 'red'}, ax=axes[1]
)
axes[1].set_title('Total Matches vs Trips POOL')
axes[1].set_xlabel('trips_pool')
axes[1].set_ylabel('total_matches')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/04_scatter_predictores.png', bbox_inches='tight')
plt.show()
print('Guardado: 04_scatter_predictores.png')
"""
    ),
    md(
        """## 2. Preprocesamiento

Estandarizamos los predictores numéricos continuos para mejorar la eficiencia del muestreo MCMC. La variable objetivo (`total_matches`) **no** se estandariza porque el modelo usa una verosimilitud de conteo y su interpretación se hace en la escala original del número esperado de matches."""
    ),
    code(
        """# Variables predictoras seleccionadas
predictores_num = ['trips_pool', 'rider_cancellations']
predictores_bin = ['treat_int', 'commute_int', 'interaction_tc']

# Estandarización manual de variables continuas
medias_orig = df[predictores_num].mean()
stds_orig = df[predictores_num].std()
X_num_df = (df[predictores_num] - medias_orig) / stds_orig
X_num_df.columns = [f'{c}_z' for c in predictores_num]

# Combinar con binarias
X_df = pd.concat([X_num_df, df[predictores_bin].reset_index(drop=True)], axis=1)
y = df['total_matches'].values.astype(int)

print('Variables en el modelo:')
print(X_df.columns.tolist())
print(f'\\nMatriz X: {X_df.shape}')
print(f'Vector y: {y.shape}')
print(f'\\nEstadísticas de X estandarizado:')
display(X_df.describe().round(3))

print('\\nMedias originales de predictores numéricos:')
print(medias_orig.round(2))

print('\\nDesviaciones estándar originales:')
print(stds_orig.round(2))
"""
    ),
    md(
        """## 3. Modelo Bayesiano con PyMC

Especificamos el siguiente modelo de regresión bayesiana para conteos:

$$\\text{total\\_matches}_i \\sim \\text{NegativeBinomial}(\\mu_i, \\alpha)$$

$$\\log(\\mu_i) = \\beta_0 + \\beta_1 \\cdot \\text{trips\\_pool}_z + \\beta_2 \\cdot \\text{rider\\_cancellations}_z + \\beta_3 \\cdot \\text{treat} + \\beta_4 \\cdot \\text{commute} + \\beta_5 \\cdot (\\text{treat} \\times \\text{commute})$$

**Priors débilmente informativos**:

| Parámetro | Prior | Justificación |
|-----------|-------|---------------|
| $\\beta_0$ | $N(\\log(\\bar y), 1.5^2)$ | Intercepto centrado en la escala natural del conteo promedio |
| $\\beta_1, \\beta_2$ | $N(0, 0.5^2)$ | Efectos de variables continuas estandarizadas en escala log |
| $\\beta_3, \\beta_5$ | $N(0, 0.4^2)$ | Efectos experimentales e interacción, moderados |
| $\\beta_4$ | $N(0, 0.5^2)$ | Hora pico puede tener efecto más marcado |
| $\\alpha$ | $HalfNormal(2)$ | Controla la sobredispersión; si crece mucho, el modelo se acerca a Poisson |

Como el enlace es logarítmico, los coeficientes se interpretan como cambios relativos en el número esperado de matches. Aproximadamente, $\\exp(\\beta_j) - 1$ da el cambio porcentual esperado asociado a un cambio unitario del predictor.
"""
    ),
    code(
        """# Convertir a arrays numpy para PyMC
trips_pool_z = X_df['trips_pool_z'].values
rider_cancellations_z = X_df['rider_cancellations_z'].values
treat_x = X_df['treat_int'].values
commute_x = X_df['commute_int'].values
interaction_x = X_df['interaction_tc'].values

with pm.Model() as modelo_matches:
    # ---- PRIORS débilmente informativos ----
    beta0 = pm.Normal('beta0', mu=np.log(y.mean()), sigma=1.5)
    beta1 = pm.Normal('beta1_pool', mu=0, sigma=0.5)
    beta2 = pm.Normal('beta2_cancel', mu=0, sigma=0.5)
    beta3 = pm.Normal('beta3_treat', mu=0, sigma=0.4)
    beta4 = pm.Normal('beta4_commute', mu=0, sigma=0.5)
    beta5 = pm.Normal('beta5_interaction', mu=0, sigma=0.4)
    alpha = pm.HalfNormal('alpha', sigma=2.0)

    # ---- PREDICTOR LINEAL Y MEDIA ----
    eta = (
        beta0
        + beta1 * trips_pool_z
        + beta2 * rider_cancellations_z
        + beta3 * treat_x
        + beta4 * commute_x
        + beta5 * interaction_x
    )
    mu = pm.Deterministic('mu', pm.math.exp(eta))

    # ---- VEROSIMILITUD ----
    y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha, observed=y)

print('Modelo especificado correctamente.')
print('Variables del modelo:', [rv.name for rv in modelo_matches.basic_RVs])
"""
    ),
    code(
        """# ---- MUESTREO MCMC (NUTS) ----
# tune alto para estabilizar una verosimilitud de conteo con enlace log
with modelo_matches:
    trace = pm.sample(
        draws=2000,
        tune=12000,
        chains=4,
        random_seed=42,
        target_accept=0.95,
        return_inferencedata=True
    )

print('\\n✓ Muestreo completado.')
"""
    ),
    md(
        """## 4. Diagnósticos MCMC

Antes de interpretar los resultados, verificamos la convergencia de las cadenas Markov. Los criterios clave son:
- **R-hat ≈ 1.0**: las cadenas convergieron al mismo espacio posterior
- **Trazas estacionarias**: mezcla adecuada sin tendencias
- **Energy plot**: buena exploración geométrica del posterior
- **Sin divergencias**: esencial en modelos con enlace log y sobredispersión
"""
    ),
    code(
        """# ---- TRACEPLOT ----
var_names = ['beta0', 'beta1_pool', 'beta2_cancel',
             'beta3_treat', 'beta4_commute', 'beta5_interaction', 'alpha']

ax = az.plot_trace(trace, var_names=var_names, figsize=(14, 14))
plt.suptitle('Traceplot — Diagnóstico de Convergencia', y=1.01, fontsize=14)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/05_traceplot.png', bbox_inches='tight')
plt.show()
print('Guardado: 05_traceplot.png')
"""
    ),
    code(
        """# ---- R-HAT ----
resumen = az.summary(trace, var_names=var_names, round_to=4)
print('Tabla de convergencia (R-hat debe estar muy cerca de 1.0):')
display(resumen[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']])

rhat_max = resumen['r_hat'].max()
if rhat_max < 1.01:
    print(f'\\n✓ Convergencia excelente: R-hat máximo = {rhat_max:.4f} (< 1.01)')
elif rhat_max < 1.05:
    print(f'\\n⚠ Convergencia aceptable: R-hat máximo = {rhat_max:.4f} (< 1.05)')
else:
    print(f'\\n✗ ADVERTENCIA: R-hat máximo = {rhat_max:.4f} — revisar modelo o aumentar tune')
"""
    ),
    code(
        """# ---- ENERGY PLOT ----
ax = az.plot_energy(trace, figsize=(8, 4))
plt.title('Energy Plot — Exploración del Espacio Posterior')
plt.savefig(f'{IMG_DIR}/06_energy_plot.png', bbox_inches='tight')
plt.show()
print('Guardado: 06_energy_plot.png')

n_div = int(trace.sample_stats.diverging.sum())
if n_div == 0:
    print(f'✓ Sin divergencias ({n_div})')
else:
    print(f'⚠ Divergencias detectadas: {n_div}')
"""
    ),
    md(
        """## 5. Resultados Posteriores

Una vez confirmada la convergencia, analizamos las distribuciones posteriores de los coeficientes. En este modelo los efectos están en **escala log**, así que la lectura más útil es en términos de cambios relativos del número esperado de matches.
"""
    ),
    code(
        """# ---- POSTERIOR DE COEFICIENTES (sin intercepto y alpha para mejor visualización) ----
coef_vars = ['beta1_pool', 'beta2_cancel', 'beta3_treat', 'beta4_commute', 'beta5_interaction']

az.plot_posterior(
    trace,
    var_names=coef_vars,
    hdi_prob=0.95,
    figsize=(14, 8),
    textsize=10
)
plt.suptitle('Distribuciones Posteriores de los Coeficientes (HDI 95%)', y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/07_posterior_coeficientes.png', bbox_inches='tight')
plt.show()
print('Guardado: 07_posterior_coeficientes.png')
"""
    ),
    code(
        """# ---- POSTERIOR DEL INTERCEPTO Y ALPHA ----
az.plot_posterior(
    trace,
    var_names=['beta0', 'alpha'],
    hdi_prob=0.95,
    figsize=(12, 4)
)
plt.suptitle('Distribuciones Posteriores: Intercepto y Alpha', y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/08_posterior_beta0_alpha.png', bbox_inches='tight')
plt.show()
"""
    ),
    code(
        """# ---- FOREST PLOT (Intervalos de Credibilidad al 95%) ----
ax = az.plot_forest(
    trace,
    var_names=coef_vars,
    hdi_prob=0.95,
    combined=True,
    figsize=(9, 5)
)
plt.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Efecto nulo')
plt.title('Forest Plot — Intervalos HDI al 95% de los Coeficientes', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/09_forest_plot.png', bbox_inches='tight')
plt.show()
print('Guardado: 09_forest_plot.png')
"""
    ),
    code(
        """# ---- TABLA RESUMEN COMPLETA ----
print('Tabla resumen de la distribución posterior:')
print('(mean=media posterior, sd=desviación, hdi_3%/hdi_97%=intervalo de credibilidad 95%)')
display(resumen)
"""
    ),
    md(
        """## 6. Posterior Predictive Check (PPC)

El PPC verifica si el modelo puede generar conteos parecidos a los observados. Si la `NegativeBinomial` captura bien la sobredispersión y la asimetría, la distribución predictiva posterior debería cubrir razonablemente a `total_matches`.
"""
    ),
    code(
        """# ---- MUESTREO PREDICTIVO POSTERIOR ----
with modelo_matches:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

print('✓ Muestreo predictivo posterior completado.')
"""
    ),
    code(
        """# ---- GRÁFICO PPC ----
az.plot_ppc(
    ppc,
    observed=True,
    num_pp_samples=200,
    figsize=(10, 5)
)
plt.title('Posterior Predictive Check — Ajuste del Modelo', fontsize=12)
plt.xlabel('total_matches')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/10_ppc.png', bbox_inches='tight')
plt.show()
print('Guardado: 10_ppc.png')
"""
    ),
    code(
        """# ---- PREDICHOS vs OBSERVADOS ----
y_pred_mean = ppc.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
y_pred_lower = ppc.posterior_predictive['y_obs'].quantile(0.025, dim=['chain', 'draw']).values
y_pred_upper = ppc.posterior_predictive['y_obs'].quantile(0.975, dim=['chain', 'draw']).values

residuales = y - y_pred_mean

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y, y_pred_mean, alpha=0.6, color='steelblue', edgecolors='white', linewidths=0.3)
lim = [min(y.min(), y_pred_mean.min()) * 0.95, max(y.max(), y_pred_mean.max()) * 1.05]
axes[0].plot(lim, lim, 'r--', linewidth=1.5, label='Predicción perfecta')
axes[0].set_xlim(lim)
axes[0].set_ylim(lim)
axes[0].set_xlabel('Valores reales (matches)')
axes[0].set_ylabel('Media posterior predicha (matches)')
axes[0].set_title('Predichos vs Observados')
axes[0].legend()

axes[1].scatter(y_pred_mean, residuales, alpha=0.6, color='darkorange', edgecolors='white', linewidths=0.3)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.2)
axes[1].set_xlabel('Media posterior predicha (matches)')
axes[1].set_ylabel('Residual (matches)')
axes[1].set_title('Residuales vs Predichos')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/11_predichos_vs_reales.png', bbox_inches='tight')
plt.show()
print('Guardado: 11_predichos_vs_reales.png')
"""
    ),
    code(
        """# ---- R² BAYESIANO ----
y_pred_2d = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(y))
r2 = az.r2_score(y, y_pred_2d)

print('Pseudo-R² Bayesiano:')
print(f'  Media:  {r2["r2"]:.4f}')
print(f'  SD:     {r2["r2_std"]:.4f}')
print(f'\\nInterpretación: el modelo explica en promedio el {r2["r2"]*100:.1f}% de la variación')
print('en el total de matches observados.')
"""
    ),
    md(
        """## 7. Interpretación Automática de Resultados

A continuación se genera automáticamente un resumen interpretativo en español basado en los valores posteriores estimados.
"""
    ),
    code(
        """# Extraer medias posteriores y HDI 95% de los coeficientes
posterior = trace.posterior

def get_stats(var):
    vals = posterior[var].values.flatten()
    mean_val = float(vals.mean())
    hdi = az.hdi(vals, hdi_prob=0.95)
    return mean_val, float(hdi[0]), float(hdi[1])

def pct_effect(beta):
    return 100 * (np.exp(beta) - 1)

def signo(val):
    return 'POSITIVO' if val > 0 else 'NEGATIVO'

def cruza_cero(low, high):
    return low < 0 < high

b0_m, b0_l, b0_h = get_stats('beta0')
b1_m, b1_l, b1_h = get_stats('beta1_pool')
b2_m, b2_l, b2_h = get_stats('beta2_cancel')
b3_m, b3_l, b3_h = get_stats('beta3_treat')
b4_m, b4_l, b4_h = get_stats('beta4_commute')
b5_m, b5_l, b5_h = get_stats('beta5_interaction')
alp_m, alp_l, alp_h = get_stats('alpha')
r2_val = r2['r2']

efectos_pct = {
    'trips_pool': abs(pct_effect(b1_m)),
    'rider_cancellations': abs(pct_effect(b2_m)),
    'treat': abs(pct_effect(b3_m)),
    'commute': abs(pct_effect(b4_m)),
    'interaccion_treat_commute': abs(pct_effect(b5_m)),
}
predictor_mayor = max(efectos_pct, key=efectos_pct.get)

print('=' * 70)
print('INTERPRETACIÓN DE RESULTADOS — REGRESIÓN BAYESIANA UBER BOGOTÁ')
print('=' * 70)

print(f\"\"\"
1. EFECTO DEL TRATAMIENTO (beta3 — wait_time = 5 mins)
   Media posterior (log-escala): {b3_m:.3f}  [HDI 95%: {b3_l:.3f} a {b3_h:.3f}]
   Cambio porcentual esperado: {pct_effect(b3_m):.1f}% en total_matches
   Signo: {signo(b3_m)}

   El coeficiente beta3_treat es {signo(b3_m).lower()}, lo que significa que pasar
   de 2 a 5 minutos de espera {'AUMENTA' if b3_m > 0 else 'REDUCE'} el número esperado
   de matches en aproximadamente {abs(pct_effect(b3_m)):.1f}% por período.
   {'El HDI cruza el cero, así que el efecto no es concluyente.' if cruza_cero(b3_l, b3_h) else 'El HDI no cruza el cero, así que hay evidencia posterior sólida del efecto.'}
\"\"\")

print(f\"\"\"
2. EFECTO DE HORA PICO (beta4 — commute = 1)
   Media posterior (log-escala): {b4_m:.3f}  [HDI 95%: {b4_l:.3f} a {b4_h:.3f}]
   Cambio porcentual esperado: {pct_effect(b4_m):.1f}% en total_matches
   Signo: {signo(b4_m)}

   El coeficiente beta4_commute es {signo(b4_m).lower()}, lo que implica que durante
   horas pico el número esperado de matches {'aumenta' if b4_m > 0 else 'disminuye'}
   en promedio {abs(pct_effect(b4_m)):.1f}%.
   {'El HDI cruza el cero: no hay evidencia concluyente del efecto de hora pico.' if cruza_cero(b4_l, b4_h) else 'El HDI no cruza el cero: el efecto de hora pico es robusto.'}
\"\"\")

print(f\"\"\"
3. PREDICTOR CON MAYOR EFECTO RELATIVO
   Variable: {predictor_mayor}
   Efecto absoluto estimado: {efectos_pct[predictor_mayor]:.1f}% sobre la media esperada

   En el Forest Plot, '{predictor_mayor}' aparece como el efecto relativo de mayor magnitud
   sobre el total esperado de matches.
\"\"\")

print(f\"\"\"
4. DISPERSIÓN EXTRA-POISSON
   Alpha posterior: {alp_m:.3f}  [HDI 95%: {alp_l:.3f} a {alp_h:.3f}]

   La presencia de un parámetro alpha finito confirma que el modelo está permitiendo
   sobredispersión. Esto refuerza la decisión de usar Negative Binomial en lugar de Poisson.
\"\"\")

print(f\"\"\"
5. CONCLUSIÓN OPERATIVA
   Pseudo-R² Bayesiano: {r2_val:.3f} ({r2_val*100:.1f}% de variación explicada)

   El modelo sugiere que `total_matches` es una métrica operativa defendible porque responde
   directamente a condiciones del experimento y del contexto de demanda.

   Desde el punto de vista operativo:
   - beta3_treat {'NO cruza el cero → cambiar el tiempo de espera sí altera la tasa esperada de matches.' if not cruza_cero(b3_l, b3_h) else 'cruza el cero → el cambio de 2 a 5 minutos no tiene un efecto suficientemente concluyente.'}
   - beta4_commute {'NO cruza el cero → la hora pico cambia de forma clara el nivel esperado de emparejamientos.' if not cruza_cero(b4_l, b4_h) else 'cruza el cero → la influencia de la hora pico sigue siendo incierta.'}
   - beta5_interaction {'NO cruza el cero → el tratamiento se comporta distinto en hora pico.' if not cruza_cero(b5_l, b5_h) else 'cruza el cero → no hay evidencia fuerte de que el tratamiento cambie específicamente en hora pico.'}

   Conclusión general: `total_matches` funciona mejor como variable objetivo que `total_double_matches`
   si el interés principal es medir eficiencia global del sistema de emparejamiento. Además, por su forma
   empírica, una Negative Binomial es más coherente que una Normal, una Beta o una Poisson simple.
\"\"\")
"""
    ),
    md(
        """## Resumen de figuras generadas

Todas las figuras se guardaron en `img/resultados_total_matches/`:
"""
    ),
    code(
        """import glob
figuras = sorted(glob.glob(f'{IMG_DIR}/*.png'))
print(f'Figuras generadas ({len(figuras)} archivos):')
for f in figuras:
    print(f'  {os.path.basename(f)}')
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(f"Notebook generado en: {OUT}")
