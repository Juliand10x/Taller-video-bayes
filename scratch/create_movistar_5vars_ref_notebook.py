import json
from pathlib import Path


ROOT = Path("/home/fabri/code/Semestre 4/EstadisticaBayesiana/TallerVideoBayes")
SRC = ROOT / "Cuadernos Jupyter" / "Multinomial_Movistar_5Vars.ipynb"
OUT = ROOT / "Cuadernos Jupyter" / "Multinomial_Movistar_5Vars_Referencia.ipynb"


def set_cell(nb, idx, text):
    nb["cells"][idx]["source"] = text.splitlines(keepends=True)


with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)


set_cell(
    nb,
    0,
    """# Regresión Multinomial Bayesiana — Compradores del Movistar Arena (5 variables + clase de referencia)

Este cuaderno implementa una **regresión logística multinomial bayesiana** con PyMC para clasificar compradores de boletas del concierto de Miguel Amezquita en el Movistar Arena dentro de tres perfiles:

| Clase | Código | Descripción |
|-------|--------|-------------|
| `Planner` | 0 | Compra con semanas de anticipación, comportamiento planificado |
| `In-Between` | 1 | Ni planificador ni compulsivo, grupo indeciso |
| `Last-Minute` | 2 | Compra a último momento, impulsado por urgencia |

**Predictores en esta versión:**
- `Age`
- `Num_Tickets_Purchased`
- `Concession_Purchases`
- `fan_int` (derivada de `Fan_Mailing_List`)
- `seat_lower` (derivada de `Seat_Location`)

**Corrección aplicada (identificabilidad):**
- En lugar de estimar logits libres para las 3 clases, fijamos `Planner` como clase de referencia.
- Se estima solo:
  - `alpha_raw` de tamaño 2 (`In-Between`, `Last-Minute`)
  - `beta_raw` de tamaño `n_pred × 2`
- Luego concatenamos ceros para la clase base (`Planner`) y reconstruimos:
  - `alpha = [0, alpha_raw]`
  - `beta = [0_col, beta_raw]`
- Así evitamos la dirección plana del softmax y anclamos la inferencia.

**Modelo matemático (multinomial con referencia):**
$$P(Y_i = c \\mid X_i) = \\frac{\\exp(\\eta_{i,c})}{\\sum_{j=0}^{2}\\exp(\\eta_{i,j})}$$
con
$$\\eta_{i,0}=0 \\quad (\\text{Planner, referencia}), \\qquad \\eta_{i,c}=\\alpha_c + X_i^\\prime\\beta_c,\\ c\\in\\{1,2\\}$$

**Facultad de Ciencia de Datos — Universidad Externado de Colombia**
""",
)

set_cell(
    nb,
    4,
    """# numpy: operaciones numéricas vectorizadas (arrays, matrices)
import numpy as np

# pandas: lectura y manipulación de datos tabulares (DataFrames)
import pandas as pd

# matplotlib: librería base de visualización en Python
import matplotlib.pyplot as plt

# seaborn: visualización estadística de alto nivel, construida sobre matplotlib
import seaborn as sns

# pymc: librería principal para modelado probabilístico bayesiano con MCMC
import pymc as pm

# arviz: librería para análisis e interpretación de trazas bayesianas (diagnósticos y plots)
import arviz as az

# pytensor tensor ops: concatenaciones/ceros para parametrización con clase de referencia
import pytensor.tensor as pt

# StandardScaler: estandariza variables (media=0, desv=1) para mejorar la convergencia MCMC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# confusion_matrix, classification_report: métricas de evaluación del clasificador
from sklearn.metrics import confusion_matrix, classification_report

# train_test_split: dividir datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# warnings: suprimir advertencias menores que no afectan los resultados
import warnings
warnings.filterwarnings('ignore')

# os: manejo de directorios del sistema (para crear carpetas de figuras)
import os

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

IMG_DIR = '../img/resultados_movistar_5vars_ref'
os.makedirs(IMG_DIR, exist_ok=True)

print(f'PyMC versión:  {pm.__version__}')
print(f'ArviZ versión: {az.__version__}')
print(f'NumPy versión: {np.__version__}')
""",
)

set_cell(
    nb,
    21,
    """## 4. Modelo Bayesiano con PyMC

### Especificación matemática completa (con clase de referencia)

Usamos una parametrización identificable fijando `Planner` como base.

**Priors más informativos:**
$$\\alpha_{raw,c} \\sim N(0, 1.5^2), \\quad c \\in \\{\\text{In-Between},\\text{Last-Minute}\\}$$
$$\\beta_{raw,p,c} \\sim N(0, 1.5^2), \\quad p=1,\\ldots,5$$

Definimos:
$$\\alpha = [0,\\alpha_{raw,1},\\alpha_{raw,2}]$$
$$\\beta = [\\mathbf{0},\\beta_{raw}]$$

donde la primera clase (`Planner`) queda anclada en cero.

Luego:
$$\\eta_i = X_i\\beta + \\alpha, \\qquad p_i = \\text{softmax}(\\eta_i), \\qquad Y_i \\sim \\text{Categorical}(p_i)$$

### ¿Por qué esta parametrización?
Porque elimina la no-identificabilidad aditiva del softmax (desplazamientos comunes de logits), mejora la estabilidad del muestreo y hace que los coeficientes sean interpretables como efectos **relativos a Planner**.
""",
)

set_cell(
    nb,
    22,
    """# Modelo multinomial con clase de referencia (Planner)
with pm.Model() as modelo_multinomial:

    # Solo dos interceptos libres: In-Between y Last-Minute (Planner=0)
    alpha_raw = pm.Normal(
        'alpha_raw',
        mu=0,
        sigma=1.5,
        shape=2
    )

    # Concatenamos la clase base fija en 0
    alpha = pm.Deterministic(
        'alpha',
        pt.concatenate([pt.zeros(1), alpha_raw])
    )

    # Solo dos columnas de coeficientes libres (n_pred x 2)
    beta_raw = pm.Normal(
        'beta_raw',
        mu=0,
        sigma=1.5,
        shape=(n_pred, 2)
    )

    zeros = pt.zeros((n_pred, 1))
    beta = pm.Deterministic(
        'beta',
        pt.concatenate([zeros, beta_raw], axis=1)
    )

    # Logits y softmax
    mu = pm.math.dot(X_train, beta) + alpha
    p = pm.math.softmax(mu, axis=-1)

    # Verosimilitud categórica
    y_obs = pm.Categorical('y_obs', p=p, observed=y_train)

print('Modelo especificado correctamente.')
print(f'Variables libres: {[rv.name for rv in modelo_multinomial.free_RVs]}')
print(f'Variables observadas: {[rv.name for rv in modelo_multinomial.observed_RVs]}')
""",
)

set_cell(
    nb,
    24,
    """# Muestreo MCMC (NUTS)
with modelo_multinomial:
    trace = pm.sample(
        draws=50000,
        tune=5000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True,
        progressbar=True
    )

print('\\n✓ Muestreo MCMC completado.')
""",
)

set_cell(
    nb,
    27,
    """# Traceplot de los interceptos libres (clases relativas a Planner)
az.plot_trace(
    trace,
    var_names=['alpha_raw'],
    figsize=(12, 5)
)
plt.suptitle('Traceplot — Interceptos libres alpha_raw (vs Planner)', y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/05_traceplot_alpha.png', bbox_inches='tight')
plt.show()
print('Guardado: 05_traceplot_alpha.png')
""",
)

set_cell(
    nb,
    28,
    """# Traceplot de coeficientes libres beta_raw (n_pred x 2)
az.plot_trace(
    trace,
    var_names=['beta_raw'],
    compact=True,
    figsize=(14, 5)
)
plt.suptitle('Traceplot — Coeficientes libres beta_raw (vs Planner)', y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/06_traceplot_beta.png', bbox_inches='tight')
plt.show()
print('Guardado: 06_traceplot_beta.png')
""",
)

set_cell(
    nb,
    30,
    """# Diagnóstico de convergencia sobre parámetros libres
resumen = az.summary(trace, var_names=['alpha_raw', 'beta_raw'], round_to=4)
print('Tabla de diagnósticos de convergencia (parámetros libres):')
display(resumen[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']])

rhat_max = resumen['r_hat'].max()
ess_min  = resumen['ess_bulk'].min()

if rhat_max < 1.01:
    print(f'\\n✓ Convergencia excelente: R-hat máximo = {rhat_max:.4f} (< 1.01)')
elif rhat_max < 1.05:
    print(f'\\n⚠ Convergencia aceptable: R-hat máximo = {rhat_max:.4f} (< 1.05)')
else:
    print(f'\\n✗ ADVERTENCIA: R-hat máximo = {rhat_max:.4f} — convergencia insuficiente')

print(f'ESS bulk mínimo: {ess_min:.0f} muestras efectivas')
""",
)

set_cell(
    nb,
    34,
    """# Posterior de interceptos libres
az.plot_posterior(
    trace,
    var_names=['alpha_raw'],
    hdi_prob=0.95,
    ref_val=0,
    figsize=(12, 4),
    textsize=10
)

for ax, label in zip(plt.gcf().axes, ['alpha_raw[0] — In-Between vs Planner', 'alpha_raw[1] — Last-Minute vs Planner']):
    ax.set_title(label)

plt.suptitle('Distribuciones Posteriores — Interceptos libres (HDI 95%)', y=1.05, fontsize=13)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/08_posterior_interceptos.png', bbox_inches='tight')
plt.show()
print('Guardado: 08_posterior_interceptos.png')
""",
)

set_cell(
    nb,
    36,
    """# Forest de coeficientes libres (comparaciones relativas a Planner)
az.plot_forest(
    trace,
    var_names=['beta_raw'],
    combined=True,
    hdi_prob=0.95,
    r_hat=True,
    figsize=(10, 10)
)
plt.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Efecto nulo (β=0)')
plt.title('Forest Plot — beta_raw (HDI 95%)\\nClase 0=In-Between vs Planner, Clase 1=Last-Minute vs Planner', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/09_forest_plot_beta.png', bbox_inches='tight')
plt.show()
print('Guardado: 09_forest_plot_beta.png')
""",
)

set_cell(
    nb,
    37,
    """# Forest de interceptos libres
az.plot_forest(
    trace,
    var_names=['alpha_raw'],
    combined=True,
    hdi_prob=0.95,
    r_hat=True,
    figsize=(8, 3)
)
plt.axvline(0, color='red', linestyle='--', linewidth=1.2)
plt.title('Forest Plot — alpha_raw (HDI 95%)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/10_forest_plot_alpha.png', bbox_inches='tight')
plt.show()
print('Guardado: 10_forest_plot_alpha.png')
""",
)

set_cell(
    nb,
    41,
    """# PPC en test usando la misma parametrización con referencia
with pm.Model() as modelo_pred:
    alpha_raw_pred = pm.Normal('alpha_raw', mu=0, sigma=1.5, shape=2)
    alpha_pred = pm.Deterministic('alpha', pt.concatenate([pt.zeros(1), alpha_raw_pred]))

    beta_raw_pred = pm.Normal('beta_raw', mu=0, sigma=1.5, shape=(n_pred, 2))
    zeros_pred = pt.zeros((n_pred, 1))
    beta_pred = pm.Deterministic('beta', pt.concatenate([zeros_pred, beta_raw_pred], axis=1))

    mu_pred = pm.math.dot(X_test, beta_pred) + alpha_pred
    p_pred = pm.math.softmax(mu_pred, axis=-1)
    y_obs_pred = pm.Categorical('y_obs', p=p_pred, observed=y_test)

    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

print('✓ Posterior Predictive Check completado.')
""",
)

set_cell(
    nb,
    50,
    r'''# ---- RESUMEN INTERPRETATIVO AUTOMÁTICO ----

def get_beta_stats(pred_idx, clase_idx):
    """Retorna (media, hdi_low, hdi_high) de beta[pred, clase]."""
    vals = trace.posterior['beta'].values[:, :, pred_idx, clase_idx].flatten()
    mean_val = float(vals.mean())
    hdi = az.hdi(vals, hdi_prob=0.95)
    return mean_val, float(hdi[0]), float(hdi[1])

def cruza_cero(low, high):
    return low < 0 < high

idx_age = nombres_pred.index('Age_z')
idx_tickets = nombres_pred.index('Num_Tickets_Purchased_z')
idx_conc = nombres_pred.index('Concession_Purchases_z')
idx_fan = nombres_pred.index('fan_int')
idx_seat = nombres_pred.index('seat_lower')

print('=' * 72)
print('PERFILES DE COMPRADORES — REGRESIÓN MULTINOMIAL BAYESIANA')
print('Movistar Arena — Modelo 5 variables con clase de referencia')
print('=' * 72)
print('Referencia fija: Planner (todos sus coeficientes = 0 por construcción).')

for c_idx, c_nombre in enumerate(nombres_clases):
    print(f'\n{"─"*70}')
    print(f'CLASE {c_idx}: {c_nombre.upper()}')
    print(f'{"─"*70}')
    if c_idx == 0:
        print('  Clase de referencia: coeficientes fijados en 0 (sin incertidumbre).')
        continue
    for p_idx, p_nombre in enumerate(nombres_pred):
        m, lo, hi = get_beta_stats(p_idx, c_idx)
        conclusion = '✓ Efecto robusto' if not cruza_cero(lo, hi) else '⚠ Efecto incierto'
        print(f'  {p_nombre:<30s}: β={m:+.3f}  [HDI95%: {lo:.3f}, {hi:.3f}]  {conclusion}')

print(f'\n{"═"*72}')
print('RESPUESTAS A LAS PREGUNTAS ORIENTADORAS DEL TALLER')
print(f'{"═"*72}')

m_tick_ib, lo_tick_ib, hi_tick_ib = get_beta_stats(idx_tickets, 1)
m_tick_lm, lo_tick_lm, hi_tick_lm = get_beta_stats(idx_tickets, 2)
print(f"""
1. ¿Qué caracteriza Planner frente a las demás clases?
   Planner es la referencia; los coeficientes estimados indican cuánto se alejan
   In-Between y Last-Minute respecto a Planner para cada predictor.
   - In-Between vs Planner (Num_Tickets): β={m_tick_ib:+.3f}
   - Last-Minute vs Planner (Num_Tickets): β={m_tick_lm:+.3f}
""")

m_fan_ib, lo_fan_ib, hi_fan_ib = get_beta_stats(idx_fan, 1)
m_fan_lm, lo_fan_lm, hi_fan_lm = get_beta_stats(idx_fan, 2)
print(f"""2. Señales de compra tardía y afinidad con la marca:
   - In-Between vs Planner (fan_int):   β={m_fan_ib:+.3f}
   - Last-Minute vs Planner (fan_int):  β={m_fan_lm:+.3f}
""")

print(f"""3. Predictor más discriminante (magnitud media):
   → {pred_mayor}: efecto absoluto promedio = {efecto_abs[pred_mayor]:.3f}
""")

accuracy = df_pred['Correcta'].mean()
print(f"""4. Desempeño predictivo:
   Accuracy en test: {accuracy*100:.1f}%
   Esta parametrización evita no-identificabilidad del softmax y hace que los HDI
   se interpreten de forma más estable en términos relativos a Planner.
""")

print('=' * 72)
''',
)

set_cell(
    nb,
    51,
    """## Resumen de figuras generadas

Todas las figuras se guardaron en `img/resultados_movistar_5vars_ref/`:
""",
)


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook generado en: {OUT}")
