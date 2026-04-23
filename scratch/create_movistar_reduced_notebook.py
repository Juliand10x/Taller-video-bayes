import json
from pathlib import Path


ROOT = Path("/home/fabri/code/Semestre 4/EstadisticaBayesiana/TallerVideoBayes")
SRC = ROOT / "Cuadernos Jupyter" / "Multinomial_Movistar.ipynb"
OUT = ROOT / "Cuadernos Jupyter" / "Multinomial_Movistar_Reducido.ipynb"


def set_cell(nb, idx, text):
    nb["cells"][idx]["source"] = text.splitlines(keepends=True)


with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)


set_cell(
    nb,
    0,
    """# Regresión Multinomial Bayesiana — Compradores del Movistar Arena (modelo reducido)

Este cuaderno implementa una **regresión logística multinomial bayesiana** con PyMC para clasificar compradores de boletas del concierto de Miguel Amezquita en el Movistar Arena dentro de tres perfiles:

| Clase | Código | Descripción |
|-------|--------|-------------|
| `Planner` | 0 | Compra con semanas de anticipación, comportamiento planificado |
| `In-Between` | 1 | Ni planificador ni compulsivo, grupo indeciso |
| `Last-Minute` | 2 | Compra a último momento, impulsado por urgencia |

**Objetivo:** No solo clasificar, sino entender *qué características* del cliente y su compra se asocian con cada tipo de comportamiento.

**Criterio de reducción del modelo:**
- En la versión completa, `Days_Before_Concierto` separaba casi perfectamente las clases, lo que generaba una matriz de confusión demasiado buena y un riesgo claro de sobreajuste.
- Para obtener un modelo más interpretable y menos tautológico, aquí usamos solo **3 predictores**:
  - `Num_Tickets_Purchased`
  - `Fan_Mailing_List`
  - `Seat_Location`
- Excluimos `Age` y `Concession_Purchases` por menor relevancia sustantiva.
- Excluimos `Days_Before_Concierto` y `Ticket_Price` para evitar que el modelo dependa de variables demasiado cercanas a la definición del comportamiento o redundantes con la ubicación y el tamaño de la compra.

**Modelo matemático (del taller):**
$$P(Y_i = c \\mid X_i) = \\frac{\\exp(\\alpha_c + X_i^\\prime \\beta_c)}{\\sum_{j=1}^{3} \\exp(\\alpha_j + X_i^\\prime \\beta_j)}, \\quad c \\in \\{\\text{Planner, In-Between, Last-Minute}\\}$$

Donde:
- $\\alpha_c$ es el intercepto de la clase $c$
- $\\beta_c$ es el vector de coeficientes para la clase $c$
- La función **softmax** garantiza que las tres probabilidades sumen 1

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

# Configuración visual: estilo limpio con grilla blanca, resolución adecuada
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

# Carpeta donde se guardarán todas las figuras generadas
IMG_DIR = '../img/resultados_movistar_reducido'
os.makedirs(IMG_DIR, exist_ok=True)

# Verificar versiones instaladas
print(f'PyMC versión:  {pm.__version__}')
print(f'ArviZ versión: {az.__version__}')
print(f'NumPy versión: {np.__version__}')
""",
)

set_cell(
    nb,
    7,
    "### 2.2 Estadísticas descriptivas del modelo reducido\n",
)

set_cell(
    nb,
    8,
    """# Estadísticas descriptivas enfocadas en las variables elegidas
print('=== Estadísticas globales de predictores seleccionados ===')
display(df[['Num_Tickets_Purchased']].describe().round(2))

print('\\n=== Media de Num_Tickets_Purchased por tipo de cliente ===')
display(
    df.groupby('Customer_Type')[['Num_Tickets_Purchased']].mean().round(2)
)

print('\\n=== Proporción de Fan_Mailing_List por tipo de cliente ===')
display(
    pd.crosstab(df['Customer_Type'], df['Fan_Mailing_List'], normalize='index').round(3)
)

print('\\n=== Proporción de Seat_Location por tipo de cliente ===')
display(
    pd.crosstab(df['Customer_Type'], df['Seat_Location'], normalize='index').round(3)
)

print('\\n=== Distribución de Customer_Type ===')
print(df['Customer_Type'].value_counts())
print(f'\\nPorcentajes:')
print((df['Customer_Type'].value_counts(normalize=True) * 100).round(1))
""",
)

set_cell(
    nb,
    11,
    "### 2.4 Distribución de predictores seleccionados por tipo de cliente\n",
)

set_cell(
    nb,
    12,
    """# En el modelo reducido trabajamos con 1 variable continua y 2 binarias
orden_clases = ['Planner', 'In-Between', 'Last-Minute']
paleta = {'Planner': '#2196F3', 'In-Between': '#FF9800', 'Last-Minute': '#4CAF50'}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

# 1) Num_Tickets_Purchased por clase
sns.boxplot(
    x='Customer_Type', y='Num_Tickets_Purchased', data=df,
    order=orden_clases, palette=paleta, ax=axes[0]
)
axes[0].set_title('Num_Tickets_Purchased por tipo de cliente')
axes[0].set_xlabel('')

# 2) Proporción de Fan_Mailing_List por clase
fan_plot = (
    pd.crosstab(df['Customer_Type'], df['Fan_Mailing_List'], normalize='index')
    .reindex(orden_clases)
)
fan_plot.plot(
    kind='bar', stacked=True, ax=axes[1],
    color=['#BDBDBD', '#9C27B0'], edgecolor='white'
)
axes[1].set_title('Fan_Mailing_List por tipo de cliente')
axes[1].set_xlabel('')
axes[1].set_ylabel('Proporción')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(title='Fan')

# 3) Proporción de Seat_Location por clase
seat_plot = (
    pd.crosstab(df['Customer_Type'], df['Seat_Location'], normalize='index')
    .reindex(orden_clases)
)
seat_plot.plot(
    kind='bar', stacked=True, ax=axes[2],
    color=['#26A69A', '#90A4AE'], edgecolor='white'
)
axes[2].set_title('Seat_Location por tipo de cliente')
axes[2].set_xlabel('')
axes[2].set_ylabel('Proporción')
axes[2].tick_params(axis='x', rotation=15)
axes[2].legend(title='Ubicación')

axes[3].set_visible(False)

plt.suptitle('Distribución de predictores seleccionados por tipo de comprador', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/02_boxplots_por_clase.png', bbox_inches='tight')
plt.show()
print('Guardado: 02_boxplots_por_clase.png')
""",
)

set_cell(
    nb,
    14,
    """# Codificación temporal de variables categóricas solo para calcular correlación
df_corr = df.copy()
df_corr['fan_num'] = (df_corr['Fan_Mailing_List'] == 'Yes').astype(int)
df_corr['seat_num'] = (df_corr['Seat_Location'] == 'Lower').astype(int)

# Variables numéricas para la matriz de correlación del modelo reducido
cols_num = ['Num_Tickets_Purchased', 'fan_num', 'seat_num']
corr = df_corr[cols_num].corr()

plt.figure(figsize=(7, 5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, square=True,
    linewidths=0.5, cbar_kws={'shrink': 0.8}
)
plt.title('Matriz de Correlación de Pearson — Modelo reducido', fontsize=13)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/03_heatmap_correlacion.png', bbox_inches='tight')
plt.show()
print('Guardado: 03_heatmap_correlacion.png')
""",
)

set_cell(
    nb,
    15,
    "### 2.6 Violin y barras de predictores clave\n",
)

set_cell(
    nb,
    16,
    """fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Violin plot de Num_Tickets_Purchased
sns.violinplot(
    x='Customer_Type', y='Num_Tickets_Purchased', data=df,
    order=orden_clases, palette=paleta, ax=axes[0], inner='box'
)
axes[0].set_title('Num_Tickets_Purchased por tipo de cliente')
axes[0].set_xlabel('Tipo de cliente')
axes[0].set_ylabel('Número de boletas')

# Countplot agrupado: Fan_Mailing_List dentro de cada clase
fan_counts = df.groupby(['Customer_Type', 'Fan_Mailing_List']).size().reset_index(name='count')
sns.barplot(
    data=fan_counts, x='Customer_Type', y='count',
    hue='Fan_Mailing_List', order=orden_clases,
    palette={'Yes': '#9C27B0', 'No': '#BDBDBD'}, ax=axes[1]
)
axes[1].set_title('Fan Mailing List por tipo de cliente')
axes[1].set_xlabel('Tipo de cliente')
axes[1].set_ylabel('Número de clientes')
axes[1].legend(title='Fan')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/04_violin_dias_fan.png', bbox_inches='tight')
plt.show()
print('Guardado: 04_violin_dias_fan.png')
""",
)

set_cell(
    nb,
    17,
    """## 3. Preprocesamiento

Preparamos los datos para PyMC siguiendo el mismo esquema del cuaderno original:
- Variables continuas → estandarizar con `StandardScaler`
- Variables binarias → codificar como 0/1
- Variable objetivo → codificar como enteros 0, 1, 2

En esta versión reducida usamos solo tres predictores para evitar un modelo excesivamente ajustado.
""",
)

set_cell(
    nb,
    18,
    """# ---- CODIFICACIÓN DE VARIABLES CATEGÓRICAS ----

df['fan_int'] = (df['Fan_Mailing_List'] == 'Yes').astype(int)
df['seat_lower'] = (df['Seat_Location'] == 'Lower').astype(int)

# ---- CODIFICACIÓN DE LA VARIABLE OBJETIVO ----
mapa_clases = {'Planner': 0, 'In-Between': 1, 'Last-Minute': 2}
df['y_encoded'] = df['Customer_Type'].map(mapa_clases)

print('Verificación de codificación de la variable objetivo:')
print(df[['Customer_Type', 'y_encoded']].value_counts().sort_index())

# ---- ESTANDARIZACIÓN DE VARIABLES CONTINUAS ----
predictores_continuos = [
    'Num_Tickets_Purchased'
]

scaler = StandardScaler()
X_continuas_z = scaler.fit_transform(df[predictores_continuos])
X_continuas_df = pd.DataFrame(
    X_continuas_z,
    columns=[f'{c}_z' for c in predictores_continuos]
)

predictores_binarios = ['fan_int', 'seat_lower']

X_df = pd.concat(
    [X_continuas_df, df[predictores_binarios].reset_index(drop=True)],
    axis=1
)

X_data = X_df.values.astype('float64')
y_data = df['y_encoded'].values

n_obs    = X_data.shape[0]
n_pred   = X_data.shape[1]
n_clases = 3

print(f'\\nDimensiones del problema:')
print(f'  Observaciones (n):  {n_obs}')
print(f'  Predictores (p):    {n_pred}  → {X_df.columns.tolist()}')
print(f'  Clases (C):         {n_clases} → {list(mapa_clases.keys())}')
print(f'  Parámetros totales: {n_clases} interceptos + {n_pred}×{n_clases} coeficientes = {n_clases + n_pred*n_clases}')

print(f'\\nEstadísticas de X estandarizado:')
display(X_df.describe().round(3))
""",
)

set_cell(
    nb,
    21,
    """## 4. Modelo Bayesiano con PyMC

### Especificación matemática completa

Mantenemos el mismo tipo de modelo multinomial bayesiano, pero ahora con solo **3 predictores**.

**Priors:**
$$\\alpha_c \\sim N(0, 5^2), \\quad c \\in \\{0, 1, 2\\}$$
$$\\beta_{p,c} \\sim N(0, 5^2), \\quad p = 1, \\ldots, 3; \\quad c \\in \\{0, 1, 2\\}$$

**Modelo lineal (logits):**
$$\\eta_{i,c} = \\alpha_c + \\sum_{p=1}^{3} X_{i,p} \\cdot \\beta_{p,c}$$

**Función softmax (probabilidades):**
$$p_{i,c} = \\text{softmax}(\\eta_i)_c = \\frac{\\exp(\\eta_{i,c})}{\\sum_{j=0}^{2} \\exp(\\eta_{i,j})}$$

**Verosimilitud:**
$$Y_i \\sim \\text{Categorical}(p_{i,0}, p_{i,1}, p_{i,2})$$

### ¿Por qué priors N(0, 5²)?
Como el profesor usa distribuciones no informativas cuando no hay datos históricos, usamos $N(0, 5^2)$: un prior amplio que no domina los datos pero sí estabiliza la exploración MCMC.
""",
)

set_cell(
    nb,
    23,
    """### 4.1 Muestreo MCMC — NUTS

Usamos el algoritmo **NUTS** (No-U-Turn Sampler), igual que en el cuaderno original.

**Parámetros:**
- `draws=50000`: número de muestras a guardar **por cadena**
- `tune=5000`: pasos de adaptación
- `chains=4`: 4 cadenas independientes
- `target_accept=0.9`: tasa de aceptación objetivo
- `random_seed=42`: reproducibilidad

> ⏱ **Nota:** Aunque el modelo reducido tiene menos predictores y es más liviano que el modelo completo, el muestreo puede seguir tomando varios minutos.
""",
)

set_cell(
    nb,
    24,
    """# El muestreo se ejecuta dentro del contexto del modelo
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
    33,
    """## 6. Resultados Posteriores

### 6.1 Distribuciones posteriores de los interceptos

Los interceptos `alpha[c]` representan el log-odds de pertenecer a la clase $c$ cuando el cliente es promedio en `Num_Tickets_Purchased` y tiene las categorías base de las variables binarias (`fan_int=0`, `seat_lower=0`).
""",
)

set_cell(
    nb,
    39,
    """# Construimos una tabla legible con los nombres de los predictores
# para facilitar la interpretación del forest plot

nombres_pred = X_df.columns.tolist()
nombres_clases = list(mapa_clases.keys())

beta_posterior = trace.posterior['beta']
beta_mean = beta_posterior.mean(dim=['chain', 'draw']).values

tabla_coef = pd.DataFrame(
    beta_mean,
    index=nombres_pred,
    columns=[f'β — {c}' for c in nombres_clases]
)

print('Media posterior de coeficientes beta por clase:')
print('(valores positivos = mayor probabilidad de esa clase, negativos = menor probabilidad)')
display(tabla_coef.round(3))

efecto_abs = tabla_coef.abs().mean(axis=1)
pred_mayor = efecto_abs.idxmax()
print(f'\\nPredictor con mayor efecto medio: {pred_mayor} (efecto abs. promedio = {efecto_abs[pred_mayor]:.3f})')
""",
)

set_cell(
    nb,
    48,
    """## 8. Perfiles de Compradores

Respondemos las preguntas orientadoras del taller basándonos en los coeficientes posteriores del **modelo reducido**.
""",
)

set_cell(
    nb,
    50,
    '''# ---- RESUMEN INTERPRETATIVO AUTOMÁTICO ----

def get_beta_stats(pred_idx, clase_idx):
    vals = trace.posterior['beta'].values[:, :, pred_idx, clase_idx].flatten()
    mean_val = float(vals.mean())
    hdi = az.hdi(vals, hdi_prob=0.95)
    return mean_val, float(hdi[0]), float(hdi[1])

def cruza_cero(low, high):
    return low < 0 < high

idx_tickets = nombres_pred.index('Num_Tickets_Purchased_z')
idx_fan = nombres_pred.index('fan_int')
idx_seat = nombres_pred.index('seat_lower')

print('=' * 72)
print('PERFILES DE COMPRADORES — REGRESIÓN MULTINOMIAL BAYESIANA')
print('Movistar Arena — Concierto Miguel Amezquita (modelo reducido)')
print('=' * 72)

for c_idx, c_nombre in enumerate(nombres_clases):
    print(f'\\n{"─"*70}')
    print(f'CLASE {c_idx}: {c_nombre.upper()}')
    print(f'{"─"*70}')
    for p_idx, p_nombre in enumerate(nombres_pred):
        m, lo, hi = get_beta_stats(p_idx, c_idx)
        conclusion = '✓ Efecto robusto' if not cruza_cero(lo, hi) else '⚠ Efecto incierto'
        print(f'  {p_nombre:<30s}: β={m:+.3f}  [HDI95%: {lo:.3f}, {hi:.3f}]  {conclusion}')

print(f'\\n{"═"*72}')
print('RESPUESTAS A LAS PREGUNTAS ORIENTADORAS DEL TALLER')
print(f'{"═"*72}')

m_tick_plan, lo_tick_plan, hi_tick_plan = get_beta_stats(idx_tickets, 0)
m_fan_plan, lo_fan_plan, hi_fan_plan = get_beta_stats(idx_fan, 0)
m_seat_plan, lo_seat_plan, hi_seat_plan = get_beta_stats(idx_seat, 0)
print(f"""
1. ¿Qué características asocian con compradores que planean con anticipación?
   → Num_Tickets_Purchased: β={m_tick_plan:+.3f} para Planner
   → Fan_Mailing_List:      β={m_fan_plan:+.3f} para Planner
   → Seat_Location(Lower):  β={m_seat_plan:+.3f} para Planner
   La lectura conjunta permite evaluar si los planners compran más boletas, tienen más vínculo con la marca
   y eligen ubicaciones premium.
""")

m_tick_lm, lo_tick_lm, hi_tick_lm = get_beta_stats(idx_tickets, 2)
m_fan_lm, lo_fan_lm, hi_fan_lm = get_beta_stats(idx_fan, 2)
m_seat_lm, lo_seat_lm, hi_seat_lm = get_beta_stats(idx_seat, 2)
print(f"""2. ¿Qué señales identifican a quienes compran a última hora?
   → Num_Tickets_Purchased: β={m_tick_lm:+.3f} para Last-Minute
   → Fan_Mailing_List:      β={m_fan_lm:+.3f} para Last-Minute
   → Seat_Location(Lower):  β={m_seat_lm:+.3f} para Last-Minute
   Si estos coeficientes son negativos, sugieren que el comprador de última hora compra menos boletas,
   participa menos en mailing y termina aceptando zonas menos premium.
""")

print(f"""3. ¿Cuál es el predictor con mayor poder discriminante?
   → {pred_mayor}: efecto absoluto promedio = {efecto_abs[pred_mayor]:.3f}
   Este predictor es el que más diferencia el perfil de compra entre los tres grupos en el modelo reducido.
""")

accuracy = df_pred['Correcta'].mean()
print(f"""4. ¿Cómo apoya esta segmentación el diseño de campañas y logística?
   El modelo reducido alcanza una accuracy del {accuracy*100:.1f}% en el conjunto de prueba.
   Al excluir Days_Before_Concierto, esta accuracy debe interpretarse como una medición más honesta de capacidad
   predictiva y no como una clasificación casi tautológica.
   - Planners: preventas, beneficios para fans, paquetes grupales y asientos premium.
   - Last-Minute: mensajes de urgencia, remanentes de inventario y ofertas de último momento.
   - In-Between: recordatorios graduales y campañas de conversión intermedia.
""")

print('=' * 72)
''',
)

set_cell(
    nb,
    51,
    """## Resumen de figuras generadas

Todas las figuras se guardaron en `img/resultados_movistar_reducido/`:
""",
)


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook generado en: {OUT}")
