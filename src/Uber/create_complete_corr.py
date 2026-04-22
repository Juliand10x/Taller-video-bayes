import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create directory if it doesn't exist
os.makedirs('img/uber', exist_ok=True)

# Load data
df = pd.read_excel('data/BaseUBER.xlsx')

# 1. Processing wait_time: extract number (e.g., '2 mins' -> 2)
df['wait_time_num'] = df['wait_time'].str.extract('(\d+)').astype(float)

# 2. Convert boolean flags to integers
df['treat_int'] = df['treat'].astype(int)
df['commute_int'] = df['commute'].astype(int)

# 3. List of variables for the complete correlation matrix
# We include all numerical/logical variables
cols_for_corr = [
    'wait_time_num', 
    'treat_int', 
    'commute_int', 
    'trips_pool', 
    'trips_express', 
    'rider_cancellations', 
    'total_driver_payout', 
    'total_matches', 
    'total_double_matches'
]

# Create a readable subset with better names
df_corr = df[cols_for_corr].copy()
df_corr.columns = [
    'Wait Time', 
    'Treat (Exp)', 
    'Commute (Peak)', 
    'Trips Pool', 
    'Trips Express', 
    'Cancellations', 
    'Driver Payout', 
    'Matches', 
    'Double Matches'
]

# Calculate Correlation Matrix
corr_matrix = df_corr.corr()

# Plotting
plt.figure(figsize=(12, 10))

# Custom aesthetic for the heatmap
sns.set_theme(style="white")
mask = None # Could use an upper triangle mask but complete is usually better for small matrices

heatmap = sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='RdBu_r', # Red-Blue divergent for correlation
    center=0,
    vmin=-1, 
    vmax=1,
    linewidths=.5, 
    cbar_kws={"shrink": .8}
)

plt.title('Matriz de Correlación Completa - Experimento Uber Bogotá', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot
output_path = 'img/uber/uber_correlation_complete.png'
plt.savefig(output_path, dpi=300)
print(f"Correlation matrix saved to {output_path}")

# Displaying summary for my confirmation
print("\nCorrelation Matrix Summary:\n")
print(corr_matrix)
