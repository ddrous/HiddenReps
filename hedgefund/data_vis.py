#%%

## Load train.parquet and visualize the first few rows

import pandas as pd
data = pd.read_parquet('train.parquet')




#%%
data.head()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# Parquet is a compressed format. It loads much faster than CSV.
print("Loading data... (this might take a minute)")
df = pd.read_parquet('train.parquet')
df = df.head(10)

# ---------------------------------------------------------
# STEP A: INSPECT THE BASICS
# ---------------------------------------------------------
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Column List ---")
print(df.columns.tolist())

# Identify the target column name automatically
# We look for a column that isn't an ID, feature, or weight.
# (Usually it is 'target', 'y', or 'return')
potential_targets = [c for c in df.columns if 'feature' not in c and c not in ['id', 'code', 'sub_code', 'sub_category', 'ts_index', 'horizon', 'weight']]
target_col = potential_targets[0] if potential_targets else 'target'
print(f"\nWe are assuming the column to predict is: '{target_col}'")

# ---------------------------------------------------------
# STEP B: VISUALIZE ONE ASSET (TIME SERIES)
# ---------------------------------------------------------
# We can't plot all assets at once. Let's pick the most common 'code' to look at.
sample_code = df['code'].value_counts().idxmax()
print(f"\nVisualizing asset code: {sample_code}")

# Filter data for just this one asset
subset = df[df['code'] == sample_code]

# We also need to separate by 'horizon' because different horizons behave differently
plt.figure(figsize=(15, 6))
sns.lineplot(data=subset, x='ts_index', y=target_col, hue='horizon', palette='viridis')

plt.title(f"Target Value over Time for Asset: {sample_code}")
plt.xlabel("Time (ts_index)")
plt.ylabel(f"Target Value ({target_col})")
plt.legend(title='Horizon')
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------
# STEP C: VISUALIZE FEATURE CORRELATION
# ---------------------------------------------------------
# Do the features actually help? Let's look at one feature vs the target.
# We'll take a small sample so the scatter plot doesn't look like a blob of ink.
sample_df = df.sample(n=5000, random_state=42) 

# Let's pick the first feature in the list
feature_to_check = [c for c in df.columns if 'feature' in c][0]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample_df, x=feature_to_check, y=target_col, alpha=0.5)

plt.title(f"Relationship: {feature_to_check} vs {target_col} (Sample of 5000 rows)")
plt.xlabel(feature_to_check)
plt.ylabel(target_col)
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------
# STEP D: UNDERSTAND THE WEIGHTS
# ---------------------------------------------------------
# This competition uses a weighted score. Let's see the distribution of weights.
plt.figure(figsize=(10, 4))
sns.histplot(df['weight'], bins=50, kde=True)
plt.title("Distribution of Weights (Are all rows equally important?)")
plt.xlabel("Weight")
plt.show()
# %%


# 9000 entries

data = {"code_sub_code_sub_cat": 3D_matrix,
        "code_sub_code_sub_cat": 3D_matrix,
}

3D_matrix: ts_index x features x horizon

Desired output:
Tensor shape: 9000 x 29 x 91 x 4
List of ids: 9000





Transformer_input: batch_size x 29 x 90*4 -> batch_size x 29 X 1*4


# ---
x0 , x1, x2, ... xn, xn+1

x0 , x1, x2, ... xn, xn+1, xn+2, ..., xn+n


(N, C_in) - (N, C_out)