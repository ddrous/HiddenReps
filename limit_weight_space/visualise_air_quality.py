#%%
# Load air_quality.csv and plot NOx vs O3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Set plotting style
sns.set(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

# Load the dataset
data_path = "air_quality.csv"  # Update this path if needed
df = pd.read_csv(data_path)

## Normalise the dataframe (Optional, but can help with visualization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['PT08.S3.NOx.', 'PT08.S5.O3.']] = scaler.fit_transform(df[['PT08.S3.NOx.', 'PT08.S5.O3.']])


# Display the first few rows to understand the structure
print(df.head())

# Plot NOx vs O3
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PT08.S5.O3.', y='PT08.S3.NOx.', alpha=0.6)
plt.title('Air Quality - O3 vs NOx')
plt.xlabel('O3')
plt.ylabel('NOx')
plt.grid(True, alpha=0.3)
plt.show()