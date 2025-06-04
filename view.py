import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the adjacency matrix
with open('data/adj/adj_pems07.pkl', 'rb') as f:
    adj_mx = pickle.load(f)

# Plot using seaborn heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(adj_mx, cmap='coolwarm', square=True, cbar_kws={'label': 'Edge Weight'})
plt.title('Adjacency Matrix Heatmap')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()
