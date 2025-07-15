import numpy as np

# Load the dataset
data = np.load('data/taxi_drop/taxi_drop/train.npz')

# Access traffic data
traffic_data = data['y']

# View the first few records (let's display the first 5)
print("First 5 records of traffic data:", traffic_data[:5])
