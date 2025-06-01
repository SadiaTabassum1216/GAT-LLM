import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/PEMS07/PEMS07.npz')
traffic_data = data['data']

print("Shape of pems traffic_data:", traffic_data.shape)

sensor_id = 0  # Can try 0 to 882
timeseries = traffic_data[:, sensor_id, 0]


data = np.load('data/taxi_drop/taxi_drop/train.npz')
# print("Keys in the loaded data:", data.files)

traffic_data = data['y']

print("Shape of taxi traffic_data:", traffic_data.shape)
# sensor_id = 3  # Can try 0 to 265 
# timeseries = traffic_data[:, sensor_id, 0]




