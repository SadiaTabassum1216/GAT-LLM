import matplotlib.pyplot as plt
import torch
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_single_forecast(real_path: str, pred_path: str, node_idx: int = 0, sample_idx: int = 0):
    """
    Plot a single time-series forecast vs ground truth like publication-quality graph.

    Args:
        real_path (str): Path to real .pt tensor file [B, N, T]
        pred_path (str): Path to predicted .pt tensor file [B, N, T]
        node_idx (int): Node index to visualize
        sample_idx (int): Batch index to visualize
    """
    import matplotlib.pyplot as plt
    import torch
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    y_true = torch.load(real_path, weights_only=True)
    y_pred = torch.load(pred_path, weights_only=True)

    true_vals = y_true[sample_idx, node_idx, :].detach().cpu().numpy()
    pred_vals = y_pred[sample_idx, node_idx, :].detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(pred_vals, label="Our Model", color='red')
    plt.plot(true_vals, label="Ground Truth", color='blue')
    plt.xlabel("Time (hour)")
    plt.ylabel("Traffic speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_full_24h_forecast(real_path: str, pred_path: str, node_idx: int = 0, sample_idx: int = 0, scaler=None, interval_min: int = 5):
    """
    Plot full 24-hour forecast (or output_len) as one continuous line chart.

    Args:
        real_path (str): Path to real ground truth tensor file (.pt)
        pred_path (str): Path to predicted tensor file (.pt)
        node_idx (int): Node index to visualize
        sample_idx (int): Batch index to visualize
        scaler (object, optional): Scaler with inverse_transform method
        interval_min (int): Minutes per timestep interval (default is 5min)
    """
    y_true = torch.load(real_path, weights_only=True)  # [B, N, T]
    y_pred = torch.load(pred_path, weights_only=True)  # [B, N, T]

    if scaler is not None:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

    true_vals = y_true[sample_idx, node_idx, :].detach().cpu().numpy()
    pred_vals = y_pred[sample_idx, node_idx, :].detach().cpu().numpy()

    T = len(true_vals)
    time_axis = np.arange(0, T * interval_min, interval_min) / 60  # in hours

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, pred_vals, label="Our Model", color='red')
    plt.plot(time_axis, true_vals, label="Ground Truth", color='blue')
    plt.xlabel("Time (hours)")
    plt.ylabel("Traffic Speed")
    plt.title("24-hour Traffic Speed Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
# Example usage:
plot_full_24h_forecast("taxi_drop_real.pt", "taxi_drop_pred.pt", node_idx=0, sample_idx=0, scaler=None, interval_min=30)
