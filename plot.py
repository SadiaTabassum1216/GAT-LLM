import matplotlib.pyplot as plt
import torch
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_traffic_t1_forecast(real_path: str, pred_path: str, sample_idx: int = 0, num_nodes: int = 266):
    """
    Plot the traffic speed predictions for the next time step (t+1) across all nodes.

    Args:
        real_path (str): Path to the real ground truth tensor file (.pt)
        pred_path (str): Path to the predicted tensor file (.pt)
        sample_idx (int): Batch index to visualize
        num_nodes (int): Total number of nodes (e.g., 266)
    """
    # Load the real and predicted values
    y_true = torch.load(real_path)  # [B, N, T]
    y_pred = torch.load(pred_path)  # [B, N, T]

    # Get the predicted and true values for t+1 (last time step)
    true_vals_t1 = y_true[sample_idx, :, -3].detach().cpu().numpy()  # t+1 (third-to-last step)
    pred_vals_t1 = y_pred[sample_idx, :, -3].detach().cpu().numpy()  # t+1 (third-to-last step)

    # Time axis for nodes (0 to 266)
    nodes = np.arange(0, num_nodes)

    # Plotting the graph for t+1
    plt.figure(figsize=(15, 6))
    plt.plot(nodes, pred_vals_t1, label="Predicted", color='red')
    plt.plot(nodes, true_vals_t1, label="Ground Truth", color='blue')
    plt.title("Traffic Speed Prediction at t+1")
    plt.xlabel("Node Index")
    plt.ylabel("Traffic Speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_traffic_forecast_for_node(real_path: str, pred_path: str, node_idx: int = 150, sample_idx: int = 0, num_time_steps: int = 12):
    """
    Plot the traffic speed predictions for the next 12 time steps for a specific node.

    Args:
        real_path (str): Path to the real ground truth tensor file (.pt)
        pred_path (str): Path to the predicted tensor file (.pt)
        node_idx (int): Node index to visualize (e.g., 150)
        sample_idx (int): Batch index to visualize
        num_time_steps (int): Number of time steps to visualize (default is 12)
    """
    # Load the real and predicted values
    y_true = torch.load(real_path)  # [B, N, T]
    y_pred = torch.load(pred_path)  # [B, N, T]

    # Get the predicted and true values for the next 12 time steps (t+1 to t+12)
    true_vals = y_true[sample_idx, node_idx, -num_time_steps:].detach().cpu().numpy()
    pred_vals = y_pred[sample_idx, node_idx, -num_time_steps:].detach().cpu().numpy()

    # Time axis for next 12 time steps (t+1 to t+12)
    time_steps = np.arange(1, num_time_steps + 1)

    # Plotting the graph for the selected node
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, pred_vals, label="Predicted", color='red')
    plt.plot(time_steps, true_vals, label="Ground Truth", color='blue')
    plt.title(f"Traffic Speed Prediction for Node {node_idx} (Next {num_time_steps} Time Steps)")
    plt.xlabel("Time Step (t+1 to t+12)")
    plt.ylabel("Traffic Speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage:
# plot_traffic_t1_forecast("taxi_drop_real.pt", "taxi_drop_pred.pt", sample_idx=0, num_nodes=266)
plot_traffic_forecast_for_node("taxi_drop_real.pt", "taxi_drop_pred.pt", node_idx=150, sample_idx=0, num_time_steps=12)








