import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    deg = adj.sum(1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return norm_adj


def analyze_adjacency(adj):
    print("Top-left 5x5 values:")
    print(adj[:5, :5])

    print(f"\nShape: {adj.shape}")
    print(f"Min: {adj.min().item():.4f}, Max: {adj.max().item():.4f}, Mean: {adj.mean().item():.4f}")
    print(f"Non-zero count: {torch.count_nonzero(adj).item()}, Total: {adj.numel()}")
    print(f"Sparsity: {100.0 * torch.sum(adj == 0).item() / adj.numel():.2f}%")

    # Export to CSV
    np.savetxt("adj_matrix.csv", adj.numpy(), delimiter=",")
    print("Adjacency matrix saved to adj_matrix.csv")


    visualize_heatmap(adj, title="Adjacency Matrix Heatmap")


def visualize_heatmap(matrix, title="Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pkl_path = "data/taxi_drop/taxi_drop/adj_mx.pkl"

    with open(pkl_path, 'rb') as f:
        adj_mx = pickle.load(f)

    if isinstance(adj_mx, (list, tuple)):
        adj = adj_mx[0]
    else:
        adj = adj_mx

    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj, dtype=torch.float32)

    analyze_adjacency(adj)
