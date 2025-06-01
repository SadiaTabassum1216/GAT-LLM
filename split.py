import numpy as np
import os

def split_pems07(npz_path, save_dir, history_len=12, predict_len=12, split_ratio=(0.6, 0.2, 0.2)):
    data = np.load(npz_path)['data']  # shape: (T, N, 1)
    T = data.shape[0]
    num_samples = T - history_len - predict_len

    x, y = [], []
    for t in range(num_samples):
        x.append(data[t:t+history_len])
        y.append(data[t+history_len:t+history_len+predict_len])

    x = np.array(x)  # (num_samples, history_len, N, 1)
    y = np.array(y)  # (num_samples, predict_len, N, 1)

    # Split into train/val/test
    train_end = int(num_samples * split_ratio[0])
    val_end = train_end + int(num_samples * split_ratio[1])

    datasets = {
        "train": (x[:train_end], y[:train_end]),
        "val":   (x[train_end:val_end], y[train_end:val_end]),
        "test":  (x[val_end:], y[val_end:])
    }

    # Save
    os.makedirs(save_dir, exist_ok=True)
    for split in datasets:
        np.savez_compressed(os.path.join(save_dir, f"{split}.npz"), x=datasets[split][0], y=datasets[split][1])
        print(f"Saved {split}.npz: {datasets[split][0].shape}, {datasets[split][1].shape}")

# Example usage
split_pems07("data/PEMS07/PEMS07.npz", "data/PEMS07/PEMS07")
