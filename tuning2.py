import os
import time
import optuna
import pickle
import numpy as np
import torch
import random
import torch.nn as nn
from util import load_dataset, MAE_torch
from model_ST_LLM import ST_LLM
from ranger21 import Ranger
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "taxi_drop"  # Update as needed

def seed_it(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_adj(adj):
    degree = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)

def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

def get_optimizer(name, params, lr, weight_decay):
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "Ranger":
        return Ranger(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def train_and_evaluate(trial):
    # Optimization Hyperparameters
    lrate = trial.suggest_loguniform("lrate", 1e-5, 1e-2)
    wdecay = trial.suggest_loguniform("wdecay", 1e-6, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Ranger"])
    clip = trial.suggest_float("clip", 0.1, 10.0)

    # Model Architecture
    llm_layer = trial.suggest_int("llm_layer", 1, 6)
    U = trial.suggest_int("U", 1, 4)
    
    # channels = trial.suggest_categorical("channels", [32, 64, 128])
    # gpt_layers = trial.suggest_int("gpt_layers", 4, 12)
    # dropout = trial.suggest_float("dropout", 0.0, 0.5)
    # hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    # activation_name = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"])
    

    # Data-related
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    input_len = trial.suggest_int("input_len", 6, 24)
    output_len = trial.suggest_int("output_len", 6, 24)
    num_nodes = 266  # fixed per dataset

    data_path = f"data//{DATA}//{DATA}"
    dataloader = load_dataset(data_path, batch_size, batch_size, batch_size)
    scaler = dataloader["scaler"]

    adj_path = os.path.join(data_path, "adj_mx.pkl")
    with open(adj_path, "rb") as f:
        adj_mx = pickle.load(f)
    adj = normalize_adj(torch.tensor(adj_mx, dtype=torch.float32))

    model = ST_LLM(
        input_dim=3,
        channels=64,
        num_nodes=num_nodes,
        input_len=input_len,
        output_len=output_len,
        llm_layer=llm_layer,
        U=U,
        device=DEVICE,
        # dropout=dropout,
        # gpt_layers=gpt_layers,
        # hidden_dim=hidden_dim,
        # activation=get_activation(activation_name),
    ).to(DEVICE)

    optimizer = get_optimizer(optimizer_name, model.parameters(), lrate, wdecay)
    loss_fn = MAE_torch

    model.train()
    best_loss = float("inf")

    for x, y in dataloader["train_loader"].get_iterator():
        trainx = torch.Tensor(x).to(DEVICE).transpose(1, 3)
        trainy = torch.Tensor(y).to(DEVICE).transpose(1, 3)
        real = trainy.permute(0, 3, 2, 1).contiguous()

        optimizer.zero_grad()
        output = model(trainx, adj)
        predict = scaler.inverse_transform(output)
        loss = loss_fn(predict, real, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        best_loss = min(best_loss, loss.item())
        break  # fast tuning

    return best_loss

if __name__ == "__main__":
    seed_it(42)
    study = optuna.create_study(direction="minimize", study_name="stllm_full_hparam_tuning")
    study.optimize(train_and_evaluate, n_trials=500, timeout=3600)

    print("✅ Best trial:")
    print(f"  Value (loss): {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    