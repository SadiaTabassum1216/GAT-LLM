import os
import time
import optuna
import pickle
import numpy as np
import torch
import random
from util import load_dataset, MAE_torch, MAPE_torch, RMSE_torch, WMAPE_torch
from model_ST_LLM import ST_LLM
from ranger21 import Ranger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "taxi_drop"  # change if needed

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

def train_and_evaluate(trial):
    # Sample hyperparameters
    lrate = trial.suggest_loguniform("lrate", 1e-5, 1e-2)
    wdecay = trial.suggest_loguniform("wdecay", 1e-6, 1e-2)
    llm_layer = trial.suggest_int("llm_layer", 1, 4)
    U = trial.suggest_int("U", 1, 4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    num_nodes = 266
    input_len = 12
    output_len = 12
    input_dim = 3

    data_path = f"data//{DATA}//{DATA}"
    dataloader = load_dataset(data_path, batch_size, batch_size, batch_size)
    scaler = dataloader["scaler"]

    adj_path = os.path.join("data", DATA, DATA, "adj_mx.pkl")
    with open(adj_path, "rb") as f:
        adj_mx = pickle.load(f)
    adj = torch.tensor(adj_mx, dtype=torch.float32)
    adj = normalize_adj(adj)

    model = ST_LLM(
        input_dim=input_dim,
        channels=64,
        num_nodes=num_nodes,
        input_len=input_len,
        output_len=output_len,
        llm_layer=llm_layer,
        U=U,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = Ranger(model.parameters(), lr=lrate, weight_decay=wdecay)
    loss_fn = MAE_torch
    clip = 5

    model.train()
    best_loss = float("inf")

    for x, y in dataloader["train_loader"].get_iterator():
        trainx = torch.Tensor(x).to(DEVICE).transpose(1, 3)
        trainy = torch.Tensor(y).to(DEVICE).transpose(1, 3)
        real = trainy.permute(0, 3, 2, 1).contiguous()

        optimizer.zero_grad()
        output = model(trainx, adj)
        predict = scaler.inverse_transform(output)
        real = real
        loss = loss_fn(predict, real, 0.0)
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        best_loss = min(best_loss, loss.item())
        break  # one batch per trial for fast tuning

    return best_loss  # minimize this


if __name__ == "__main__":
    seed_it(42)
    study = optuna.create_study(direction="minimize", study_name="stllm_hparam_tuning")
    study.optimize(train_and_evaluate, n_trials=50, timeout=3600)

    print("✅ Best trial:")
    trial = study.best_trial
    print(f"  Value (loss): {trial.value:.4f}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
