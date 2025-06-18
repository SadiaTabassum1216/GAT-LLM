import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random

from model_ST_LLM import ST_LLM
# from model_STLLM2 import ST_LLM
from ranger21 import Ranger
from tqdm import tqdm
import pickle

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--data", type=str, default="taxi_pick")
parser.add_argument("--input_dim", type=int, default=3)  # 3 for bike and taxi, 1 for PEMS07
parser.add_argument("--num_nodes", type=int, default=250)
parser.add_argument("--input_len", type=int, default=12)
parser.add_argument("--output_len", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lrate", type=float, default=1e-3)
parser.add_argument("--llm_layer", type=int, default=1)
parser.add_argument("--U", type=int, default=2)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--gpt_layers", type=int, default=6)
parser.add_argument("--wdecay", type=float, default=0.0001)
parser.add_argument("--save", type=str, default="./logs/x")
parser.add_argument("--es_patience", type=int, default=100)
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint")

args = parser.parse_args()


class trainer:
    def __init__(
        self,
        scaler,
        lrate,
        wdecay,
        input_dim,
        num_nodes,
        input_len,
        output_len,
        llm_layer,
        U,
        device,
        adj,
    ):
        self.model = ST_LLM(
            input_dim=input_dim,
            channels=64,
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            llm_layer=llm_layer,
            U=U,
            device=device,
        )
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        self.adj = adj
        print("The number of parameters: {}".format(self.model.param_num()))
        print("node_emb shape at init:", self.model.node_emb.shape)
        print(self.model)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, self.adj)  # [B, output_len, N, 1]
        predict = self.scaler.inverse_transform(output)  # [B, output_len, N, 1]
        real = real_val  # already [B, output_len, N, 1]
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mae = util.MAE_torch(predict, real, 0.0).item()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, wmape


    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, self.adj)  # [B, output_len, N, 1]
            predict = self.scaler.inverse_transform(output)
            real = real_val  # already [B, output_len, N, 1]
            loss = self.loss(predict, real, 0.0)
            mae = util.MAE_torch(predict, real, 0.0).item()
            mape = util.MAPE_torch(predict, real, 0.0).item()
            rmse = util.RMSE_torch(predict, real, 0.0).item()
            wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse, wmape


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def normalize_adj(adj):
    degree = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)


def main():
    seed_it(6666)
    data = args.data
    adj_path = os.path.join("data", args.data, args.data, "adj_mx.pkl")
    # adj_path = os.path.join("data", args.data, "adj_pems07.pkl")

    with open(adj_path, "rb") as f:
        adj_mx = pickle.load(f)

    adj = torch.tensor(adj_mx, dtype=torch.float32)
    adj = normalize_adj(adj)

    if args.data == "bike_drop":
        args.data = "data/bike_drop/" + args.data
        args.num_nodes = 250
    elif args.data == "bike_pick":
        args.data = "data//bike_pick//" + args.data
        args.num_nodes = 250
    elif args.data == "taxi_drop":
        args.data = "data//taxi_drop//" + args.data
        args.num_nodes = 266
    elif args.data == "taxi_pick":
        args.data = "data//taxi_pick//" + args.data
        args.num_nodes = 266
    elif args.data == "PEMS07":
        args.data = "data/PEMS07/" + args.data
        args.num_nodes = 883

    device = torch.device(args.device)
    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    path = args.save + data + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler,
        args.lrate,
        args.wdecay,
        args.input_dim,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.llm_layer,
        args.U,
        device,
        adj,
    )

    best_loss = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            engine.model.load_state_dict(checkpoint["model_state_dict"])
            engine.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            print(
                f"Resumed from epoch {checkpoint['epoch']} with best loss {best_loss:.4f}"
            )
        else:
            engine.model.load_state_dict(checkpoint)
            print("Loaded model weights only (no optimizer or epoch info).")

    print("start training...", flush=True)

    total_epochs = args.epochs

    for i in range(start_epoch, total_epochs + 1):
    # for i in range(start_epoch, args.epochs + 1):
        train_loss, train_mae, train_mape, train_rmse, train_wmape = [], [], [], [], []
        t1 = time.time()

        for iter, (x, y) in enumerate(
            tqdm(dataloader["train_loader"].get_iterator(), desc=f"Epoch {i}")
        ):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            # metrics = engine.train(trainx, trainy[:, 0, :, :])
            real = trainy[:, :, :, :].permute(0, 3, 2, 1).contiguous()  # [B, output_len, N, 1]
            metrics = engine.train(trainx, real)

            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            train_wmape.append(metrics[4])
            if iter % args.print_every == 0:
                print(
                    f"\n  Iter {iter:03d} | Loss: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, MAPE: {metrics[2]:.4f}, RMSE: {metrics[3]:.4f}, WMAPE: {metrics[4]:.4f}"
                )

        t2 = time.time()
        avg_loss = np.mean(train_loss)
        print(
            f"Epoch {i:03d} completed in {t2 - t1:.2f} seconds. Avg Loss: {avg_loss:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = i
            epochs_since_best = 0
            checkpoint_path = os.path.join(path, f"best_model.pth")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": engine.model.state_dict(),
                    "optimizer_state_dict": engine.optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )
            print(
                f"Model saved to {checkpoint_path} at epoch {i} with loss {avg_loss:.4f}"
            )
        else:
            epochs_since_best += 1

        if i % 10 == 0:
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": engine.model.state_dict(),
                    "optimizer_state_dict": engine.optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(path, f"checkpoint_epoch_{i}.pth"),
            )

        if epochs_since_best >= args.es_patience and i >= 200:
            print("Early stopping triggered.")
            break

    print(
        "\nTraining complete. Best model from epoch {} with loss {:.4f}".format(
            best_epoch, best_loss
        )
    )


if __name__ == "__main__":
    # bottleneck.main()
    torch.cuda.empty_cache()
    t1 = time.time()
    main()
    t2 = time.time()
    # print("Total time spent: {:.4f} seconds".format(t2 - t1))
    time_spent = t2 - t1
    hours = int(time_spent // 3600)
    minutes = int((time_spent % 3600) // 60)
    seconds = int(time_spent % 60)
    print(
        "\nTotal time spent: {:2d} hours, {:02d} minutes, {:.4f} seconds".format(
            hours, minutes, seconds
        )
    )
