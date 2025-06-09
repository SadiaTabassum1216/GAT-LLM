import util
import argparse
import torch
import numpy as np
import pickle
# from model_ST_LLM import ST_LLM
from model_STLLM2 import ST_LLM

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--data", type=str, default="PEMS07")
parser.add_argument("--input_dim", type=int, default=3)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--num_nodes", type=int, default=170)
parser.add_argument("--llm_layer", type=int, default=1)
parser.add_argument("--input_len", type=int, default=12)
parser.add_argument("--output_len", type=int, default=48)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)

parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()


def normalize_adj(adj):
    degree = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)


def main():
    dataset_config = {
        "taxi_drop": {
            "path": "data/taxi_drop/taxi_drop",
            "num_nodes": 266,
            "input_len": 12,
            "output_len": 12,
            "adj_path": "data/taxi_drop/taxi_drop/adj_mx.pkl",
        },
        "taxi_pick": {
            "path": "data/taxi_pick/taxi_pick",
            "num_nodes": 266,
            "input_len": 12,
            "output_len": 12,
            "adj_path": "data/taxi_pick/taxi_pick/adj_mx.pkl",
        },
        "PEMS07": {
            "path": "data/PEMS07",
            "num_nodes": 883,
            "adj_path": "data/adj/adj_PEMS07_gs.npy",
            "input_dim": 1,
        },
        "PEMS08": {
            "path": "data/PEMS08",
            "num_nodes": 170,
            "adj_path": "data/adj/adj_PEMS08_gs.npy",
            "input_dim": 1,
        },
        "PEMS08_36": {
            "path": "data/PEMS08_36",
            "num_nodes": 170,
            "adj_path": "data/adj/adj_PEMS08_gs.npy",
            "input_dim": 1,
        },
        "PEMS08_48": {
            "path": "data/PEMS08_48",
            "num_nodes": 170,
            "adj_path": "data/adj/adj_PEMS08_gs.npy",
        },
        "PEMS03": {
            "path": "data/PEMS03",
            "num_nodes": 358,
            "adj_path": "data/adj/adj_PEMS03_gs.npy",
            "input_dim": 1,
        },
        "PEMS04": {
            "path": "data/PEMS04",
            "num_nodes": 307,
            "adj_path": "data/adj/adj_PEMS04_gs.npy",
            "input_dim": 1,
        },
        "PEMS04_36": {
            "path": "data/PEMS04_36",
            "num_nodes": 307,
            "adj_path": "data/adj/adj_PEMS04_gs.npy",
            "input_dim": 1,
        },
        "PEMS04_48": {
            "path": "data/PEMS04_48",
            "num_nodes": 307,
            "adj_path": "data/adj/adj_PEMS04_gs.npy",
            "input_dim": 1,
        },
        "bike_drop": {
            "path": "data/bike_drop",
            "num_nodes": 250,
            "adj_path": "data/adj/adj_PEMS07_gs.npy",
        },
        "bike_pick": {
            "path": "data/bike_pick",
            "num_nodes": 250,
            "adj_path": "data/adj/adj_PEMS07_gs.npy",
        },
        "CAir_AQI": {
            "path": "data/CAir_AQI",
            "num_nodes": 265,
            "input_len": 12,
            "output_len": 12,
        },
        "CAir_AQI_36": {
            "path": "data/CAir_AQI_36",
            "num_nodes": 265,
            "input_len": 36,
            "output_len": 36,
        },
        "CAir_AQI_48": {
            "path": "data/CAir_AQI_48",
            "num_nodes": 265,
            "input_len": 48,
            "output_len": 48,
        },
        "CAir_AQI_60": {
            "path": "data/CAir_AQI_60",
            "num_nodes": 265,
            "input_len": 60,
            "output_len": 60,
        },
        "CAir_PM": {
            "path": "data/CAir_PM",
            "num_nodes": 265,
            "input_len": 12,
            "output_len": 12,
        },
        "CAir_PM_36": {
            "path": "data/CAir_PM_36",
            "num_nodes": 265,
            "input_len": 36,
            "output_len": 36,
        },
        "CAir_PM_48": {
            "path": "data/CAir_PM_48",
            "num_nodes": 265,
            "input_len": 48,
            "output_len": 48,
        },
        "CAir_PM_60": {
            "path": "data/CAir_PM_60",
            "num_nodes": 265,
            "input_len": 60,
            "output_len": 60,
        },
    }

    if args.data not in dataset_config:
        raise ValueError(f"Unsupported dataset: {args.data}")

    config = dataset_config[args.data]
    args.path = config["path"]
    # args.data = config["path"]
    args.num_nodes = config["num_nodes"]
    args.input_len = config.get("input_len", args.input_len)
    args.output_len = config.get("output_len", args.output_len)
    args.adjdata = config.get("adj_path", f"data/adj/adj_{args.path}.npy")

    device = torch.device(args.device)

    with open(args.adjdata, "rb") as f:
        adj_mx = pickle.load(f)
    adj_tensor = torch.tensor(
        adj_mx[0] if isinstance(adj_mx, list) else adj_mx, dtype=torch.float32
    ).to(device)
    
    adj_tensor = normalize_adj(adj_tensor)

    model = ST_LLM(
        input_dim=args.input_dim,
        channels=args.channels,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        llm_layer=args.llm_layer,
        U=1,
        device=device,
    )
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if "node_emb" not in checkpoint:
        raise KeyError(
            "--------------------Trained checkpoint does not contain node_emb. Please ensure the model was saved correctly."
        )

    model.load_state_dict(checkpoint, strict=True)
    print("Model load successfully...")
    model.eval()

    dataloader = util.load_dataset(
        args.path, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]
    realy = torch.Tensor(dataloader["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]

    outputs = []
    for _, (x, _) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx, adj_tensor).transpose(1, 3)
        outputs.append(preds.squeeze())

    # yhat = torch.cat(outputs, dim=0)[:realy.size(0)]
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae, amape, armse, awmape = [], [], [], []
    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        mae, mape, rmse, wmape = util.metric(pred, real)
        print(
            f"Horizon {i+1:02d} | MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}, WMAPE: {wmape:.4f}"
        )
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)
        awmape.append(wmape)

    log = "\nOn average over 48 horizons(24 hours with 30min intervals), \nTest MAE: {:.4f}, \nTest MAPE: {:.4f}, \nTest RMSE: {:.4f}, \nTest WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(awmape)))

    realy = realy.to("cpu")
    yhat1 = scaler.inverse_transform(yhat)
    yhat1 = yhat1.to("cpu")

    # print(f"Saving results to {args.data}_real.pt and {args.data}_pred.pt")

    torch.save(realy, f"{args.data}_real.pt")
    torch.save(yhat1, f"{args.data}_pred.pt")


if __name__ == "__main__":
    main()
