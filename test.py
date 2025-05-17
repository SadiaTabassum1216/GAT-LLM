import util
import argparse
import torch
from model_ST_LLM import ST_LLM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", help="")
parser.add_argument("--data", type=str, default="PEMS07", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=64, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--llm_layer", type=int, default=1, help="number of LLM layers")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=48, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--checkpoint", type=str, default="./logs/xtaxi_pick/best_model.pth", help="")
parser.add_argument("--plotheatmap", type=str, default="True", help="")
args = parser.parse_args()


def main():

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"

    elif args.data == "PEMS08_36":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"

    elif args.data == "PEMS08_48":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.adjdata = "data/adj/adj_PEMS03_gs.npy"

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.adjdata = "data/adj/adj_PEMS04_gs.npy"

    elif args.data == "PEMS04_36":
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.adjdata = "data/adj/adj_PEMS04_gs.npy"

    elif args.data == "PEMS04_48":
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.adjdata = "data/adj/adj_PEMS04_gs.npy"

    elif args.data == "PEMS07":
        args.data = "data//" + args.data
        args.num_nodes = 883
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"

    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        args.num_nodes = 250
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"

    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        args.num_nodes = 250
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"

    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        args.num_nodes = 266
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        args.num_nodes = 266
        args.adjdata = "data/adj/adj_pems07.pkl"
        args.input_len = 12
        args.output_len = 12

    #     args.data = "data//" + args.data
    #     args.num_nodes = 266
    #     args.adjdata = "data/adj/adj_PEMS07_gs.npy"
    
        
    elif args.data == "CAir_AQI":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 12
        args.output_len = 12

    elif args.data == "CAir_AQI_36":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 36
        args.output_len = 36

    elif args.data == "CAir_AQI_48":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 48
        args.output_len = 48

    elif args.data == "CAir_AQI_60":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 60
        args.output_len = 60

    elif args.data == "CAir_PM":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 12
        args.output_len = 12

    elif args.data == "CAir_PM_36":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 36
        args.output_len = 36

    elif args.data == "CAir_PM_48":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 48
        args.output_len = 48

    elif args.data == "CAir_PM_60":
        args.data = "data//" + args.data
        args.num_nodes = 265
        args.input_len = 60
        args.output_len = 60

    device = torch.device(args.device)

    # _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    with open(args.adjdata, 'rb') as f:
        adj_mx = pickle.load(f)
    np.save("data/adj/adj_PEMS07_gs.npy", adj_mx)
    pre_adj = [torch.tensor(i).to(device) for i in adj_mx]

    model = ST_LLM(
        input_dim=args.input_dim,
        channels=args.channels,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        llm_layer=args.llm_layer,
        # llm_layer=args.dropout,
        U=1,  # or args.U if defined
        device=device
    )

    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print("model load successfully")

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    awmape = []
    armse = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 48 horizons(24 hours with 30min intervals), Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(awmape)))

    realy = realy.to("cpu")
    yhat1 = scaler.inverse_transform(yhat)
    yhat1 = yhat1.to("cpu")

    print(realy.shape)
    print(yhat1.shape)

    torch.save(realy, "stamt_CAir_PM_real.pt")
    torch.save(yhat1, "stamt_CAir_PM_pred.pt")


if __name__ == "__main__":
    main()
