import torch
import torch.nn as nn
from GAT import MultiHeadGAT
from temporal_embedding import TemporalEmbedding
from PFA import PFA


class ST_LLM(nn.Module):
    def __init__(
        self,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        llm_layer=6,
        U=2,
        device="cpu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.device = device

        if num_nodes in [170, 307, 883]:
            time = 288
        elif num_nodes in [250, 266]:
            time = 48
        else:
            time = 288

        gpt_channel = 256
        to_gpt_channel = 768

        self.Temb = TemporalEmbedding(time, gpt_channel)

        self.learned_pe = nn.Parameter(torch.empty(self.input_len, gpt_channel))  # [T, C]
        nn.init.xavier_uniform_(self.learned_pe)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))  # [N, C]
        nn.init.xavier_uniform_(self.node_emb)

        self.gat = MultiHeadGAT(
            n_heads=2,
            in_features=gpt_channel,
            out_features=gpt_channel // 2,
            dropout=0.1,
            alpha=0.2,
            merge="concat",
        )

        # 🔧 Project 3-channel input (C) to gpt_channel
        self.input_projection = nn.Conv2d(
            in_channels=self.input_dim, out_channels=gpt_channel, kernel_size=(1, 1)
        )

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )

        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U)
        self.norm = nn.LayerNorm(to_gpt_channel)
        self.residual = True

        self.regression_layer = nn.Conv2d(
            to_gpt_channel, 1, kernel_size=(1, 1)  # predict one step at a time
        )

    def param_num(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, history_data, adj):
        B, C, N, T = history_data.shape
        assert T == self.input_len and N == self.num_nodes and C == self.input_dim

        predictions = []
        current_input = history_data.clone()  # [B, C, N, T]

        # ✅ Compute node embeddings via GAT just once
        node_emb = self.gat(
            self.node_emb.unsqueeze(0).expand(B, -1, -1), adj
        )  # [B, N, gpt_channel]
        node_emb = node_emb.permute(0, 2, 1).unsqueeze(-1)  # [B, gpt_channel, N, 1]

        for step in range(self.output_len):
            # --- Temporal embedding ---
            tem_emb = self.Temb(current_input)  # [B, gpt_channel, N, 1]

            # --- Project input ---
            x = current_input.permute(0, 2, 1, 3)  # [B, N, C, T]
            x = x.reshape(B * N, C, T, 1)  # [B*N, C, T, 1]
            x = self.input_projection(x)  # [B*N, gpt_channel, T, 1]
            x = x.squeeze(-1).reshape(B, N, -1, T).permute(0, 2, 1, 3)  # [B, gpt_channel, N, T]

            # --- Add positional embedding over time ---
            pe = self.learned_pe.T.unsqueeze(0).unsqueeze(2)  # [1, gpt_channel, 1, T]
            input_seq = x + pe  # [B, gpt_channel, N, T]

            # Use only last step’s feature for fusion (can experiment with mean)
            input_seq = input_seq[:, :, :, -1:]  # [B, gpt_channel, N, 1]

            # --- Feature fusion ---
            data_st = torch.cat([input_seq, tem_emb, node_emb], dim=1)  # [B, 3*gpt_channel, N, 1]
            data_st = self.feature_fusion(data_st)  # [B, 768, N, 1]

            # --- GPT ---
            data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)  # [B, N, 768]
            gpt_input = self.norm(data_st)
            gpt_out = self.gpt(gpt_input)
            if self.residual:
                gpt_out = gpt_out + gpt_input

            # --- Regression ---
            gpt_out = gpt_out.permute(0, 2, 1).unsqueeze(-1)  # [B, 768, N, 1]
            pred = self.regression_layer(gpt_out)  # [B, 1, N, 1]
            predictions.append(pred)

            # --- Autoregressive update (repeat pred to match C=3) ---
            pred_repeated = pred.repeat(1, self.input_dim, 1, 1)  # [B, C, N, 1]
            current_input = torch.cat([current_input[:, :, :, 1:], pred_repeated], dim=-1)  # [B, C, N, T]

        return torch.cat(predictions, dim=1)  # [B, output_len, N, 1]


