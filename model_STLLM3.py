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
        U=1,
        device="cpu",
    ):
        super().__init__()
        self.attn_implementation = "eager"

        # input length = last 12 time steps, output length= next 12 time steps

        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.device = device

        # this is chosen according to the dataset
        if num_nodes == 170 or num_nodes == 307 or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        else:
            time = 288

        gpt_channel = 256
        to_gpt_channel = 768

        self.Temb = TemporalEmbedding(time, gpt_channel)

        # Node embedding
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)
        print(f"node_emb: {self.node_emb.shape}")  # Should be [num_nodes, gpt_channel]

        # GCN layer for graph-aware node embeddings
        # self.gcn = GraphConvolution(gpt_channel, gpt_channel)
        # self.gat = GraphAttention(gpt_channel, gpt_channel)
        self.gat = MultiHeadGAT(
            n_heads=2,
            in_features=gpt_channel,
            out_features=gpt_channel // 2,  # so total output dim = gpt_channel
            dropout=0.1,
            alpha=0.2,
            merge="concat",
        )

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        # embedding layer
        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U)

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )

        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel * 3, self.output_len, kernel_size=(1, 1)
        )

    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data, adj):
        batch_size, inputdim, num_nodes, input_len = history_data.shape
        device = history_data.device

        # Temporal embedding for the last known time step
        tem_emb = self.Temb(history_data[:, :, :, -1:])  # [B, gpt_channel, N, 1]

        # Node embeddings using GAT
        node_emb = self.gat(
            self.node_emb.clone().unsqueeze(0).expand(batch_size, -1, -1), adj
        )  # [B, N, gpt_channel]
        node_emb = node_emb.permute(0, 2, 1).unsqueeze(-1)  # [B, gpt_channel, N, 1]

        # Initialize the input buffer with the original history
        current_input = history_data.clone()  # [B, C, N, T]
        outputs = []

        for step in range(self.output_len):
            # Slice the most recent `input_len` time steps
            x = current_input[:, :, :, -self.input_len:]  # [B, C, N, T]

            # Reshape and encode temporal input
            input_data = x.permute(0, 2, 1, 3).contiguous()  # [B, N, C, T]
            input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)  # [B, C*T, N, 1]
            input_data = self.start_conv(input_data)  # [B, gpt_channel, N, 1]

            # Optional: Add step-wise temporal position embedding
            # if hasattr(self, 'temporal_pos'):
            #     tem_emb = tem_emb + self.temporal_pos[step].view(1, -1, 1, 1)

            # Feature fusion
            fusion = torch.cat([input_data, tem_emb, node_emb], dim=1)  # [B, 3*gpt_channel, N, 1]
            fusion = self.feature_fusion(fusion)  # [B, 768, N, 1]

            # GPT forward pass
            gpt_input = fusion.permute(0, 2, 1, 3).squeeze(-1)  # [B, N, 768]
            gpt_out = self.gpt(gpt_input)  # [B, N, 768]
            gpt_out = gpt_out.permute(0, 2, 1).unsqueeze(-1)  # [B, 768, N, 1]

            # Predict one future step
            pred = self.regression_layer(gpt_out)  # [B, 1, N, 1]
            pred_step = pred.squeeze(1).unsqueeze(1)  # [B, 1, N, 1]

            outputs.append(pred_step)
            current_input = torch.cat([current_input, pred_step], dim=3)  # update input buffer

        # Concatenate predictions
        prediction = torch.cat(outputs, dim=1)  # [B, output_len, N, 1]
        return prediction

