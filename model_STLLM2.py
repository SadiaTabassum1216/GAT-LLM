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

        # 1. Temporal embeddings
        tem_emb = self.Temb(history_data)  # [B, gpt_channel, N, 1]

        # 2. Node embedding from GAT: [N, gpt_channel]
        node_emb = self.gat(
            self.node_emb.clone()
            .unsqueeze(0)
            .expand(batch_size, -1, -1),  # [B, N, gpt_channel_in]
            adj,
        )  # [B, N, gpt_channel_out]
        node_emb = node_emb.permute(0, 2, 1).unsqueeze(-1)  # [B, gpt_channel, N, 1]
        # print("GAT output:", node_emb.shape)

        # 3. Prepare input for CNN
        input_data = history_data.permute(0, 2, 1, 3).contiguous()  # [B, N, C, T]
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )  # [B, C*T, N, 1]
        input_data = self.start_conv(input_data)  # [B, gpt_channel, N, 1]

        # 4. Assert same shape
        assert (
            input_data.shape[2] == tem_emb.shape[2] == node_emb.shape[2]
        ), f"Mismatch in num_nodes dimension: input_data={input_data.shape}, tem_emb={tem_emb.shape}, node_emb={node_emb.shape}"

        # 5. Fusion
        data_st = torch.cat(
            [input_data, tem_emb, node_emb], dim=1
        )  # [B, 3*gpt_channel, N, 1]
        data_st = self.feature_fusion(data_st)  # [B, 768, N, 1]

        # 6. GPT: [B, 768, N, 1] → [B, N, 768]
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.gpt(data_st)  # [B, N, 768]

        # 7. Back to conv
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)  # [B, 768, N, 1]

        # 8. Regression
        prediction = self.regression_layer(data_st)  # [B, output_len, N, 1]

        return prediction
