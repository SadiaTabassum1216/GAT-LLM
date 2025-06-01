# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class GATLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True, use_residual=True, use_layernorm=True):
#         super(GATLayer, self).__init__()
#         self.concat = concat
#         self.use_residual = use_residual
#         self.use_layernorm = use_layernorm

#         self.W = nn.Linear(in_features, out_features, bias=False)
#         self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)

#         self.leakyrelu = nn.LeakyReLU(alpha)
#         self.dropout = nn.Dropout(dropout)

#         if use_residual:
#             self.res_connection = nn.Linear(in_features, out_features, bias=False)

#         if use_layernorm:
#             self.norm = nn.LayerNorm(out_features)

#     def forward(self, h, adj):
#         """
#         h: [B, N, F]
#         adj: [B, N, N]
#         """
#         if h.dim() == 2:  # handle unbatched input
#             h = h.unsqueeze(0)
#             adj = adj.unsqueeze(0)

#         B, N, _ = h.shape
#         Wh = self.W(h)  # [B, N, out_features]

#         Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, out_features]
#         Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, out_features]
#         a_input = torch.cat([Wh_i, Wh_j], dim=-1)     # [B, N, N, 2*out_features]

#         e = self.leakyrelu(self.attn_fc(a_input).squeeze(-1))  # [B, N, N]

#         # Mask non-neighbors
#         e = torch.where(adj > 0, e, torch.full_like(e, -9e15))

#         attention = F.softmax(e, dim=-1)  # [B, N, N]
#         attention = self.dropout(attention)

#         h_prime = torch.bmm(attention, Wh)  # [B, N, out_features]

#         if self.use_residual:
#             h_res = self.res_connection(h)
#             h_prime = h_prime + h_res  # residual connection

#         if self.concat:
#             h_prime = F.elu(h_prime)

#         if self.use_layernorm:
#             h_prime = self.norm(h_prime)

#         return h_prime


# class MultiHeadGAT(nn.Module):
#     def __init__(self, n_heads, in_features, out_features, dropout=0.1, alpha=0.2, merge='concat'):
#         super(MultiHeadGAT, self).__init__()
#         self.merge = merge

#         self.heads = nn.ModuleList([
#             GATLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=(merge == 'concat'))
#             for _ in range(n_heads)
#         ])

#     def forward(self, h, adj):
#         """
#         h: [B, N, F]
#         adj: [B, N, N]
#         """
#         head_outputs = [head(h, adj) for head in self.heads]

#         if self.merge == 'concat':
#             return torch.cat(head_outputs, dim=-1)  # [B, N, out_features * n_heads]
#         else:
#             return torch.mean(torch.stack(head_outputs, dim=0), dim=0)  # [B, N, out_features]


import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat  # Whether to use ELU after attention

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # (N, out_features)
        N = Wh.size(0)

        # Compute pairwise attention scores
        Wh_repeat_i = Wh.repeat(1, N).view(N * N, -1)
        Wh_repeat_j = Wh.repeat(N, 1)
        a_input = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=1).view(N, N, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # mask non-edges
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)  # ELU activation for hidden layers
        else:
            return h_prime  # No activation for output layer


class MultiHeadGAT(nn.Module):
    def __init__(self, n_heads, in_features, out_features, dropout=0.1, alpha=0.2, merge='concat'):
        super(MultiHeadGAT, self).__init__()
        self.merge = merge  # 'concat' or 'average'

        self.heads = nn.ModuleList([
            GATLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(n_heads)
        ])

    def forward(self, h, adj):
        head_outputs = [attn_head(h, adj) for attn_head in self.heads]

        if self.merge == 'concat':
            return torch.cat(head_outputs, dim=1)  # [N, out_features * n_heads]
        else:
            return torch.mean(torch.stack(head_outputs), dim=0)  # [N, out_features]