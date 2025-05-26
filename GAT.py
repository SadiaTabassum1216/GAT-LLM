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
