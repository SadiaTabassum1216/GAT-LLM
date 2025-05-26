import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Basic Graph Convolution Layer (GCN) supporting 2D and 3D inputs.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        # x: [N, F] or [B, N, F]
        # adj: [N, N] (shared across batch)
        if x.dim() == 2:
            # Single sample: [N, F]
            x = torch.matmul(adj, x)
            x = self.linear(x)  # [N, F_out]
        elif x.dim() == 3:
            # Batched input: [B, N, F]
            x = torch.matmul(adj, x.transpose(0, 1)).transpose(0, 1)  # [B, N, F]
            x = self.linear(x)  # [B, N, F_out]
        else:
            raise ValueError("Input x must be 2D or 3D")
        return x