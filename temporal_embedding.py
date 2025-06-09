import torch
import torch.nn as nn

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time  # e.g., 288 or 48
        self.features = features

        # Learned temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

        # Cycle-based embeddings projected to feature space
        self.cycle_day_proj = nn.Linear(1, features)
        self.cycle_week_proj = nn.Linear(1, features)

    def forward(self, x):
        B = x.size(0)

        # ----- Time of day -----
        day_emb = x[..., 1]  # shape: [B, S]
        day_idx = (day_emb[:, -1, :] * self.time).long().clamp(0, self.time - 1)  # [B, 1]
        time_day = self.time_day[day_idx]  # [B, 1, F]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)  # [B, F, 1]

        # ----- Day of week -----
        week_emb = x[..., 2]  # shape: [B, S]
        week_idx = week_emb[:, -1, :].long().clamp(0, 6)  # [B, 1]
        time_week = self.time_week[week_idx]  # [B, 1, F]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)  # [B, F, 1]

        # ----- Cycle-based embeddings -----
        # Normalize to [0, 1]
        norm_day = day_emb[:, -1, :].unsqueeze(-1)  # [B, 1]
        norm_week = week_emb[:, -1, :].unsqueeze(-1)  # [B, 1]

        sincos_day = torch.sin(norm_day * 2 * torch.pi / self.time) + \
                     torch.cos(norm_day * 2 * torch.pi / self.time)  # [B, 1]
        sincos_week = torch.sin(norm_week * 2 * torch.pi / 7) + \
                      torch.cos(norm_week * 2 * torch.pi / 7)  # [B, 1]

        cycle_day = self.cycle_day_proj(sincos_day).unsqueeze(-1).transpose(1, 2)  # [B, F, 1]
        cycle_week = self.cycle_week_proj(sincos_week).unsqueeze(-1).transpose(1, 2)  # [B, F, 1]

        # ----- Combine -----
        tem_emb = time_day + time_week + cycle_day + cycle_week  # [B, F, 1]
        return tem_emb
