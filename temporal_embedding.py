import torch
import torch.nn as nn

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time    # 288 or 48
        
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        day_idx = (day_emb[:, -1, :] * self.time).long().clamp(0, self.time - 1)
        time_day = self.time_day[day_idx]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        week_idx = (week_emb[:, -1, :]).long().clamp(0, 6)
        time_week = self.time_week[week_idx]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb
