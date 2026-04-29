import torch
import torch.nn as nn


class NHiTSBlock(nn.Module):
    def __init__(self, input_size, n_features, horizon, pool_size, mlp_units=256, dropout=0.2):
        super().__init__()
        self.pool_size = pool_size
        pooled_input = (input_size // pool_size) * n_features

        self.mlp = nn.Sequential(
            nn.Linear(pooled_input, mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, mlp_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.backcast_head = nn.Linear(mlp_units, input_size)
        self.forecast_head = nn.Linear(mlp_units, horizon)

    def forward(self, x):
        x_pooled = x.unfold(1, self.pool_size, self.pool_size).max(dim=-1).values
        x_flat = x_pooled.reshape(x_pooled.size(0), -1)
        h = self.mlp(x_flat)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast, h  # also return hidden state


class NHiTS(nn.Module):
    def __init__(self, input_size=72, n_features=7, horizon=12, mlp_units=256, dropout=0.2):
        super().__init__()
        pool_sizes = [input_size // 4, input_size // 8, 1]

        self.blocks = nn.ModuleList([
            NHiTSBlock(input_size, n_features, horizon, ps, mlp_units, dropout)
            for ps in pool_sizes
        ])

        # Classification head takes concatenated hidden states from all blocks
        self.classifier = nn.Sequential(
            nn.Linear(mlp_units * len(pool_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # raw logit — BCEWithLogitsLoss handles sigmoid
        )

    def forward(self, x):
        residual = x.clone()
        forecast_total = torch.zeros(
            x.size(0),
            self.blocks[0].forecast_head.out_features,
            device=x.device
        )
        hidden_states = []

        for block in self.blocks:
            backcast, forecast, h = block(residual)
            residual = residual - backcast.unsqueeze(-1).expand_as(residual)
            forecast_total = forecast_total + forecast
            hidden_states.append(h)

        # Concatenate hidden states for classification
        combined = torch.cat(hidden_states, dim=-1)
        hypo_logit = self.classifier(combined).squeeze(-1)

        return forecast_total, hypo_logit