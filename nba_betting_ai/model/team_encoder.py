import torch
from torch import nn


class TeamEncoder(nn.Module):
    def __init__(self, team_count: int, feature_cols: int, embedding_dim: int | None, hidden_dim: int, no_layers: int):
        """
        Model for encoding the team data. It can include an embedding layer for the team index.

        Params:
            - team_count (int): number of teams in the dataset
            - feature_cols (int): number of input features per team
            - embedding_dim (int): dimension of the team embedding. Can be None or 0 if no embedding is used.
            - hidden_dim (int): dimension of the hidden layer in the encoder
            - no_layers (int): number of layers in the encoder    
        """
        super(TeamEncoder, self).__init__()
        self.embedding_dim = embedding_dim or 0
        if embedding_dim:
            self.embedding = nn.Embedding(team_count, embedding_dim)
        layers = [
            nn.Linear(int(self.embedding_dim) + feature_cols, hidden_dim, dtype=torch.float32),
            nn.ReLU()
        ]
        for _ in range(no_layers - 1):
            layers.extend([
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32),
                nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
                nn.ReLU()
            ])
        layers.append(
            nn.BatchNorm1d(hidden_dim, dtype=torch.float32)
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor, team: torch.Tensor):
        if self.embedding_dim:
            embedded = self.embedding(team)
            input = torch.cat((data, embedded), dim=-1)
        else:
            input = data
        output = self.layers(input)
        return output