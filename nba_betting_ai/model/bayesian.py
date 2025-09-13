import torch
from torch import nn

from nba_betting_ai.model.team_encoder import TeamEncoder


class BayesianResultPredictor(nn.Module):
    def __init__(
        self,
        team_count: int,
        team_features: int,
        embedding_dim: int | None,
        team_hidden_dim: int,
        team_layers: int,
        res_hidden_dim: int,
        res_layers: int,
        time_scaling: bool = True
    ):
        """
        Model for predicting the result of a game based on the team data, remaining time and the difference of scores.
        Params:
            - team_count (int): number of teams in the dataset
            - team_features (int): number of features per team
            - embedding_dim (int): dimension of the team embedding. Can be None or 0 if no embedding is used.
            - team_hidden_dim (int): dimension of the hidden layer in the team encoder
            - team_layers (int): number of layers in the team encoder
            - res_hidden_dim (int): dimension of the hidden layer in the result predictor
            - res_layers (int): number of layers in the result predictor
            - time_scaling (bool): whether to scale the last layer of the network by the time remaining in the game
        """
        super(BayesianResultPredictor, self).__init__()
        self.team_encoder = TeamEncoder(team_count, team_features, embedding_dim, team_hidden_dim, team_layers)

        layers = [
            nn.Linear(2*team_hidden_dim + 2, res_hidden_dim, dtype=torch.float32),
            nn.BatchNorm1d(res_hidden_dim, dtype=torch.float32),
            nn.ReLU(),
        ]
        for _ in range(res_layers - 1):
            layers.extend([
                nn.Linear(res_hidden_dim, res_hidden_dim, dtype=torch.float32),
                nn.BatchNorm1d(res_hidden_dim, dtype=torch.float32),
                nn.ReLU(),
            ])
        self.layers = nn.Sequential(*layers)
        self.time_scaling = time_scaling
        if self.time_scaling:
            self.time_encoder = nn.Linear(1, res_hidden_dim, dtype=torch.float32)
            self.post_encode = nn.Linear(res_hidden_dim, res_hidden_dim, dtype=torch.float32)
            self.bn = nn.BatchNorm1d(res_hidden_dim, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(res_hidden_dim, 2, dtype=torch.float32)
    
    def forward(self, home_team, home_data, away_team, away_data, diff, time_remaining):
        home = self.team_encoder(home_data, home_team)
        away = self.team_encoder(away_data, away_team)
        diff, time_remaining = diff.reshape(-1, 1), time_remaining.reshape(-1, 1)
        input = torch.cat((home, away, diff, time_remaining), dim=-1)
        mid = self.layers(input)
        if self.time_scaling:
            time = self.time_encoder(time_remaining)
            mid = mid + time
            mid = self.bn(mid)
        final = self.relu(mid)
        return self.output(final)
    