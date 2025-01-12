import pandas as pd
import torch
from torch.utils.data import Dataset


class NBADataset(Dataset):
    def __init__(self, feature_cols: list[str], target_cols: list[str], X: pd.DataFrame, df_teams: pd.DataFrame):
        """
        NBADataset is a PyTorch Dataset wrapper around dataframes containing NBA data. It is created from the
        list of feature columns, target columns, the input data dataframe X, and the dataframe containing team lists.

        Args:
            - feature cols (list[str]): List of feature columns
            - target_cols (list[str]): List of target columns
            - X (pd.DataFrame): Input data
            - df_teams (pd.DataFrame): DataFrame with NBA teams
        """
        home_team_cols = [
            col for col in feature_cols
            if col.startswith('home_season') or col.startswith('home_last_5')
        ]
        self.home_data = X[home_team_cols]
        away_team_cols = [
            col for col in feature_cols
            if col.startswith('away_season') or col.startswith('away_last_5')
        ]
        self.away_data = X[away_team_cols]
        self.team_encoder = dict(zip(df_teams['abbreviation'], df_teams.index))
        self.team_decoder = dict(zip(df_teams.index, df_teams['abbreviation']))
        self.home_team = X['home_team_abbreviation']
        self.away_team = X['away_team_abbreviation']
        self.diff = X['score_diff']
        self.time_remaining = X['time_remaining']
        self.target_cols = target_cols
        self.y = X[target_cols]

    def encode_team(self, team_abbreviation: str) -> int:
        """
        Given a team abbreviation, return the corresponding team integer code,
        from 0 to the number of teams.

        Args:
            - team_abbreviation (str): Team abbreviation

        Returns:
            - int: Team code
        """
        return self.team_encoder[team_abbreviation]
    
    def decode_team(self, team_id: int) -> str:
        """
        Given a team integer code, return the corresponding team abbreviation.

        Args:
            - team_id (int): Team code

        Returns:
            - str: Team abbreviation
        """
        return self.team_decoder[team_id]

    def __len__(self):
        return len(self.home_data)

    def __getitem__(self, idx):
        data = {
            'home_team': torch.tensor(self.encode_team(self.home_team.iloc[idx]), dtype=torch.int32),
            'away_team': torch.tensor(self.encode_team(self.away_team.iloc[idx]), dtype=torch.int32),
            'home_data': torch.tensor(self.home_data.iloc[idx].values, dtype=torch.float64),
            'away_data': torch.tensor(self.away_data.iloc[idx].values, dtype=torch.float64),
            'diff': torch.tensor(self.diff.iloc[idx], dtype=torch.float64),
            'time_remaining': torch.tensor(self.time_remaining.iloc[idx], dtype=torch.float64),
            'y': torch.tensor(self.y.iloc[idx].values, dtype=torch.float64)
        }
        return data
    
