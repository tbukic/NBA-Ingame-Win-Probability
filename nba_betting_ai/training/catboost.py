import numpy as np
import pandas as pd

from nba_betting_ai.model.inputs import scale_data
from nba_betting_ai.training.pipeline import DataSet


def data_pipeline_cb(X, scalers, team_features, team_encoder, include_teams: bool = True, target: str | None = None) -> tuple[pd.DataFrame]:
    features = team_features + ['score_diff', 'time_remaining']
    if include_teams:
        features += ['home_team', 'away_team']
    X = scale_data(X, scalers, team_features)
    if include_teams:
        X['home_team'] = X['home_team_abbreviation'].map(team_encoder)
        X['away_team'] = X['away_team_abbreviation'].map(team_encoder)
    y = np.clip(X[target]/4 + 0.5, 0, 1) if target else None
    X = X[features]
    return (X, y) if target else X

