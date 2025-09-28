import numpy as np
import pandas as pd

from nba_ingame_prob.model.inputs import scale_data, home_away_unification


def rename(team):
    opponent = 'away' if team == 'home' else 'home'
    def _rename_col(col):
        if col.startswith(team):
            return col[len(team) + 1:]
        if col.startswith(opponent):
            return col[len(opponent) + 1:] + '_opponent'
        return col
    return _rename_col

def data_pipeline_cb(X, scalers, team_features, include_teams: bool = True, 
                     target: str | None = None,  unify_home_away: bool = True, drop_game_id: bool=True) -> tuple[pd.DataFrame]:
    features = team_features + ['score_diff', 'time_remaining']
    if not drop_game_id:
        features.append('game_id')
    if include_teams:
        features += ['home_team', 'away_team']
    X = scale_data(X, scalers, team_features)
    if include_teams:
        X['home_team'] = X['home_team_abbreviation']
        X['away_team'] = X['away_team_abbreviation']
    X = X[features + (target if target else [])]
    if unify_home_away:
        X = home_away_unification(X)
    y = np.clip(X[target]/4 + 0.5, 0, 1) if target else None
    X = X.drop(columns=target if target else [])
    return (X, y) if target else X
