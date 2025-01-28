import pandas as pd
import pickle

from pathlib import Path
from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import TypeAlias

from nba_betting_ai.consts import game_info

Scalers: TypeAlias = dict[str, OneToOneFeatureMixin]

def prepare_scalers(
        X_train: pd.DataFrame,
        team_features: list[str],
        save_path: Path | None = None
    ) -> Scalers:
    """
    Tune scalers on training data and save them to a file.

    Params:
        X_train: pd.DataFrame - training data
        team_features: list[str] - list of features related to teams
        save_path: Path - path to save scalers to

    Returns:
        Scalers - dictionary of scalers
    """
    scaler_game_time = MinMaxScaler(feature_range=(-1, 1))
    scaler_game_time.fit(X_train[['time_remaining']])
    scaler_game_time.min_ = -1
    scaler_game_time.scale_ = 2/game_info.match_time
    
    scaler_score_diff = StandardScaler(with_mean=False)
    scaler_score_diff.fit(X_train[['score_diff']])

    scaler_final_score_diff = StandardScaler(with_mean=False)
    scaler_final_score_diff.fit(X_train[['final_score_diff']])
    
    scalers = {
        'time_remaining': scaler_game_time,
        'score_diff': scaler_score_diff,
        'final_score_diff': scaler_final_score_diff,
    }
    if team_features:
        scaler_team_features = StandardScaler()
        scaler_team_features.fit(X_train[team_features])
        scalers['team_features'] = scaler_team_features

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open('wb') as f:
            pickle.dump(scalers, f)
    return scalers

def load_scalers(scalers_path: Path) -> Scalers:
    """
    Load scalers from a pickle file.

    Params:
        scalers_path: Path - path to the pickle file

    Returns:
        Scalers - dictionary of scalers
    """
    with scalers_path.open('rb') as f:
        scalers = pickle.load(f)
    return scalers

def scale_data(df: pd.DataFrame, scalers: Scalers, team_features: list[str]) -> pd.DataFrame:
    """
    Scale the data using the provided scalers.

    Params:
        df: pd.DataFrame - data to scale
        scalers: Scalers - dictionary of scalers
        team_features: list[str] - list of features related to teams
    
    Returns:
        pd.DataFrame - scaled data
    """
    df = df.copy()
    df['time_remaining'] = scalers['time_remaining'].transform(df[['time_remaining']])
    df['score_diff'] = scalers['score_diff'].transform(df[['score_diff']])
    if 'final_score_diff' in df.columns:
        df['final_score_diff'] = scalers['final_score_diff'].transform(df[['final_score_diff']])
    if not team_features:
        return df
    transformed_values = scalers['team_features'].transform(df[team_features])
    for i, feature in enumerate(team_features):
        df[feature] = transformed_values[:, i]
    return df
