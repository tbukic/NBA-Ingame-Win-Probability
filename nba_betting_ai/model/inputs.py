import pandas as pd
import pickle

from pathlib import Path
from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import TypeAlias


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
    period_min = 12
    periods_regular = 4
    min_to_sec = 60

    scaler_game_time = MinMaxScaler(feature_range=(-1, 1))
    scaler_game_time.fit(X_train[['time_remaining']])
    scaler_game_time.min_ = -1
    scaler_game_time.scale_ = 2/(period_min*periods_regular*min_to_sec)
    
    scaler_score_diff = StandardScaler(with_mean=False)
    scaler_score_diff.fit(X_train[['score_diff']])

    scaler_final_score_diff = StandardScaler(with_mean=False)
    scaler_final_score_diff.fit(X_train[['final_score_diff']])

    scaler_team_features = StandardScaler()
    scaler_team_features.fit(X_train[team_features])

    scalers = {
        'time_remaining': scaler_game_time,
        'score_diff': scaler_score_diff,
        'final_score_diff': scaler_final_score_diff,
        'team_features': scaler_team_features
    }

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
    df['final_score_diff'] = scalers['final_score_diff'].transform(df[['final_score_diff']])
    df[team_features] = scalers['team_features'].transform(df[team_features])
    return df
