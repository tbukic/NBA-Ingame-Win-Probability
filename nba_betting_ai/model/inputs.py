import pandas as pd
import pickle

from pathlib import Path
from sklearn.base import OneToOneFeatureMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import TypeAlias

from nba_betting_ai.consts import game_info

Scalers: TypeAlias = dict[str, OneToOneFeatureMixin]


def rename(team):
    """Helper function to rename columns for home/away unification."""
    opponent = 'away' if team == 'home' else 'home'
    def _rename_col(col):
        if col.startswith(team):
            return col[len(team) + 1:]
        if col.startswith(opponent):
            return col[len(opponent) + 1:] + '_opponent'
        return col
    return _rename_col


def home_away_unification(X: pd.DataFrame) -> pd.DataFrame:
    """
    Unify home and away perspectives by creating two views of each game.
    
    This transformation doubles the dataset by creating:
    1. Home team perspective: home_* -> base columns, away_* -> *_opponent columns
    2. Away team perspective: away_* -> base columns, home_* -> *_opponent columns
    
    Score differences are flipped for the away perspective to maintain consistency.
    
    Params:
        X (pd.DataFrame): Input dataframe with home_* and away_* prefixed columns
        
    Returns:
        pd.DataFrame: Unified dataframe with doubled rows and is_home indicator
    """
    home_view = X.copy()
    home_view.columns = [
        rename('home')(col)
        for col in home_view.columns
    ]
    home_view['is_home'] = True
    home_view['score_diff'] = -home_view['score_diff']
    if 'final_score_diff' in home_view.columns:
        home_view['final_score_diff'] = -home_view['final_score_diff']
    
    away_view = X.copy()
    away_view.columns = [
        rename('away')(col)
        for col in away_view.columns
    ]
    away_view['is_home'] = False

    X = pd.concat([home_view, away_view], axis=0)
    return X

def prepare_scalers(
        X_train: pd.DataFrame,
        team_features: list[str],
        save_path: Path | None = None,
        fit_onehot_teams: bool = False,
    ) -> Scalers:
    """
    Tune scalers on training data and optionally persist them.

    Params:
        X_train (pd.DataFrame): Training data used to fit scalers.
        team_features (list[str]): List of team-related numeric features to standardize.
        save_path (Path | None): Optional path to write the fitted scalers pickle.
        fit_onehot_teams (bool): If True, also fit and store OneHotEncoders for
            'home_team_abbreviation' and 'away_team_abbreviation' under keys
            'onehot_home' and 'onehot_away'. These encoders are intended for
            linear/logistic baselines and are not applied by scale_data.

    Returns:
        Scalers: Dictionary of fitted transformers keyed by name.
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

    # Optionally include one-hot encoders for teams alongside numeric scalers.
    # These encoders are intended for linear/logistic baselines and are not
    # applied by scale_data; consume them explicitly in those pipelines.
    if fit_onehot_teams:
        if 'home_team_abbreviation' in X_train.columns and 'away_team_abbreviation' in X_train.columns:
            enc_home = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            enc_away = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            enc_home.fit(X_train[['home_team_abbreviation']])
            enc_away.fit(X_train[['away_team_abbreviation']])
            scalers['onehot_home'] = enc_home
            scalers['onehot_away'] = enc_away

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
