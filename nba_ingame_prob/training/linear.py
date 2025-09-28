import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Optional

from nba_ingame_prob.model.inputs import scale_data, Scalers, home_away_unification


def _get_team_categories(encoder: OneHotEncoder) -> list[str]:
    """
    Extract categories from a fitted OneHotEncoder for a single-column input.

    Params:
        encoder (OneHotEncoder): Fitted encoder.

    Returns:
        list[str]: List of category labels as strings.
    """
    cats = encoder.categories_[0].tolist()
    return [str(c) for c in cats]


def _apply_team_ohe(df: pd.DataFrame, enc: OneHotEncoder, col: str, prefix: str) -> pd.DataFrame:
    """
    Apply a fitted OneHotEncoder to a single categorical column.

    Params:
        df (pd.DataFrame): Source dataframe.
        enc (OneHotEncoder): Fitted encoder matching the column categories.
        col (str): Column name to transform.
        prefix (str): Prefix for resulting one-hot column names.

    Returns:
        pd.DataFrame: One-hot expanded columns aligned to df.index.
    """
    arr = enc.transform(df[[col]])
    cats = _get_team_categories(enc)
    cols = [f"{prefix}{c}" for c in cats]
    return pd.DataFrame(arr, columns=cols, index=df.index)


def data_pipeline_linear(
    X: pd.DataFrame,
    scalers: Scalers,
    team_features: list[str],
    include_teams: bool = True,
    target: Optional[str] = None,
    drop_game_id: bool = True,
    unify_home_away: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Linear/logistic data pipeline for baseline models.

    - Scales numeric features via scale_data (time_remaining, score_diff, team_features).
    - Optionally expands one-hot features for home/away teams if encoders are present.
    - Optionally unifies home/away perspectives similar to CatBoost pipeline.

    Params:
        X (pd.DataFrame): Input data.
        scalers (Scalers): Fitted scalers (optionally includes 'onehot_home'/'onehot_away').
        team_features (list[str]): Team numeric features to scale.
        include_teams (bool): If True, include OHE team columns when available.
        target (str | None): If provided, return y as the given target column.
        drop_game_id (bool): If True, exclude 'game_id' from features.
        unify_home_away (bool): If True, apply home/away unification (doubles dataset).

    Returns:
        tuple[pd.DataFrame, Optional[pd.Series]]: Features X and optional target y.
    """
    # Pull y BEFORE scaling to keep it in original units if needed
    y = X[target].copy() if target else None

    Xs = scale_data(X.copy(), scalers, team_features)
    features = team_features + ['score_diff', 'time_remaining']
    if not drop_game_id:
        features.append('game_id')

    if include_teams and 'onehot_home' in scalers and 'onehot_away' in scalers:
        Xs['home_team_abbreviation'] = X['home_team_abbreviation']
        Xs['away_team_abbreviation'] = X['away_team_abbreviation']
        
        # Apply unification before one-hot encoding if requested
        if unify_home_away:
            # For unification, we need to preserve team columns and apply unification first
            features_for_unify = team_features + ['score_diff', 'time_remaining']
            if 'home_team_abbreviation' in X.columns and 'away_team_abbreviation' in X.columns:
                features_for_unify += ['home_team_abbreviation', 'away_team_abbreviation']
            
            if not drop_game_id:
                features_for_unify.append('game_id')
            if target:
                features_for_unify.append(target)
                
            Xs_for_unify = Xs[features_for_unify]
            Xs_unified = home_away_unification(Xs_for_unify)
            
            # After unification, teams become 'team_abbreviation' and 'team_opponent_abbreviation'
            # Apply one-hot encoding to the unified columns
            home_ohe = _apply_team_ohe(Xs_unified, scalers['onehot_home'], 'team_abbreviation', 'team_')
            away_ohe = _apply_team_ohe(Xs_unified, scalers['onehot_away'], 'team_opponent_abbreviation', 'team_opponent_')
            
            Xs_unified = pd.concat([Xs_unified, home_ohe, away_ohe], axis=1)
            final_features = team_features + ['score_diff', 'time_remaining', 'is_home']
            if not drop_game_id:
                final_features.append('game_id')
            final_features += list(home_ohe.columns) + list(away_ohe.columns)
            
            Xs = Xs_unified[final_features + ([target] if target else [])]
            
            # Update y for unification (double the target)
            if y is not None:
                y_unified = home_away_unification(pd.DataFrame({target: y}))
                y = y_unified[target]
        else:
            # Standard one-hot encoding without unification
            home_ohe = _apply_team_ohe(Xs, scalers['onehot_home'], 'home_team_abbreviation', 'home_')
            away_ohe = _apply_team_ohe(Xs, scalers['onehot_away'], 'away_team_abbreviation', 'away_')
            Xs = pd.concat([Xs, home_ohe, away_ohe], axis=1)
            features += list(home_ohe.columns) + list(away_ohe.columns)
            
            Xs = Xs[features + ([target] if target else [])]
    else:
        # No one-hot encoding available or not requested - just select numeric features
        Xs = Xs[features + ([target] if target else [])]

    if target:
        Xs = Xs.drop(columns=[target])
        
    return Xs, y


@dataclass
class LinearRegressionModel:
    model: Ridge
    sigma: float


def fit_logistic_baseline(
    X: pd.DataFrame,
    y_bin: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int | None = None,
) -> LogisticRegression:
    """
    Fit a logistic regression classifier.

    Params:
        X (pd.DataFrame): Feature matrix.
        y_bin (pd.Series): Binary labels (0/1).
        C (float): Inverse regularization strength.
        max_iter (int): Maximum solver iterations.
        random_state (int | None): Optional model seed.

    Returns:
        LogisticRegression: Fitted classifier.
    """
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    clf.fit(X.values, y_bin.values)
    return clf


def fit_linear_regression_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 42,
) -> LinearRegressionModel:
    """
    Fit a ridge regression baseline and estimate residual sigma on a holdout.

    Params:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target margins.
        alpha (float): Ridge regularization strength.
        test_size (float): Fraction for the holdout set.
        seed (int): Random seed.

    Returns:
        LinearRegressionModel: Fitted ridge model and estimated sigma.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X.values, y.values, test_size=test_size, random_state=seed)
    reg = Ridge(alpha=alpha)
    reg.fit(X_tr, y_tr)
    resid = y_te - reg.predict(X_te)
    sigma = float(np.std(resid)) if len(resid) > 1 else 1.0
    return LinearRegressionModel(model=reg, sigma=sigma)
