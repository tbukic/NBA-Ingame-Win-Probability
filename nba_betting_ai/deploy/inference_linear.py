import pickle
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
from scipy.stats import norm
from typing import Any, Tuple

from nba_betting_ai.model.inputs import load_scalers
from nba_betting_ai.training.linear import data_pipeline_linear


def load_linear_model(model_path: Path, config_path: Path, scalers_path: Path) -> Tuple[Any, dict, list, dict]:
    """
    Load a linear/logistic baseline model, config and scalers.

    Params:
        model_path (Path): Path to the pickled model (.pkl).
        config_path (Path): Path to the run_config used for training.
        scalers_path (Path): Path to the scalers pickle (includes OHE if used).

    Returns:
        tuple: (model, scalers, team_features, config)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    scalers = load_scalers(scalers_path)
    config = OmegaConf.load(config_path)
    team_features = config.get('inputs_team', []) or []
    return model, scalers, team_features, config


def calculate_probs_linear(
    abbrev_home: str,
    abbrev_away: str,
    experiment: pd.DataFrame,
    model: Any,
    config: dict,
    scalers: dict,
    team_features: list,
    team_encoder: dict,
) -> pd.DataFrame:
    """
    Compute probabilities for linear/logistic baselines.

    - Logistic: use predict_proba directly.
    - Regression: treat predicted margin as Gaussian with sigma from model.sigma.

    Params:
        abbrev_home (str): Home team abbreviation.
        abbrev_away (str): Away team abbreviation.
        experiment (pd.DataFrame): Input samples to score.
        model (Any): Fitted linear/logistic model.
        config (dict): Loaded run config; should include 'linear_type'.
        scalers (dict): Fitted scalers/OHEs used in training.
        team_features (list): Team numeric feature names.
        team_encoder (dict): Not used; kept for API compatibility.

    Returns:
        pd.DataFrame: experiment with an added 'probs' column.
    """
    experiment = experiment.copy()
    # Ensure team abbreviations present for OHE when available
    experiment['home_team_abbreviation'] = abbrev_home
    experiment['away_team_abbreviation'] = abbrev_away
    X, _ = data_pipeline_linear(experiment, scalers, team_features, include_teams=True, target=None)
    linear_type = config.get('linear_type', 'logistic')
    if linear_type == 'logistic':
        probs = model.predict_proba(X.values)[:, 1]
    else:
        # Regression to margin: interpret as P(home win) = P(margin < 0) = Phi(0; mu, sigma)
        mu = model.model.predict(X.values) if hasattr(model, 'model') else model.predict(X.values)
        sigma = getattr(model, 'sigma', 1.0)
        probs = norm.cdf(0, loc=mu, scale=sigma)
    experiment.loc[:, 'probs'] = probs
    return experiment
