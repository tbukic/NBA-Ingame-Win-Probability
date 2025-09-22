import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from omegaconf import OmegaConf

from nba_betting_ai.model.inputs import load_scalers
from nba_betting_ai.training.catboost import data_pipeline_cb


def load_catboost_model(model_path, config_path, scalers_path):
    scalers = load_scalers(scalers_path)
    config = OmegaConf.load(config_path)
    team_features = config.inputs_team if hasattr(config, 'inputs_team') and config.inputs_team else []
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model, scalers, team_features, config


def calculate_probs_catboost(abbrev_home, abbrev_away, experiment, model, config, scalers, team_features, team_encoder) -> pd.DataFrame:
    experiment = experiment.copy()
    # Get include_teams from the training.model_config section
    try:
        include_teams = config.training.model_config.include_teams
    except (AttributeError, KeyError):
        include_teams = False
    
    try:
        unify_home_away = config.training.model_config.unify_home_away
    except (AttributeError, KeyError):
        unify_home_away = True
    
    if include_teams:
        experiment['home_team_abbreviation'] = abbrev_home
        experiment['away_team_abbreviation'] = abbrev_away
    
    # Call data_pipeline_cb with correct parameters for inference
    processed_data = data_pipeline_cb(
        X=experiment, 
        scalers=scalers, 
        team_features=team_features, 
        include_teams=include_teams,
        target=None,  # No target for inference
        unify_home_away=unify_home_away,
        drop_game_id=True
    )
    # Since target=None, it should return just X (DataFrame), but add safety check
    if isinstance(processed_data, tuple):
        experiment = processed_data[0]
    else:
        experiment = processed_data
    
    # If unify_home_away=True, the data is duplicated (home and away perspectives)
    # For inference, we only want one perspective (home perspective), so take the first half
    if unify_home_away:
        original_length = len(experiment) // 2
        experiment = experiment.iloc[:original_length].copy()
    
    # After data_pipeline_cb, the categorical features are named 'team' and 'team_opponent'
    # not 'home_team' and 'away_team'
    cat_features = ['team', 'team_opponent'] if include_teams else None
    pool = Pool(
        data=experiment,
        cat_features=cat_features
    )
    preds = model.predict(pool)
    # Ensure preds has the expected shape (n_samples, 2) for mean and variance
    if len(preds.shape) != 2 or preds.shape[1] != 2:
        raise ValueError(f"Expected CatBoost predictions shape (n_samples, 2), got {preds.shape}")
    
    mean = preds[:,0]  # Only need the mean prediction
    
    # CatBoost was trained with target transformed as: y = clip(score_diff/4 + 0.5, 0, 1)
    # So the predictions are in this transformed space where:
    # - 0.5 = tied game
    # - > 0.5 = home team advantage  
    # - < 0.5 = away team advantage
    # The predicted mean IS the probability directly since it's in [0,1] space
    probs = mean
    
    # Ensure probs is 1D and in valid range
    if len(probs.shape) != 1:
        raise ValueError(f"Expected probs to be 1D, got shape {probs.shape}")
    
    # Clip to ensure valid probabilities
    probs = np.clip(probs, 0.0, 1.0)
    
    experiment.loc[:, 'probs'] = probs
    return experiment
