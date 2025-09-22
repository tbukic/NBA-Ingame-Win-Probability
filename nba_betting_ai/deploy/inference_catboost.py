import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from omegaconf import OmegaConf
from scipy.stats import norm

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
    
    cat_features = ['home_team', 'away_team'] if include_teams else None
    pool = Pool(
        data=experiment,
        cat_features=cat_features
    )
    preds = model.predict(pool)
    mean, var_preds = preds[:,0], preds[:,1]
    std = np.sqrt(var_preds)
    probs = norm.cdf(0, mean - 0.5, std)
    experiment.loc[:, 'probs'] = probs
    return experiment
