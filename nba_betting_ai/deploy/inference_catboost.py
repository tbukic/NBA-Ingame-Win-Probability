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
    team_features = config['inputs_team'] or []
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model, scalers, team_features, config 


def calculate_probs_catboost(abbrev_home, abbrev_away, experiment, model, config, scalers, team_features, team_encoder) -> pd.DataFrame:
    experiment = experiment.copy()
    include_teams = config.get('include_teams', False)
    if include_teams:
        experiment['home_team_abbreviation'] = abbrev_home
        experiment['away_team_abbreviation'] = abbrev_away
    experiment = data_pipeline_cb(experiment, scalers, team_features, team_encoder)
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
    # experiment['probs'] = prob
    return experiment