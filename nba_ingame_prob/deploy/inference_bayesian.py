import mlflow
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from pathlib import Path
from scipy.stats import norm
from torch import nn

from nba_ingame_prob.consts import proj_paths, game_info
from nba_ingame_prob.model.bayesian import BayesianResultPredictor
from nba_ingame_prob.model.inputs import load_scalers, Scalers, scale_data


def download_model(run_id, model_uri) -> tuple[Path]:
    """
    Downloads the model and its artifacts to the models directory.

    Args:
        run_id (str): The MLFlow run id.
        model_uri (str): The MLFlow model uri.

    Returns:
        tuple[Path]: The paths to the downloaded artifacts: model, model_init, config, scalers.
    """
    model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=proj_paths.models)
    config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='run_config.yaml', dst_path=proj_paths.models)
    scalers_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='scalers.pkl', dst_path=proj_paths.models)
    model_init_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='model_init.yaml', dst_path=proj_paths.models)
    paths = tuple(
        Path(path)
        for path in (model_path, model_init_path, config_path, scalers_path)      
    )
    return paths

def download_model_to(model_uri: str) -> tuple[Path]:
    """
    Downloads the model and its artifacts to the models directory.

    Args:
        model_uri (str): The MLFlow model uri.

    Returns:
        tuple[Path]: The paths to the downloaded artifacts: config, scalers, model, model_init.
    """
    run_id = model_uri.split('/')[2]
    paths = download_model(run_id, model_uri)
    model_path = paths[0]
    model_descr = model_path.stem.split('-', 1)[1]
    paths_w_descr = [
        path.with_stem(f'{path.stem}-{model_descr}')
        for path in paths[1:]
    ]
    for path, new_path in zip(paths[1:], paths_w_descr):
        path.rename(new_path)
    return (model_path, ) + tuple(paths_w_descr)

def load_bayesian_model(model_path: Path, model_init_path: Path, config_path: Path, scalers_path: Path) -> tuple[nn.Module, Scalers, list]:
    """
    Loads the model, scalers, team features and config from the given paths.

    Args:
        model_path (Path): The path to the model file.
        model_init_path (Path): The path to the model init file.
        config_path (Path): The path to the config file.
        scalers_path (Path): The path to the scalers file.

    Returns:
        tuple[nn.Module, Scalers, list]: The model, scalers, team features and config.
    """
    scalers = load_scalers(scalers_path)
    config = OmegaConf.load(config_path)
    model_init = OmegaConf.load(model_init_path)
    team_features = config['inputs_team'] or []
    model = BayesianResultPredictor(**model_init)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model, scalers, team_features, config

def get_last_game(team_abbrev: str, plays_next: str, X: pd.DataFrame, teams: pd.DataFrame) -> pd.Series:
    team_id = teams[teams['abbreviation'] == team_abbrev]['team_id'].values[0]
    team_flag = (X.home_team_id == team_id) | (X.away_team_id == team_id)
    X_team = X[team_flag].sort_values(by='game_date').sort_values(by='game_date').iloc[-1]
    played_last = 'home' if X_team.home_team_id == team_id else 'away'
    columns_names = [
        col 
        for col in X.columns
        if col.startswith(played_last)
    ]
    X_team = X_team[columns_names]
    X_team.index = [
        col.replace(played_last, plays_next)
        for col in X_team.index
    ]
    return X_team


def prepare_experiment(abbrev_home: str, abbrev_away: str, score_diff: int, X: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    last_game_data_home = get_last_game(abbrev_home, 'home', X, teams)
    last_game_data_away = get_last_game(abbrev_away, 'away', X, teams)
    team_input = pd.concat([last_game_data_home, last_game_data_away])
    time_remaining = np.linspace(0, game_info.match_time, game_info.match_time + 1)[::-1]
    df_input = pd.DataFrame(team_input).transpose()
    df_input = pd.concat([df_input]*len(time_remaining), ignore_index=True)
    df_input['time_remaining'] = time_remaining
    return df_input

def prepare_input_for_model(abbrev_home: str, abbrev_away: str, team_features: list[str], scalers, X: pd.DataFrame, team_encoder: dict[str, int]):
    home_team_cols = [
        col
        for col in team_features
        if col.startswith('home')
    ]
    away_team_cols = [
        col
        for col in team_features
        if col.startswith('away')
    ]
    X = scale_data(X.copy(), scalers, team_features)
    torch_input = {
        'home_team': torch.tensor([team_encoder[abbrev_home]]*len(X), dtype=torch.int32),
        'away_team': torch.tensor([team_encoder[abbrev_away]]*len(X), dtype=torch.int32),
        'home_data': torch.tensor(X[home_team_cols].values.astype('float32'), dtype=torch.float32),
        'away_data': torch.tensor(X[away_team_cols].values.astype('float32'), dtype=torch.float32),
        'diff': torch.tensor(X['score_diff'], dtype=torch.float32),
        'time_remaining': torch.tensor(X['time_remaining'], dtype=torch.float32),
    }
    return torch_input

def run_inference(model: nn.Module, data: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = model(**data)
    mu, logvar = torch.chunk(output, 2, dim=-1)
    mu = mu.detach().numpy()
    std = torch.sqrt(torch.exp(logvar)).detach().numpy()
    probs = norm.cdf(0, mu, std).flatten()
    return probs, mu, std

def calculate_probs_bayesian(abbrev_home, abbrev_away, experiment, model, config, scalers, team_features, team_encoder) -> pd.DataFrame:
    input = prepare_input_for_model(abbrev_home, abbrev_away, team_features, scalers, experiment, team_encoder)
    probs, mu, std = run_inference(model, input)
    experiment.loc[:, 'probs'] = probs
    return experiment
