import mlflow
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from pathlib import Path
from scipy.stats import norm
from torch import nn
from torch.utils.data import DataLoader
from torchviz import make_dot

from nba_betting_ai.consts import proj_paths, game_info
from nba_betting_ai.model.bayesian import BayesianResultPredictor
from nba_betting_ai.model.inputs import load_scalers, Scalers, scale_data
from nba_betting_ai.training.dataset import NBADataset
from nba_betting_ai.training.pipeline import prepare_data


def download_model(run_id, model_uri) -> tuple[Path]:
    """
    Downloads the model and its artifacts to the models directory.

    Args:
        run_id (str): The MLFlow run id.
        model_uri (str): The MLFlow model uri.

    Returns:
        tuple[Path]: The paths to the downloaded artifacts: config, scalers, model, model_init.
    """
    config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='run_config.yaml', dst_path=proj_paths.models)
    scalers_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='scalers.pkl', dst_path=proj_paths.models)
    model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=proj_paths.models)
    model_init_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='model_init.yaml', dst_path=proj_paths.models)
    paths = tuple(
        Path(path)
        for path in (config_path, scalers_path, model_path, model_init_path)      
    )
    return paths

def download_model_to(run_id: str, model_uri: str, prefix: str) -> tuple[Path]:
    """
    Downloads the model and its artifacts to the models directory, renaming them with a prefix.
    Prefix is added to the stem of the file, and it is either 'w' for weighted model or
    'nw' for non-weighted model.

    Args:
        run_id (str): The MLFlow run id.
        model_uri (str): The MLFlow model uri.
        prefix (str): The prefix to add to the model artifacts.

    Returns:
        tuple[Path]: The paths to the downloaded artifacts: config, scalers, model, model_init.
    """
    paths = download_model(run_id, model_uri)
    new_paths = [
        path.with_stem(f'{prefix}_{path.stem}')
        for path in paths
    ]
    for path, new_path in zip(paths, new_paths):
        path.rename(new_path)
    return tuple(new_paths)

def load_model(config_path: Path, scalers_path: Path, model_path: Path, model_init_path: Path) -> tuple[nn.Module, Scalers, list]:
    """
    Loads the model, scalers, team features and config from the given paths.

    Args:
        config_path (Path): The path to the config file.
        scalers_path (Path): The path to the scalers file.
        model_path (Path): The path to the model file.
        model_init_path (Path): The path to the model init file.

    Returns:
        tuple[nn.Module, Scalers, list]: The model, scalers, team features and config.
    """
    scalers = load_scalers(scalers_path)
    config = OmegaConf.load(config_path)
    model_init = OmegaConf.load(model_init_path)
    team_features = config['inputs_team']
    model = BayesianResultPredictor(**model_init)
    model.load_state_dict(torch.load(model_path))
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
    team_input['score_diff'] = score_diff
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
        'home_data': torch.tensor(X[home_team_cols].values.astype('float64'), dtype=torch.float64),
        'away_data': torch.tensor(X[away_team_cols].values.astype('float64'), dtype=torch.float64),
        'diff': torch.tensor(X['score_diff'], dtype=torch.float64),
        'time_remaining': torch.tensor(X['time_remaining'], dtype=torch.float64),
    }
    return torch_input

def run_inference(model: nn.Module, data: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = model(**data)
    mu, logvar = torch.chunk(output, 2, dim=-1)
    mu = mu.detach().numpy()
    std = torch.sqrt(torch.exp(logvar)).detach().numpy()
    probs = norm.cdf(0, mu, std).flatten()
    return probs, mu, std

def calculate_probs_for_diff(abbrev_home, abbrev_away, experiment, w_model, nw_model, w_scalers, nw_scalers, w_config, nw_config, team_encoder) -> pd.DataFrame:
    nw_input = prepare_input_for_model(abbrev_home, abbrev_away, nw_config['inputs_team'], nw_scalers, experiment, team_encoder)
    nw_probs, nw_mu, nw_std = run_inference(nw_model, nw_input)

    w_input = prepare_input_for_model(abbrev_home, abbrev_away, w_config['inputs_team'], w_scalers, experiment, team_encoder)
    w_probs, w_mu, w_std = run_inference(w_model, w_input)

    weights = (0.5 - 0.5*nw_input['time_remaining']/nw_input['time_remaining'].max()).detach().numpy().flatten()
    probs = weights*w_probs + (1-weights)*nw_probs
    experiment['probs_nw'] = nw_probs
    experiment['probs_w'] = w_probs
    experiment['probs'] = probs
    return experiment
