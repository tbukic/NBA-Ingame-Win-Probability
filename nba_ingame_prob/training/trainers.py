import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Callable, Iterable, Tuple

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from omegaconf import OmegaConf
from scipy.stats import norm

from nba_ingame_prob.consts import proj_paths, game_info
from nba_ingame_prob.training.dataset import NBADataset
from nba_ingame_prob.training.metrics import compute_all_metrics
from nba_ingame_prob.training.mlflow_utils import (
    log_artifact_path,
    log_metrics_dict,
    log_params_flat,
    ensure_experiment,
    set_tags,
)
from nba_ingame_prob.training.pipeline import DataSet
from nba_ingame_prob.training.plotting import save_feature_importance_plots, save_calibration_plot
from nba_ingame_prob.training.run_torch import run_training
from nba_ingame_prob.training.catboost import data_pipeline_cb
from nba_ingame_prob.training.linear import (
    data_pipeline_linear,
    fit_linear_regression_baseline,
    fit_logistic_baseline,
)


def _log_artifacts_to_mlflow(model_obj: Any, config: OmegaConf, scalers_path: Path, 
                           model_suffix: str = '.pkl', artifact_folder: str = 'artifacts', 
                           model_init: Any = None) -> None:
    """
    DRY helper to always log model, config, and scalers to MLflow.
    
    Params:
        model_obj: The trained model object to serialize and log.
        config (OmegaConf): The run configuration to log.
        scalers_path (Path): Path to the scalers file to log.
        model_suffix (str): File extension for the model (.pkl, .cbm, etc.).
        model_init (OmegaConf): Optional model initialization config for Bayesian models.
    """
    artifact_path = artifact_folder.strip('/')

    with tempfile.NamedTemporaryFile(suffix=model_suffix, delete=False) as tmp_model:
        if model_suffix == '.cbm':
            model_obj.save_model(tmp_model.name)
        elif model_suffix == '.pth':
            if hasattr(model_obj, 'exists') and model_obj.exists():
                import shutil
                shutil.copyfile(str(model_obj), tmp_model.name)
            else:
                import torch
                torch.save(model_obj, tmp_model.name)
        else:
            with open(tmp_model.name, 'wb') as f:
                pickle.dump(model_obj, f)
    temp_dir = os.path.dirname(tmp_model.name)
    model_file = os.path.join(temp_dir, f'model{model_suffix}')
    os.rename(tmp_model.name, model_file)
    mlflow.log_artifact(model_file, artifact_path=artifact_path)
    os.unlink(model_file)
    
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_config:
        tmp_config.write(OmegaConf.to_yaml(config).encode())
        tmp_config.flush()
    temp_dir = os.path.dirname(tmp_config.name)
    config_file = os.path.join(temp_dir, 'run_config.yaml')
    os.rename(tmp_config.name, config_file)
    mlflow.log_artifact(config_file, artifact_path=artifact_path)
    os.unlink(config_file)
    
    if scalers_path.exists():
        import tempfile as tf
        temp_dir = tf.gettempdir()
        scalers_file = os.path.join(temp_dir, 'scalers.pkl')
        import shutil
        shutil.copyfile(str(scalers_path), scalers_file)
        mlflow.log_artifact(scalers_file, artifact_path=artifact_path)
        os.unlink(scalers_file)
    else:
        print(f"Warning: Scalers file not found at {scalers_path}")
    
    # Log model_init.yaml for Bayesian models
    if model_init is not None:
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_model_init:
            tmp_model_init.write(OmegaConf.to_yaml(model_init).encode())
            tmp_model_init.flush()
        temp_dir = os.path.dirname(tmp_model_init.name)
        model_init_file = os.path.join(temp_dir, 'model_init.yaml')
        os.rename(tmp_model_init.name, model_init_file)
        mlflow.log_artifact(model_init_file, artifact_path=artifact_path)
        os.unlink(model_init_file)


@dataclass
class TrainingResult:
    """
    Container for training outputs.

    Params:
        model_path (Path | None): Path to the saved model artifact in `models/`.
        config_path (Path | None): Path to the saved run_config associated with the model.
        scalers_path (Path | None): Path to the saved scalers pickle used during training.
        metrics (dict): Logged metrics for the evaluation split.
        run_id (str | None): MLflow run ID for the training run.
    """
    model_path: Optional[Path]
    config_path: Optional[Path]
    scalers_path: Optional[Path]
    metrics: dict
    run_id: Optional[str]


def _fmt_value(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


def _params_to_dict(cfg_section: Any) -> dict[str, Any]:
    if isinstance(cfg_section, dict):
        return dict(cfg_section)
    try:
        converted = OmegaConf.to_container(cfg_section, resolve=True)  # type: ignore[arg-type]
        if isinstance(converted, dict):
            return dict(converted)
    except Exception:
        pass
    try:
        return dict(cfg_section)
    except Exception:
        return {}


def _build_run_name(
    base: str,
    params: dict[str, Any],
    trial_no: Any | None,
    timestamp: str,
    key_aliases: Iterable[Tuple[str, str]],
) -> str:
    pieces: list[str] = []
    for key, alias in key_aliases:
        if key in params:
            pieces.append(f"{alias}={_fmt_value(params[key])}")
    suffix = ", ".join(pieces)
    if trial_no is not None:
        try:
            trial_idx = int(trial_no)
            prefix = f"{base} Trial {trial_idx:03d}"
        except (ValueError, TypeError):
            prefix = f"{base} Trial {trial_no}"
    else:
        prefix = base
    if suffix:
        prefix = f"{prefix} ({suffix})"
    prefix = f"{prefix} - {timestamp}"
    return prefix


def _start_training_run(run_name: str, experiment_name: str):
    """Start a new MLflow run, nesting when an outer run is already active."""
    active = mlflow.active_run()
    if active:
        return mlflow.start_run(run_name=run_name, nested=True)
    ensure_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def _weights_from_time_remaining(times: np.ndarray) -> np.ndarray:
    """
    Build time-based weights consistent with existing training logic.

    Params:
        times (np.ndarray): Time remaining in seconds for each sample.

    Returns:
        np.ndarray: Weights emphasizing later game moments.
    """
    scale_time = 2.0 / game_info.match_time * times - 1.0
    return 1.0 - 0.5 * scale_time


def _to_eval_prefixed(metrics: dict) -> dict:
    """
    Build an eval_* view of given metrics for MLflow consistency across families.

    - For keys like 'log_loss' -> 'eval_log_loss'
    - For weighted keys 'w_log_loss' -> 'eval_log_loss_w'
    - Skip keys that already start with 'train_' or 'eval_'
    """
    out: dict[str, float] = {}
    for key, val in metrics.items():
        if key.startswith('train_') or key.startswith('eval_'):
            continue
        if key.startswith('w_'):
            out[f"eval_{key[2:]}_w"] = val
        else:
            out[f"eval_{key}"] = val
    return out


def _log_feature_importance_plots(feature_importance_data: dict, timestamp: str, model_suffix: str) -> None:
    """
    Generate and log feature importance plots to MLflow artifacts.
    
    Args:
        feature_importance_data: Dictionary with keys like 'catboost_importance', 'linear_coef' etc.
        timestamp: Timestamp string for filename
        model_suffix: Model type suffix for filename (e.g., 'cat', 'reg', 'log')
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_files = save_feature_importance_plots(
                output_dir=temp_dir,
                prefix=f"feature_importance-{timestamp}-{model_suffix}",
                **feature_importance_data
            )
            for plot_file in plot_files:
                log_artifact_path(plot_file)
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot: {e}")


def _log_csv_artifact(data: pd.Series, filepath: Path, save_local: bool) -> None:
    """
    Save CSV data and log to MLflow artifacts if save_local is False.
    """
    data.to_csv(filepath)
    if not save_local:
        log_artifact_path(filepath)
        filepath.unlink(missing_ok=True)


def _log_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, timestamp: str, model_suffix: str) -> None:
    """
    Generate and log calibration plot to MLflow artifacts.
    
    Args:
        y_true: True binary labels (0/1) for validation set
        y_prob: Predicted probabilities for validation set
        model_name: Model name for plot title
        timestamp: Timestamp string for filename
        model_suffix: Model type suffix for filename (e.g., 'cat', 'reg', 'log', 'bay')
    """
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = save_calibration_plot(
                y_true, y_prob, model_name,
                output_dir=temp_dir,
                prefix=f"calibration-{timestamp}-{model_suffix}"
            )
            log_artifact_path(plot_path)
    except Exception as e:
        print(f"Warning: Could not generate calibration plot: {e}")


def train_linear_logistic(config: OmegaConf, data: DataSet, scalers: dict, extra_tags: dict | None = None, save_local_artifacts: bool = True) -> TrainingResult:
    """
    Train a logistic regression baseline using one-hot team encoding.

    Params:
        config (OmegaConf): Global run configuration (reads training/params).
        data (DataSet): Prepared dataset (X_train/X_test/teams).
        scalers (dict): Fitted scalers including team one-hot encoders.

    Returns:
        TrainingResult: Paths and metrics for this run.
    """
    now = time.strftime('%Y%m%d-%H%M%S')
    input_teams = config['inputs_team'] or []

    trial_no = extra_tags.get('trial_no') if extra_tags else None
    params = _params_to_dict(config.get('training', {}).get('params', {}))
    run_name = _build_run_name('Linear Logistic', params, trial_no, now, [('C', 'C')])
    with _start_training_run(run_name, config['experiment_name']) as run:
        set_tags({'model_family': 'linear', 'linear_type': 'logistic', 'team_encoding': 'one-hot'})
        if extra_tags:
            set_tags(extra_tags)
        log_params_flat(config)
        X_train, _ = data_pipeline_linear(data.X_train, scalers, input_teams, include_teams=True, target=None)
        y_train_bin = (data.X_train['final_score_diff'] < 0).astype(int)
        C = params.get('C', 1.0)
        max_iter = params.get('max_iter', 1000)
        rs = params.get('random_state')
        clf = fit_logistic_baseline(X_train, y_train_bin, C=C, max_iter=max_iter, random_state=rs)

        X_test, _ = data_pipeline_linear(data.X_test, scalers, input_teams, include_teams=True, target=None)
        y_test_bin = (data.X_test['final_score_diff'] < 0).astype(int).to_numpy()
        prob_test = clf.predict_proba(X_test.values)[:, 1]

        prob_train = clf.predict_proba(X_train.values)[:, 1]
        train_metrics = compute_all_metrics(y_train_bin.to_numpy(), prob_train, y_margin_true=None, mu=None, var=None, weights=None)
        
        eval_metrics_raw = compute_all_metrics(y_test_bin, prob_test, y_margin_true=None, mu=None, var=None, weights=None)
        eval_metrics = _to_eval_prefixed(eval_metrics_raw)
        
        log_metrics_dict(train_metrics)
        log_metrics_dict(eval_metrics)

        _log_calibration_plot(y_test_bin, prob_test, "Linear Logistic", now, "log")

        if hasattr(clf, 'coef_') and clf.coef_ is not None:
            fi_s = pd.Series(clf.coef_[0], index=X_train.columns)
            
            _log_feature_importance_plots({'linear_coef': fi_s}, now, 'log')
            
            fi_path = proj_paths.output / f'feature_importance-{now}-log.csv'
            _log_csv_artifact(fi_s, fi_path, save_local_artifacts)

        scalers_src = proj_paths.output / 'scalers.pkl'
        try:
            if not scalers_src.exists():
                with open(scalers_src, 'wb') as f:
                    pickle.dump(scalers, f)
            _log_artifacts_to_mlflow(clf, config, scalers_src, '.pkl')
        except Exception as e:
            print(f"Warning: Failed to log artifacts to MLflow: {e}")

        model_path = None
        config_path = None
        scalers_dst = None
        
        if save_local_artifacts:
            model_path = proj_paths.models / f'lin_model-{now}-log.pkl'
            with model_path.open('wb') as f:
                pickle.dump(clf, f)
            config_path = proj_paths.models / f'run_config-{now}-log.yaml'
            config_path.write_text(OmegaConf.to_yaml(config))
            scalers_dst = proj_paths.models / f'scalers-{now}-log.pkl'
            if scalers_src.exists():
                shutil.copyfile(scalers_src.as_posix(), scalers_dst.as_posix())
        else:
            model_path = proj_paths.output / f'temp_lin_model-{now}-log.pkl'
            with model_path.open('wb') as f:
                pickle.dump(clf, f)
            scalers_src.unlink(missing_ok=True)

        run_id = run.info.run_id
    return TrainingResult(model_path, config_path, scalers_dst if scalers_dst and scalers_dst.exists() else None, eval_metrics, run_id)


def train_linear_regression(config: OmegaConf, data: DataSet, scalers: dict, extra_tags: dict | None = None, save_local_artifacts: bool = True) -> TrainingResult:
    """
    Train a linear regression (Ridge) baseline and map margin to probability via Gaussian residuals.

    Params:
        config (OmegaConf): Global run configuration (reads training/params).
        data (DataSet): Prepared dataset.
        scalers (dict): Fitted scalers including team one-hot encoders.

    Returns:
        TrainingResult: Paths and metrics for this run.
    """
    now = time.strftime('%Y%m%d-%H%M%S')
    input_teams = config['inputs_team'] or []
    trial_no = extra_tags.get('trial_no') if extra_tags else None
    params = _params_to_dict(config.get('training', {}).get('params', {}))
    run_name = _build_run_name('Linear Regression', params, trial_no, now, [('alpha', 'alpha')])
    with _start_training_run(run_name, config['experiment_name']) as run:
        set_tags({'model_family': 'linear', 'linear_type': 'regression', 'team_encoding': 'one-hot'})
        if extra_tags:
            set_tags(extra_tags)
        log_params_flat(config)
        X_train, y_train = data_pipeline_linear(data.X_train, scalers, input_teams, include_teams=True, target='final_score_diff')
        params = dict(config.get('training', {}).get('params', {}))
        alpha = params.get('alpha', 1.0)
        model_reg = fit_linear_regression_baseline(X_train, y_train, alpha=alpha)

        X_test, y_test = data_pipeline_linear(data.X_test, scalers, input_teams, include_teams=True, target='final_score_diff')
        mu_test = model_reg.model.predict(X_test.values)
        sigma = float(model_reg.sigma)
        probs_test = norm.cdf(0, loc=mu_test, scale=sigma)

        mu_train = model_reg.model.predict(X_train.values)
        probs_train = norm.cdf(0, loc=mu_train, scale=sigma)
        y_train_bin = (y_train < 0).astype(int).to_numpy()
        train_metrics = compute_all_metrics(y_train_bin, probs_train, y_margin_true=y_train.to_numpy(), mu=mu_train, var=(sigma ** 2) * np.ones_like(mu_train), weights=None)

        y_test_bin = (y_test < 0).astype(int).to_numpy()
        var_test = (sigma ** 2) * np.ones_like(mu_test)
        eval_metrics_raw = compute_all_metrics(y_test_bin, probs_test, y_margin_true=y_test.to_numpy(), mu=mu_test, var=var_test, weights=None)
        eval_metrics = _to_eval_prefixed(eval_metrics_raw)
        
        log_metrics_dict(train_metrics)
        log_metrics_dict(eval_metrics)

        _log_calibration_plot(y_test_bin, probs_test, "Linear Regression", now, "reg")

        if hasattr(model_reg.model, 'coef_') and model_reg.model.coef_ is not None:
            fi_s = pd.Series(model_reg.model.coef_, index=X_train.columns)
            
            _log_feature_importance_plots({'linear_coef': fi_s}, now, 'reg')
            
            fi_path = proj_paths.output / f'feature_importance-{now}-reg.csv'
            _log_csv_artifact(fi_s, fi_path, save_local_artifacts)

        scalers_src = proj_paths.output / 'scalers.pkl'
        try:
            if not scalers_src.exists():
                with open(scalers_src, 'wb') as f:
                    pickle.dump(scalers, f)
            _log_artifacts_to_mlflow(model_reg, config, scalers_src, '.pkl')
        except Exception as e:
            print(f"Warning: Failed to log artifacts to MLflow: {e}")
        
        model_path = None
        config_path = None
        scalers_dst = None
        
        if save_local_artifacts:
            model_path = proj_paths.models / f'lin_model-{now}-reg.pkl'
            with model_path.open('wb') as f:
                pickle.dump(model_reg, f)
            config_path = proj_paths.models / f'run_config-{now}-reg.yaml'
            config_path.write_text(OmegaConf.to_yaml(config))
            scalers_dst = proj_paths.models / f'scalers-{now}-reg.pkl'
            if scalers_src.exists():
                shutil.copyfile(scalers_src.as_posix(), scalers_dst.as_posix())
        else:
            model_path = proj_paths.output / f'temp_lin_model-{now}-reg.pkl'
            with model_path.open('wb') as f:
                pickle.dump(model_reg, f)
            scalers_src.unlink(missing_ok=True)

        run_id = run.info.run_id
    return TrainingResult(model_path, config_path, scalers_dst if scalers_dst and scalers_dst.exists() else None, eval_metrics, run_id)


def train_catboost(config: OmegaConf, data: DataSet, scalers: dict, extra_tags: dict | None = None, save_local_artifacts: bool = True) -> TrainingResult:
    """
    Train CatBoost regressor with uncertainty and compute unified metrics.

    Params:
        config (OmegaConf): Global run configuration.
        data (DataSet): Prepared dataset.
        scalers (dict): Fitted scalers.

    Returns:
        TrainingResult: Paths and metrics for this run.
    """
    include_teams = config.training.model_config.get('include_teams', False)
    now = time.strftime('%Y%m%d-%H%M%S')

    input_teams = config['inputs_team'] or []
    unify = config.training.model_config.get('unify_home_away', True)
    trial_no = extra_tags.get('trial_no') if extra_tags else None
    params_full = _params_to_dict(config.training.params)
    run_name = _build_run_name(
        'CatBoost',
        params_full,
        trial_no,
        now,
        [
            ('learning_rate', 'lr'),
            ('max_depth', 'depth'),
            ('iterations', 'iter'),
        ],
    )

    with _start_training_run(run_name, config['experiment_name']) as run:
        set_tags({'model_family': 'catboost', 'team_encoding': 'label' if include_teams else 'index'})
        if extra_tags:
            set_tags(extra_tags)
        log_params_flat(config)
        X_train, y_train = data_pipeline_cb(
            data.X_train, scalers, input_teams, include_teams,
            config['target'], unify_home_away=unify
        )
        X_test, y_test = data_pipeline_cb(
            data.X_test, scalers, input_teams, include_teams,
            config['target'], unify_home_away=unify, drop_game_id=False
        )
        if include_teams:
            expected_sets = [
                ['team', 'team_opponent'],
                ['home_team', 'away_team'],
            ]
            cat_features = next((cols for cols in expected_sets if all(c in X_train.columns for c in cols)), [])
        else:
            cat_features = []

        X_train_mat = X_train.drop(columns=['game_id'], errors='ignore')
        X_test_mat = X_test.drop(columns=['game_id'], errors='ignore')
        y_train_arr = np.asarray(y_train).reshape(-1)
        y_test_arr = np.asarray(y_test).reshape(-1)
        pool_train = Pool(data=X_train_mat, label=y_train_arr, cat_features=cat_features)
        pool_test = Pool(data=X_test_mat, label=y_test_arr, cat_features=cat_features)

        params = _params_to_dict(params_full)
        if 'logging_level' in params:
            for conflict in ('verbose', 'verbose_eval', 'silent'):
                params.pop(conflict, None)
        
        params['train_dir'] = None
        params['allow_writing_files'] = False
        
        model = CatBoostRegressor(**params)
        fit_verbose = params.get('verbose')
        if isinstance(fit_verbose, int) and 'metric_period' in params:
            fit_verbose = params['metric_period']
        model.fit(pool_train, eval_set=pool_test, verbose=fit_verbose)

        # Compute training metrics
        preds_train = model.predict(pool_train, prediction_type='RMSEWithUncertainty')
        if getattr(preds_train, 'ndim', 1) != 2 or preds_train.shape[1] < 2:
            raise ValueError('CatBoost predict did not return mean+variance for training set.')
        mean_train = preds_train[:, 0]
        var_preds_train = preds_train[:, 1]
        std_train = np.sqrt(np.maximum(var_preds_train, 1e-12))
        probs_train = norm.cdf(0, mean_train - 0.5, std_train)
        y_train_bin = (y_train_arr < 0.5).astype(int)
        train_metrics = compute_all_metrics(y_train_bin, probs_train, y_margin_true=None, mu=None, var=None, weights=None)

        # Compute evaluation metrics
        preds_test = model.predict(pool_test, prediction_type='RMSEWithUncertainty')
        if getattr(preds_test, 'ndim', 1) != 2 or preds_test.shape[1] < 2:
            raise ValueError(
                'CatBoost predict did not return mean+variance. '
                'Ensure training.params.loss_function == "RMSEWithUncertainty" '
                'and CatBoost version supports uncertainty prediction.'
            )
        mean_test = preds_test[:, 0]
        var_preds_test = preds_test[:, 1]
        std_test = np.sqrt(np.maximum(var_preds_test, 1e-12))
        probs_test = norm.cdf(0, mean_test - 0.5, std_test)
        y_test_bin = (y_test_arr < 0.5).astype(int)
        eval_metrics_raw = compute_all_metrics(y_test_bin, probs_test, y_margin_true=None, mu=None, var=None, weights=None)
        eval_metrics = _to_eval_prefixed(eval_metrics_raw)
        
        log_metrics_dict(train_metrics)
        log_metrics_dict(eval_metrics)

        _log_calibration_plot(y_test_bin, probs_test, "CatBoost", now, "cat")

        scalers_src = proj_paths.output / 'scalers.pkl'
        try:
            if not scalers_src.exists():
                with open(scalers_src, 'wb') as f:
                    pickle.dump(scalers, f)
            _log_artifacts_to_mlflow(model, config, scalers_src, '.cbm')
        except Exception as e:
            print(f"Warning: Failed to log artifacts to MLflow: {e}")

        model_path = None
        config_path = None
        scalers_dst = None
        
        if save_local_artifacts:
            model_path = proj_paths.models / f'cb_model-{now}-cat.cbm'
            model.save_model(model_path.as_posix())
            config_path = proj_paths.models / f'run_config-{now}-cat.yaml'
            config_path.write_text(OmegaConf.to_yaml(config))
            scalers_dst = proj_paths.models / f'scalers-{now}-cat.pkl'
            if scalers_src.exists():
                shutil.copyfile(scalers_src.as_posix(), scalers_dst.as_posix())
        else:
            model_path = proj_paths.output / f'temp_cb_model-{now}-cat.cbm'
            model.save_model(model_path.as_posix())

        fi = model.get_feature_importance()
        fi_s = pd.Series(fi, index=X_train.drop(columns=['game_id'], errors='ignore').columns)
        
        _log_feature_importance_plots({'catboost_importance': fi_s}, now, 'cat')
        
        fi_path = proj_paths.output / f'feature_importance-{now}-cat.csv'
        _log_csv_artifact(fi_s, fi_path, save_local_artifacts)

        if not save_local_artifacts:
            scalers_file = proj_paths.output / 'scalers.pkl'
            scalers_file.unlink(missing_ok=True)

        run_id = run.info.run_id
    return TrainingResult(model_path, config_path, scalers_dst if scalers_src.exists() else None, eval_metrics, run_id)


def train_bayesian(config: OmegaConf, data: DataSet, scalers: dict, extra_tags: dict | None = None, pruner: Callable[[int, dict[str, float]], bool] | None = None, save_local_artifacts: bool = True) -> TrainingResult:
    """
    Train Bayesian NN on NBA game outcomes.

    Params:
        config (OmegaConf): Run configuration; must specify 'bayesian_nn'.
        data (DataSet): Prepared dataset.
        scalers (dict): Fitted scalers.
        extra_tags (dict | None): Additional MLflow tags.
        pruner: Optuna pruner (for hyperparameter optimization).
        save_local_artifacts (bool): Whether to save timestamped artifacts to local models directory.

    Returns:
        TrainingResult: Paths and MLflow run ID for this run.
    """
    assert config.training.algorithm == 'bayesian_nn', "Set training.algorithm to 'bayesian_nn' in run_config.yaml."

    train_dataset = NBADataset(config['inputs_team'], config['target'], data.X_train, data.teams, scalers)
    test_dataset = NBADataset(config['inputs_team'], config['target'], data.X_test, data.teams, scalers)
    train_loader = __import__('torch').utils.data.DataLoader(train_dataset, batch_size=config['training']['params']['batch_size'], shuffle=True)
    test_loader = __import__('torch').utils.data.DataLoader(test_dataset, batch_size=config['training']['params']['batch_size'], shuffle=False)

    try:
        _torch = __import__('torch')
        device = 'cuda' if getattr(_torch.cuda, 'is_available', lambda: False)() else 'cpu'
    except Exception:
        device = 'cpu'
    optimizer = __import__('torch').optim.Adam
    scheduler_cls = __import__('torch').optim.lr_scheduler.ExponentialLR
    model_init = OmegaConf.create(dict(
        team_count=len(data.teams),
        team_features=len(config['inputs_team']) // 2,
        **config['training']['model_config']
    ))
    # Optional seed for reproducibility in seed study
    seed = config['training']['params'].get('seed', config['training']['params'].get('random_state'))
    if seed is not None:
        try:
            import random
            import torch as _torch
            random.seed(int(seed))
            np.random.seed(int(seed))
            _torch.manual_seed(int(seed))
        except Exception:
            pass
    model = __import__('nba_ingame_prob.model.bayesian', fromlist=['BayesianResultPredictor']).BayesianResultPredictor(**model_init).to(device)
    opt = optimizer(model.parameters(), lr=config['training']['params']['learning_rate'], weight_decay=config['training']['params'].get('weight_decay', 0.0))
    scheduler = scheduler_cls(opt, gamma=config['training']['params'].get('lr_gamma', 1.0))

    now_run = time.strftime('%Y%m%d-%H%M%S')
    trial_no = extra_tags.get('trial_no') if extra_tags else None
    params = _params_to_dict(config.get('training', {}).get('params', {}))
    run_name = _build_run_name(
        'Bayesian NN',
        params,
        trial_no,
        now_run,
        [
            ('learning_rate', 'lr'),
            ('weight_decay', 'wd'),
            ('lr_gamma', 'gamma'),
            ('team_hidden_dim', 'team_hidden'),
            ('res_hidden_dim', 'res_hidden'),
            ('team_layers', 'team_layers'),
            ('res_layers', 'res_layers'),
            ('embedding_dim', 'emb'),
        ],
    )

    with _start_training_run(run_name, config['experiment_name']) as run:
        set_tags({'model_family': 'bayesian'})
        if extra_tags:
            set_tags(extra_tags)
        last_metrics = run_training(model, train_loader, test_loader, config, opt, device=device, scheduler=scheduler, timestamp=now_run, pruner=pruner)
        
        # TODO: Add calibration plot for Bayesian model
        # This requires getting final predictions on validation set after training
        # For now, the Bayesian model calibration is computed within the training loop
        
        best_src = proj_paths.output / (config['model_name'] + '_best.pth')
        scalers_src = proj_paths.output / 'scalers.pkl'
        try:
            if not scalers_src.exists():
                with open(scalers_src, 'wb') as f:
                    pickle.dump(scalers, f)
            _log_artifacts_to_mlflow(best_src, config, scalers_src, '.pth', model_init=model_init)
        except Exception as e:
            print(f"Warning: Failed to log artifacts to MLflow: {e}")
        
        model_init_path = None
        config_path = None
        model_path = None
        scalers_dst = None
        
        if save_local_artifacts:
            model_init_path = proj_paths.models / ('model_init-' + now_run + '.yaml')
            model_init_path.write_text(OmegaConf.to_yaml(model_init))
            config_path = proj_paths.models / ('run_config-' + now_run + '.yaml')
            config_path.write_text(OmegaConf.to_yaml(config))
            best_src = proj_paths.output / (config['model_name'] + '_best.pth')
            model_path = proj_paths.models / (config['model_name'] + '-' + now_run + '.pth')
            if best_src.exists():
                shutil.copyfile(best_src.as_posix(), model_path.as_posix())
            scalers_src = proj_paths.output / 'scalers.pkl'
            scalers_dst = proj_paths.models / ('scalers-' + now_run + '.pkl')
            if scalers_src.exists():
                shutil.copyfile(scalers_src.as_posix(), scalers_dst.as_posix())
        else:
            import glob
            checkpoint_pattern = str(proj_paths.output / f"{config['model_name']}-{now_run}-*.pth")
            for checkpoint_file in glob.glob(checkpoint_pattern):
                Path(checkpoint_file).unlink(missing_ok=True)
            best_model_file = proj_paths.output / (config['model_name'] + '_best.pth')
            best_model_file.unlink(missing_ok=True)
            scalers_file = proj_paths.output / 'scalers.pkl'
            scalers_file.unlink(missing_ok=True)
        run_id = run.info.run_id
    return TrainingResult(model_path, config_path, scalers_dst, last_metrics or {}, run_id)


def train_from_config(config: OmegaConf, data: DataSet, scalers: dict, extra_tags: dict | None = None, save_local_artifacts: bool = True) -> TrainingResult:
    """
    Dispatch to the appropriate trainer based on config.training.algorithm.

    Params:
        config (OmegaConf): Run configuration.
        data (DataSet): Prepared dataset bundle.
        scalers (dict): Fitted scalers.
        extra_tags (dict | None): Additional MLflow tags.
        save_local_artifacts (bool): Whether to save timestamped artifacts to local models directory.

    Returns:
        TrainingResult: Output of the selected trainer.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    algo = config.training.algorithm
    if algo == 'linear_logistic':
        return train_linear_logistic(config, data, scalers, extra_tags=extra_tags, save_local_artifacts=save_local_artifacts)
    if algo == 'linear_regression':
        return train_linear_regression(config, data, scalers, extra_tags=extra_tags, save_local_artifacts=save_local_artifacts)
    if algo == 'catboost_regressor':
        return train_catboost(config, data, scalers, extra_tags=extra_tags, save_local_artifacts=save_local_artifacts)
    if algo == 'bayesian_nn':
        return train_bayesian(config, data, scalers, extra_tags=extra_tags, save_local_artifacts=save_local_artifacts)
    raise ValueError(f"Unsupported training.algorithm: {algo}")
