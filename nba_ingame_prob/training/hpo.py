import optuna
import mlflow
import numpy as np
import os
import tempfile
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path

from nba_ingame_prob.consts import proj_paths
from nba_ingame_prob.model.inputs import prepare_scalers
from nba_ingame_prob.training.pipeline import prepare_data
from nba_ingame_prob.training.trainers import (
    train_catboost,
    train_bayesian,
    train_linear_logistic,
    train_linear_regression,
)
from nba_ingame_prob.training.mlflow_utils import ensure_experiment


def _prepare_run(config):
    data = prepare_data(**config["data_params"]) 
    # Fit team one-hot encoders only for linear models; keep categorical/index for others
    fit_ohe = config['training']['algorithm'] in ('linear_logistic', 'linear_regression')
    scalers = prepare_scalers(
        data.X_train,
        config['inputs_team'] or [],
        proj_paths.output / 'scalers.pkl',
        fit_onehot_teams=fit_ohe,
    )
    return data, scalers


def _value(metrics: dict[str, float]) -> float:
    return float(metrics.get('eval_log_loss', metrics.get('log_loss')))


def _log_failed_trial(family: str, cfg, trial: optuna.Trial, err: Exception):
    """
    Create a small FAILED run in MLflow so failed trials are visible.
    """
    try:
        active = mlflow.active_run()
        payload = {
            'family': family,
            'trial': trial.number,
            'study': trial.study.study_name,
            'experiment_name': cfg['experiment_name'],
            'algorithm': cfg['training']['algorithm'],
            'error': str(err)[:500],
        }
        if active:
            artifact_name = f"failures/trial_{trial.number:04d}.json"
            mlflow.log_dict(payload, artifact_name)
        else:
            exp_name = f"{cfg.get('experiment_name', 'NBA')}-HPO-FAILS"
            ensure_experiment(exp_name)
            run_name = f"{family}-hpo-fail-{trial.number}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({'model_family': family, 'hpo_study': trial.study.study_name, 'trial_no': str(trial.number), 'status': 'failed'})
                mlflow.log_dict(payload, 'failure.json')
    except Exception:
        # Swallow logging errors to avoid masking the trial failure cause
        pass


def _ensure_safety_catboost(p: dict):
    """
    Enforce safe/default flags for CatBoost during HPO to avoid degenerate solutions.
    """
    p['logging_level'] = 'Silent'
    p['allow_writing_files'] = False
    p['use_best_model'] = True
    p['early_stopping_rounds'] = 200
    p['metric_period'] = 200


def run_study_catboost(config_path: Path, study_name: str = 'cb_default', n_trials: int = 50, experiment_name: str | None = None, n_jobs: int = 1):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name

    def objective(trial: optuna.Trial):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        # Safe search bounds
        p['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        p['max_depth'] = trial.suggest_int('max_depth', 4, 9)
        p['iterations'] = trial.suggest_int('iterations', 500, 1500)
        p['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 5e-2, 10.0, log=True)
        p['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
        p['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        p['max_bin'] = trial.suggest_int('max_bin', 32, 128)
        _ensure_safety_catboost(p)
        # Limit internal threads so parallel trials don't oversubscribe CPUs
        try:
            import os
            p['thread_count'] = max(1, os.cpu_count() // max(1, n_jobs))
        except Exception:
            pass
        try:
            data, scalers = _prepare_run(cfg)
            tags = {'hpo_study': study_name, 'trial_no': str(trial.number)}
            res = train_catboost(cfg, data, scalers, extra_tags=tags, save_local_artifacts=False)
            trial.set_user_attr('run_id', res.run_id)
            return _value(res.metrics)
        except Exception as e:
            trial.set_user_attr('error', str(e)[:500])
            _log_failed_trial('catboost', cfg, trial, e)
            return float('inf')

    exp_name = base['experiment_name']
    exp_id = ensure_experiment(exp_name)
    run_name = f"CatBoost HPO ({study_name}) [lr,max_depth,iterations,l2]"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags({'model_family': 'catboost', 'hpo_study': study_name, 'hpo_driver': 'optuna', 'algorithm': base['training']['algorithm']})
        mlflow.log_param('n_trials', n_trials)
        mlflow.log_param('n_jobs', n_jobs)
        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metric('completed_trials', len(completed))
        if completed:
            best = study.best_trial
            mlflow.log_metric('best_value', float(best.value))
            mlflow.log_param('best_trial', best.number)
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
    return study


def run_study_linear_logistic(config_path: Path, study_name: str = 'lin_log_default', n_trials: int = 50, experiment_name: str | None = None, n_jobs: int = 1):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name

    def objective(trial: optuna.Trial):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        p['C'] = trial.suggest_float('C', 1e-3, 100.0, log=True)
        try:
            data, scalers = _prepare_run(cfg)
            tags = {'hpo_study': study_name, 'trial_no': str(trial.number)}
            res = train_linear_logistic(cfg, data, scalers, extra_tags=tags, save_local_artifacts=False)
            trial.set_user_attr('run_id', res.run_id)
            return _value(res.metrics)
        except Exception as e:
            trial.set_user_attr('error', str(e)[:500])
            _log_failed_trial('linear_logistic', cfg, trial, e)
            return float('inf')

    exp_name = base['experiment_name']
    exp_id = ensure_experiment(exp_name)
    run_name = f"Linear Logistic HPO ({study_name}) [C]"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags({'model_family': 'linear', 'linear_type': 'logistic', 'hpo_study': study_name, 'hpo_driver': 'optuna', 'algorithm': base['training']['algorithm']})
        mlflow.log_param('n_trials', n_trials)
        mlflow.log_param('n_jobs', n_jobs)
        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metric('completed_trials', len(completed))
        if completed:
            best = study.best_trial
            mlflow.log_metric('best_value', float(best.value))
            mlflow.log_param('best_trial', best.number)
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
    return study


def run_study_linear_regression(config_path: Path, study_name: str = 'lin_reg_default', n_trials: int = 50, experiment_name: str | None = None, n_jobs: int = 1):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name

    def objective(trial: optuna.Trial):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        p['alpha'] = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
        try:
            data, scalers = _prepare_run(cfg)
            tags = {'hpo_study': study_name, 'trial_no': str(trial.number)}
            res = train_linear_regression(cfg, data, scalers, extra_tags=tags, save_local_artifacts=False)
            trial.set_user_attr('run_id', res.run_id)
            return _value(res.metrics)
        except Exception as e:
            trial.set_user_attr('error', str(e)[:500])
            _log_failed_trial('linear_regression', cfg, trial, e)
            return float('inf')

    exp_name = base['experiment_name']
    exp_id = ensure_experiment(exp_name)
    run_name = f"Linear Regression HPO ({study_name}) [alpha]"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags({'model_family': 'linear', 'linear_type': 'regression', 'hpo_study': study_name, 'hpo_driver': 'optuna', 'algorithm': base['training']['algorithm']})
        mlflow.log_param('n_trials', n_trials)
        mlflow.log_param('n_jobs', n_jobs)
        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metric('completed_trials', len(completed))
        if completed:
            best = study.best_trial
            mlflow.log_metric('best_value', float(best.value))
            mlflow.log_param('best_trial', best.number)
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
    return study


def run_study_bayesian(config_path: Path, study_name: str = 'bayes_default', n_trials: int = 50, experiment_name: str | None = None):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name

    def objective(trial: optuna.Trial):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        m = cfg['training']['model_config']
        # Shorten training during HPO for quicker feedback
        if 'epochs' in p:
            p['epochs'] = min(int(p['epochs']), 25)
        p['learning_rate'] = trial.suggest_float('learning_rate', 5e-4, 5e-2, log=True)
        p['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        p['lr_gamma'] = trial.suggest_float('lr_gamma', 0.8, 1.0)
        m['team_hidden_dim'] = trial.suggest_int('team_hidden_dim', 4, 32)
        m['team_layers'] = trial.suggest_int('team_layers', 1, 3)
        m['res_hidden_dim'] = trial.suggest_int('res_hidden_dim', 4, 64)
        m['res_layers'] = trial.suggest_int('res_layers', 2, 6)
        m['embedding_dim'] = trial.suggest_categorical('embedding_dim', [None, 1, 2, 4, 8, 16])
        try:
            data, scalers = _prepare_run(cfg)
            def pruner(epoch: int, metrics: dict[str, float]) -> bool:
                val = float(metrics.get('eval_log_loss', metrics.get('log_loss', 1e9)))
                trial.report(val, step=epoch)
                return trial.should_prune()
            tags = {'hpo_study': study_name, 'trial_no': str(trial.number)}
            res = train_bayesian(cfg, data, scalers, extra_tags=tags, pruner=pruner, save_local_artifacts=False)
            trial.set_user_attr('run_id', res.run_id)
            return _value(res.metrics)
        except Exception as e:
            trial.set_user_attr('error', str(e)[:500])
            _log_failed_trial('bayesian', cfg, trial, e)
            return float('inf')

    exp_name = base['experiment_name']
    exp_id = ensure_experiment(exp_name)
    run_name = f"Bayesian HPO ({study_name}) [lr,weight_decay,lr_gamma,team_hidden,res_hidden]"
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags({'model_family': 'bayesian', 'hpo_study': study_name, 'hpo_driver': 'optuna', 'algorithm': base['training']['algorithm']})
        mlflow.log_param('n_trials', n_trials)
        study = optuna.create_study(direction='minimize', study_name=study_name, pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metric('completed_trials', len(completed))
        if completed:
            best = study.best_trial
            mlflow.log_metric('best_value', float(best.value))
            mlflow.log_param('best_trial', best.number)
            mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
    return study


def replicate_catboost_best(config_path: Path, study: optuna.Study, n_repeats: int | None = 5, seeds: list[int] | None = None, experiment_name: str | None = None):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name
    best = study.best_trial
    pbest = {k: v for k, v in best.params.items()}
    if not seeds and n_repeats is None:
        raise ValueError("Either n_repeats or seeds must be provided")
    seeds = seeds or [42 + i for i in range(n_repeats)]
    results = []
    for i, sd in enumerate(seeds, start=1):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        p.update(pbest)
        p['random_state'] = int(sd)
        data, scalers = _prepare_run(cfg)
        tags = {'seed_study': 'catboost', 'replicate': str(i), 'seed': str(sd), 'parent_study': study.study_name, 'best_trial': str(best.number)}
        res = train_catboost(cfg, data, scalers, extra_tags=tags)
        results.append(res)
    return results


def replicate_bayesian_best(config_path: Path, study: optuna.Study, n_repeats: int | None = 5, seeds: list[int] | None = None, experiment_name: str | None = None):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name
    best = study.best_trial
    pbest = {k: v for k, v in best.params.items()}
    if not seeds and n_repeats is None:
        raise ValueError("Either n_repeats or seeds must be provided")
    seeds = seeds or [42 + i for i in range(n_repeats)]
    results = []
    for i, sd in enumerate(seeds, start=1):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        m = cfg['training']['model_config']
        for k, v in pbest.items():
            if k in p:
                p[k] = v
            elif k in m:
                m[k] = v
        p['seed'] = int(sd)
        data, scalers = _prepare_run(cfg)
        tags = {'seed_study': 'bayesian', 'replicate': str(i), 'seed': str(sd), 'parent_study': study.study_name, 'best_trial': str(best.number)}
        res = train_bayesian(cfg, data, scalers, extra_tags=tags)
        results.append(res)
    return results


def replicate_linear_logistic_best(config_path: Path, study: optuna.Study, n_repeats: int | None = 5, seeds: list[int] | None = None, experiment_name: str | None = None):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name
    best = study.best_trial
    pbest = {k: v for k, v in best.params.items()}
    if not seeds and n_repeats is None:
        raise ValueError("Either n_repeats or seeds must be provided")
    seeds = seeds or [42 + i for i in range(n_repeats)]
    results = []
    for i, sd in enumerate(seeds, start=1):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        p.update(pbest)
        p['random_state'] = int(sd)
        data, scalers = _prepare_run(cfg)
        tags = {'seed_study': 'linear_logistic', 'replicate': str(i), 'seed': str(sd), 'parent_study': study.study_name, 'best_trial': str(best.number)}
        res = train_linear_logistic(cfg, data, scalers, extra_tags=tags)
        results.append(res)
    return results


def replicate_linear_regression_best(config_path: Path, study: optuna.Study, n_repeats: int | None = 5, seeds: list[int] | None = None, experiment_name: str | None = None):
    base = OmegaConf.load(config_path)
    if experiment_name:
        base = deepcopy(base)
        base['experiment_name'] = experiment_name
    best = study.best_trial
    pbest = {k: v for k, v in best.params.items()}
    if not seeds and n_repeats is None:
        raise ValueError("Either n_repeats or seeds must be provided")
    seeds = seeds or [42 + i for i in range(n_repeats)]
    results = []
    for i, sd in enumerate(seeds, start=1):
        cfg = deepcopy(base)
        p = cfg['training']['params']
        p.update(pbest)
        p['random_state'] = int(sd)
        data, scalers = _prepare_run(cfg)
        tags = {'seed_study': 'linear_regression', 'replicate': str(i), 'seed': str(sd), 'parent_study': study.study_name, 'best_trial': str(best.number)}
        res = train_linear_regression(cfg, data, scalers, extra_tags=tags)
        results.append(res)
    return results


def run_hpo_with_override(hpo_fn, config_path: Path, study_name: str, experiment_name: str, n_trials: int, reserve_validation: bool | None = None, n_jobs: int = 1):
    """
    Run HPO with optional config override for reserve_validation.
    
    Args:
        hpo_fn: The HPO function to call
        config_path: Path to the base config file
        study_name: Name for the Optuna study
        experiment_name: MLflow experiment name
        n_trials: Number of trials to run
        reserve_validation: Override for reserve_validation setting
        n_jobs: Number of parallel jobs (for CatBoost)
    
    Returns:
        Optuna Study object
    """
    if reserve_validation is None:
        return hpo_fn(config_path, study_name=study_name, experiment_name=experiment_name, n_trials=n_trials)
    
    cfg = OmegaConf.load(config_path)
    cfg = deepcopy(cfg)
    if 'data_params' not in cfg:
        cfg['data_params'] = {}
    cfg['data_params']['reserve_validation'] = reserve_validation
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(cfg, f.name)
        tmp = f.name
    try:
        if 'catboost' in hpo_fn.__name__:
            result = hpo_fn(tmp, study_name=study_name, experiment_name=experiment_name, n_trials=n_trials, n_jobs=n_jobs)
        else:
            result = hpo_fn(tmp, study_name=study_name, experiment_name=experiment_name, n_trials=n_trials)
    finally:
        os.unlink(tmp)
    return result


def replicate_with_custom_config(replicate_fn, config_path: Path, study, seeds: list[int], experiment_name: str, reserve_validation: bool | None = None):
    """
    Run replication with optional config override for reserve_validation.
    
    Args:
        replicate_fn: The replication function to call
        config_path: Path to the base config file
        study: Optuna Study object with best parameters
        seeds: List of seeds to use for replication
        experiment_name: MLflow experiment name
        reserve_validation: Override for reserve_validation setting
    
    Returns:
        tuple: (results_list, summary_dict) where summary_dict contains best_seed, best_metric, etc.
    """
    if reserve_validation is None:
        results = replicate_fn(config_path, study, seeds=seeds, experiment_name=experiment_name)
        # Extract summary from results
        final_metrics = [r.metrics.get('eval_log_loss', 1e9) for r in results if r]
        best_seed_idx = np.argmin(final_metrics)
        summary = {
            "mean": np.mean(final_metrics),
            "std": np.std(final_metrics),
            "individual": final_metrics,
            "best_seed": seeds[best_seed_idx],
            "best_metric": final_metrics[best_seed_idx]
        }
        return results, summary
        
    cfg = OmegaConf.load(config_path)
    cfg = deepcopy(cfg)
    if 'data_params' not in cfg:
        cfg['data_params'] = {}
    cfg['data_params']['reserve_validation'] = reserve_validation
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        OmegaConf.save(cfg, f.name)
        temp_config_path = f.name
    try:
        results = replicate_fn(temp_config_path, study, seeds=seeds, experiment_name=experiment_name)
        # Extract summary from results
        final_metrics = [r.metrics.get('eval_log_loss', 1e9) for r in results if r]
        best_seed_idx = np.argmin(final_metrics)
        summary = {
            "mean": np.mean(final_metrics),
            "std": np.std(final_metrics),
            "individual": final_metrics,
            "best_seed": seeds[best_seed_idx],
            "best_metric": final_metrics[best_seed_idx]
        }
        return results, summary
    finally:
        os.unlink(temp_config_path)
