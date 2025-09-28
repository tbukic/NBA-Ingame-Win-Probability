import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def flatten_params(d: MutableMapping[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten a nested mapping using dot-separated keys.

    Params:
        d (MutableMapping[str, Any]): Mapping to flatten.
        parent_key (str): Key prefix for recursion.
        sep (str): Separator between nested keys.

    Returns:
        dict[str, Any]: Flattened mapping.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_params(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_params_flat(config: MutableMapping[str, Any]) -> None:
    """
    Log a nested configuration dict into MLflow as flattened params.

    Params:
        config (MutableMapping[str, Any]): Configuration mapping.
    """
    params = flatten_params(config)
    mlflow.log_params(params)


def log_metrics_dict(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log a metrics dictionary to the active MLflow run.

    Params:
        metrics (dict[str, float]): Metric name to value mapping.
        step (int | None): Optional training step.
    """
    if step is None:
        mlflow.log_metrics(metrics)
    else:
        mlflow.log_metrics(metrics, step=step)


def log_artifact_path(path: Path) -> None:
    """
    Log a single file as an artifact to the active MLflow run.

    Params:
        path (Path): File path to log.
    """
    mlflow.log_artifact(path.as_posix())


def log_artifacts_in_dir(dir_path: Path) -> None:
    """
    Recursively log all files under the given directory as MLflow artifacts.

    Params:
        dir_path (Path): Directory to traverse.
    """
    if not dir_path.exists():
        return
    for root, _, files in os.walk(dir_path):
        for f in files:
            mlflow.log_artifact(Path(root, f).as_posix())


def set_tags(tags: dict[str, Any]) -> None:
    """
    Set multiple MLflow tags on the active run.

    Params:
        tags (dict[str, Any]): Tag mapping.
    """
    mlflow.set_tags(tags)


def ensure_experiment(name: str) -> str:
    """
    Ensure an MLflow experiment with the given name exists and is active.

    - If missing: creates it via mlflow.set_experiment(name)
    - If present and deleted: restores it and sets active

    Returns:
        str: Experiment ID
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        # Create fresh
        mlflow.set_experiment(name)
        exp2 = client.get_experiment_by_name(name)
        return exp2.experiment_id if exp2 is not None else '0'

    # If the experiment exists but is soft-deleted, prefer permanent deletion if supported,
    # otherwise fall back to creating a new unique experiment name.
    if getattr(exp, 'lifecycle_stage', '') == 'deleted':
        # Prefer restoring the original experiment instead of creating timestamped variants.
        try:
            client.restore_experiment(exp.experiment_id)
            mlflow.set_experiment(name)
            restored = client.get_experiment_by_name(name)
            if restored is not None:
                return restored.experiment_id
        except Exception:
            pass
        # Fall back to creating a fresh experiment with the requested name.
        mlflow.set_experiment(name)
        exp2 = client.get_experiment_by_name(name)
        if exp2 is not None:
            return exp2.experiment_id
        return '0'

    # Normal active case: just set as current and return id
    mlflow.set_experiment(name)
    return exp.experiment_id
