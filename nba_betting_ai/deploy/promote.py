from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from mlflow.tracking import MlflowClient

from nba_betting_ai.consts import proj_paths


def get_top_runs(
    experiment_name: str,
    model_family: str,
    top_k: int = 1,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    base = f"tags.model_family = '{model_family}'"
    if filter_string:
        base += f" and ({filter_string})"
    order = order_by or ["metrics.eval_log_loss ASC"]
    return client.search_runs([exp.experiment_id], filter_string=base, order_by=order, max_results=top_k)


def _family_dir(model_family: str) -> Path:
    return proj_paths.models / 'production' / model_family


def _dst_for(model_family: str, run_id: str, artifact: str) -> Path:
    ext = Path(artifact).suffix
    if model_family == 'catboost' and ext == '.cbm':
        return _family_dir(model_family) / f'cb_model-{run_id}.cbm'
    if model_family == 'bayesian' and ext == '.pth':
        return _family_dir(model_family) / f'bayesian_model-{run_id}.pth'
    if artifact.startswith('run_config') and ext == '.yaml':
        return _family_dir(model_family) / f'run_config-{run_id}.yaml'
    if artifact.startswith('model_init') and ext == '.yaml':
        return _family_dir(model_family) / f'model_init-{run_id}.yaml'
    if artifact.startswith('scalers') and ext == '.pkl':
        return _family_dir(model_family) / f'scalers-{run_id}.pkl'
    return _family_dir(model_family) / f'{Path(artifact).stem}-{run_id}{ext}'


def stage_run_artifacts(run, model_family: str, keep: int = 3) -> list[Path]:
    client = MlflowClient()
    out_dir = _family_dir(model_family)
    out_dir.mkdir(parents=True, exist_ok=True)

    arts = client.list_artifacts(run.info.run_id)
    wanted = [a for a in arts if Path(a.path).suffix in {'.cbm', '.pth', '.yaml', '.pkl'}]
    staged: list[Path] = []
    for a in wanted:
        src = client.download_artifacts(run.info.run_id, a.path)
        dst = _dst_for(model_family, run.info.run_id, a.path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        staged.append(dst)

    # retention
    existing = sorted(out_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in existing[keep:]:
        try:
            old.unlink()
        except Exception:
            pass
    return staged


def promote_best(
    experiment_name: str,
    families: Iterable[str] = ('catboost', 'bayesian'),
    top_k: int = 1,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
    keep: int = 3,
) -> dict[str, list[Path]]:
    promoted: dict[str, list[Path]] = {}
    for fam in families:
        runs = get_top_runs(experiment_name, fam, top_k=top_k, filter_string=filter_string, order_by=order_by)
        files: list[Path] = []
        for r in runs:
            files.extend(stage_run_artifacts(r, fam, keep=keep))
        promoted[fam] = files
    return promoted

