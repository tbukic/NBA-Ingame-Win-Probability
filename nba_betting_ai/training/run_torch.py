import mlflow
import shutil
import torch
import math
from collections.abc import MutableMapping
from enum import Enum
from functools import partial
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Any, Callable
from tqdm import tqdm

from nba_betting_ai.consts import proj_paths
from nba_betting_ai.training.metrics import expected_calibration_error


def convert_flatten(d, parent_key ='', sep ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
 
        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class MetricReporter:
    def __init__(self, update_step: int, epochs: int):
        self._update_step = update_step
        self._loss_key = None
        self._epochs = epochs

    def live_report(self, metrics: dict[str, Any], progress_bar: tqdm, epoch: int, force: bool = False) -> None:
        step = progress_bar.n
        if not self._loss_key:
            self._loss_key = next(filter(lambda key: 'loss' in key, metrics))
        progress_bar.set_description(f'loss: {metrics[self._loss_key]:.4f} | epoch:  {epoch}/{self._epochs}')
        progress_bar.update()
        if step % self._update_step == 0 or force:
            mlflow.log_metrics(metrics, step=step)


class Mode(Enum):
    TRAIN = 'train'
    EVAL = 'eval'

def _noop(_):
    """No-op function."""
    pass

def run_training(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: MutableMapping[Any, Any],
        optimizer: Optimizer,
        device: str = 'cpu',
        scheduler: LRScheduler | None = None,
        timestamp: str = 'missing',
        pruner: Callable[[int, dict[str, float]], bool] | None = None,
    ) -> dict[str, float] | None:
    scalers_path = proj_paths.output / 'scalers.pkl'
    config_archive = proj_paths.config.default.with_stem(f'config-{timestamp}')

    flat_config = convert_flatten(config)
    mlflow.log_params(flat_config)
    mlflow.log_artifact(scalers_path)
    mlflow.log_artifact(proj_paths.config.default)
    shutil.copy(proj_paths.config.default.as_posix(), config_archive.as_posix())
    params = config['training']['params']
    metric_reporter = MetricReporter(update_step=20, epochs=params['epochs'])
    best_loss = float('inf')
    with tqdm(total=params['epochs']*len(train_loader), desc='Training Progress') as progress_bar:
        with torch.no_grad():
            test_metrics = run_epoch(model, test_loader, Mode.EVAL, device, params['weighted_loss'], optimizer)
            mlflow.log_metrics(test_metrics, step=0)
        for epoch in range(params['epochs']):
            report_metric = partial(metric_reporter.live_report, progress_bar=progress_bar, epoch=epoch)
            train_metrics = run_epoch(model, train_loader, Mode.TRAIN, device, params['weighted_loss'], optimizer, report_metric)
            if scheduler:
                scheduler.step()
            metric_reporter.live_report(train_metrics, progress_bar, epoch, force=True)
            step = progress_bar.n
            mlflow.log_metrics(train_metrics, step=step)
            if epoch % params['eval_freq'] != 0 and epoch < params['epochs'] - 1:
                continue
            with torch.no_grad():
                test_metrics = run_epoch(model, test_loader, Mode.EVAL, device, params['weighted_loss'], optimizer)
                mlflow.log_metrics(test_metrics, step=step)
            # Optuna-style pruning hook (non-weighted eval_log_loss)
            if pruner is not None and pruner(epoch, test_metrics):
                break
            avg_loss = test_metrics['eval_avg_loss']
            model_path = proj_paths.output / f"{config['model_name']}-{timestamp}-{step}.pth"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = proj_paths.output / f"{config['model_name']}_best.pth"
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(best_model_path)
    return test_metrics if 'test_metrics' in locals() else None


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    mode: str = 'eval',
    device: str = 'cpu',
    weighted_loss: bool = False,
    optimizer: Optimizer | None = None,
    live_report_metrics: Callable[[dict], None] = _noop
) -> dict[str, float]:
    if mode is Mode.TRAIN:
        model.train()
        optimizer.zero_grad()
    elif mode is Mode.EVAL:
        model.eval()
    correct = 0
    avg_loss = 0
    avg_var = 0
    n = 0
    const = 0.5 * torch.log(2 * torch.tensor(torch.pi, dtype=torch.float64)).to(device)
    # Unified metrics accumulators (non-weighted)
    eps = 1e-6
    sum_brier = 0.0
    sum_logloss = 0.0
    sum_nll = 0.0
    valid_brier_n = 0
    valid_logloss_n = 0
    prob_nonfinite = 0
    probs_ece: list[torch.Tensor] = []
    targets_ece: list[torch.Tensor] = []
    for data in data_loader:
        data = {
            k: v.to(device)
            for k, v in data.items()
        }
        target = data.pop('y')
        weights = (1 - 0.5*data['time_remaining']).detach()
        output = model(**data)
        mu, s = torch.chunk(output, 2, dim=-1)
        var = F.softplus(s) + eps
        logvar = torch.log(var)
        loss = const + logvar + (target - mu)**2 / var
        loss = torch.mean(loss)
        next_n = n + len(target)
        avg_loss = avg_loss*(n/next_n) + loss.item()*(len(target)/next_n)
        avg_var = avg_var*(n/next_n) + torch.mean(var).item()*(len(target)/next_n)
        n = next_n
        correct += (mu*target >= 0).sum().item()
        # Unified metrics: probability of home win via Gaussian CDF
        std = torch.sqrt(var)
        prob = 0.5 * (1.0 + torch.erf((-mu) / (std * math.sqrt(2.0))))
        y_bin = (target < 0).float()
        # Brier and log-loss (unweighted) with robust masking against non-finite probs
        finite_mask = torch.isfinite(prob)
        prob_nonfinite += int((~finite_mask).sum().item())
        if finite_mask.any():
            pb = prob[finite_mask]
            yb = y_bin[finite_mask]
            sum_brier += torch.sum((pb - yb) ** 2).item()
            prob_clipped = torch.clamp(pb, eps, 1 - eps)
            sum_logloss += torch.sum(- (yb*torch.log(prob_clipped) + (1 - yb)*torch.log(1 - prob_clipped))).item()
            valid_brier_n += int(finite_mask.sum().item())
            valid_logloss_n += int(finite_mask.sum().item())
            probs_ece.append(pb.detach().cpu())
            targets_ece.append(yb.detach().cpu())
        # NLL with 0.5 factors (non-weighted)
        nll = const + 0.5*torch.log(var + eps) + 0.5*((target - mu) ** 2) / (var + eps)
        sum_nll += torch.sum(nll).item()
        metrics = {
            f'{mode.value}_accuracy': correct / n,
            f'{mode.value}_avg_var': avg_var,
            f'{mode.value}_avg_loss': avg_loss,
        }
        if mode is Mode.TRAIN:
            if weighted_loss:
                (loss * weights.mean()).backward()
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        live_report_metrics(metrics)
    # Finalize unified metrics
    if n > 0:
        metrics[f'{mode.value}_brier_score'] = sum_brier / max(valid_brier_n, 1)
        metrics[f'{mode.value}_log_loss'] = sum_logloss / max(valid_logloss_n, 1)
        metrics[f'{mode.value}_nll'] = sum_nll / n
        metrics[f'{mode.value}_prob_nonfinite'] = float(prob_nonfinite)
        if probs_ece:
            probs_all = torch.cat(probs_ece).numpy()
            targets_all = torch.cat(targets_ece).numpy()
            metrics[f'{mode.value}_ece'] = expected_calibration_error(targets_all, probs_all)
    return metrics
