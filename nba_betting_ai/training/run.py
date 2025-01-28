import mlflow
import shutil
import torch
from collections.abc import MutableMapping
from enum import Enum
from functools import partial
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Any, Callable
from tqdm import tqdm

from nba_betting_ai.consts import proj_paths


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

_noop = lambda _: None

def run_training(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: MutableMapping[Any, Any],
        optimizer: Optimizer,
        device: str = 'cpu',
        scheduler: LRScheduler | None = None,
        timestamp: str = 'missing',
    ) -> None:
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
            test_metrics = run_epoch(model, test_loader, Mode.EVAL, device, params['weighted_loss'], params['clamp_var'], optimizer)
            mlflow.log_metrics(test_metrics, step=0)
        for epoch in range(params['epochs']):
            report_metric = partial(metric_reporter.live_report, progress_bar=progress_bar, epoch=epoch)
            train_metrics = run_epoch(model, train_loader, Mode.TRAIN, device, params['weighted_loss'], params['clamp_var'], optimizer, report_metric)
            if scheduler:
                scheduler.step()
            metric_reporter.live_report(train_metrics, progress_bar, epoch, force=True)
            step = progress_bar.n
            mlflow.log_metrics(train_metrics, step=step)
            if epoch % params['eval_freq'] != 0 and epoch < params['epochs'] - 1:
                continue
            with torch.no_grad():
                test_metrics = run_epoch(model, test_loader, Mode.EVAL, device, params['weighted_loss'], params['clamp_var'], optimizer)
                mlflow.log_metrics(test_metrics, step=step)
            avg_loss = test_metrics['eval_avg_loss']
            model_path = proj_paths.output / f'{config['model_name']}-{timestamp}-loss_{str(avg_loss).replace('.', '_')}.pth'
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = proj_paths.output / f'{config['model_name']}_best.pth'
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(best_model_path)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    mode: str = 'eval',
    device: str = 'cpu',
    weighted_loss: bool = False,
    clamp_var: float | None = None,
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
    correct_w = 0
    avg_loss_w = 0
    avg_var_w = 0
    for data in data_loader:
        data = {
            k: v.to(device)
            for k, v in data.items()
        }
        target = data.pop('y')
        weights = (1 - 0.5*data['time_remaining']).detach()
        output = model(**data)
        mu, logvar = torch.chunk(output, 2, dim=-1)
        if clamp_var:
            logvar = torch.clamp(logvar, min=-clamp_var, max=clamp_var)
        var = torch.exp(logvar)
        var_w = var * weights
        loss = const + logvar + (target - mu)**2 / var
        loss_w = loss * weights
        loss = torch.mean(loss)
        loss_w = torch.mean(loss_w)
        next_n = n + len(target)
        avg_loss = avg_loss*(n/next_n) + loss.item()*(len(target)/next_n)
        avg_loss_w = avg_loss_w*(n/next_n) + loss_w.item()*(len(target)/next_n)
        avg_var = avg_var*(n/next_n) + torch.mean(var).item()*(len(target)/next_n)
        avg_var_w = avg_var_w*(n/next_n) + torch.mean(var_w).item()*(len(target)/next_n)
        n = next_n
        correct += (mu*target >= 0).sum().item()
        correct_w += ((mu*target >= 0)*weights).sum().item()
        metrics = {
            f'{mode.value}_accuracy': correct / n,
            f'{mode.value}_avg_var': avg_var,
            f'{mode.value}_avg_loss': avg_loss,
            f'{mode.value}_accuracy_w': correct_w / n,
            f'{mode.value}_avg_var_w': avg_var_w,
            f'{mode.value}_avg_loss_w': avg_loss_w,
        }
        if mode is Mode.TRAIN:
            if weighted_loss:
                loss_w.backward()
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        live_report_metrics(metrics)
    return metrics
