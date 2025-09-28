import math
import numpy as np
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)


def _safe_weights(y: np.ndarray, weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != y.shape[0]:
        raise ValueError("weights must be same length as y")
    return w


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width bins over [0, 1].

    Params:
        y_true (np.ndarray): Binary ground truth labels (0/1).
        y_prob (np.ndarray): Predicted probabilities in [0, 1].
        n_bins (int): Number of calibration bins.
        sample_weight (np.ndarray | None): Optional per-sample weights.

    Returns:
        float: ECE value in [0, 1]. Lower is better.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    sw = _safe_weights(y_true, sample_weight)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        w = sw[mask] if sw is not None else None
        conf = np.average(y_prob[mask], weights=w) if w is not None else y_prob[mask].mean()
        acc = np.average(y_true[mask], weights=w) if w is not None else y_true[mask].mean()
        frac = (w.sum() if w is not None else mask.sum()) / (sw.sum() if sw is not None else len(y_true))
        ece += frac * abs(acc - conf)
    return float(ece)


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Average Gaussian negative log-likelihood for given mean and variance.

    Params:
        y_true (np.ndarray): True regression targets.
        mu (np.ndarray): Predicted means.
        var (np.ndarray): Predicted variances (must be positive).
        sample_weight (np.ndarray | None): Optional per-sample weights.

    Returns:
        float: Mean negative log-likelihood.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    eps = 1e-12
    var = np.maximum(var, eps)
    const = 0.5 * math.log(2 * math.pi)
    nll = const + 0.5 * np.log(var) + 0.5 * (y_true - mu) ** 2 / var
    if sample_weight is not None:
        sw = _safe_weights(y_true, sample_weight)
        return float(np.average(nll, weights=sw))
    return float(nll.mean())


def compute_classification_metrics(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    prefix: str = "",
) -> dict:
    """
    Compute common classification metrics for probabilistic predictions.

    Params:
        y_true_bin (np.ndarray): Binary labels (0/1).
        y_prob (np.ndarray): Predicted probabilities.
        sample_weight (np.ndarray | None): Optional per-sample weights.
        prefix (str): Optional prefix for metric keys (e.g., 'w_').

    Returns:
        dict: Metrics including accuracy, brier_score, log_loss, ece.
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}
    w = _safe_weights(y_true_bin, sample_weight)
    metrics[f"{prefix}accuracy"] = float(accuracy_score(y_true_bin, y_pred, sample_weight=w))
    metrics[f"{prefix}brier_score"] = float(brier_score_loss(y_true_bin, y_prob, sample_weight=w))
    # Clip probs to avoid log(0)
    eps = 1e-12
    metrics[f"{prefix}log_loss"] = float(log_loss(y_true_bin, np.clip(y_prob, eps, 1 - eps), sample_weight=w))
    metrics[f"{prefix}ece"] = expected_calibration_error(y_true_bin, y_prob, sample_weight=w)
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    prefix: str = "",
) -> dict:
    """
    Compute MAE and RMSE for regression predictions.

    Params:
        y_true (np.ndarray): True targets.
        y_pred (np.ndarray): Predicted targets.
        sample_weight (np.ndarray | None): Optional per-sample weights.
        prefix (str): Optional prefix for metric keys (e.g., 'w_').

    Returns:
        dict: Metrics with keys '<prefix>mae' and '<prefix>rmse'.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    w = _safe_weights(y_true, sample_weight)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=w)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred, sample_weight=w))
    return {f"{prefix}mae": float(mae), f"{prefix}rmse": float(rmse)}


def compute_all_metrics(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    y_margin_true: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    var: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute a unified set of non-weighted metrics on the eval split.

    Params:
        y_true_bin (np.ndarray): Binary labels (0/1) for home win.
        y_prob (np.ndarray): Predicted probabilities for home win.
        y_margin_true (np.ndarray | None): True margins (if available).
        mu (np.ndarray | None): Predicted means (if available).
        var (np.ndarray | None): Predicted variances (if available).
        weights (np.ndarray | None): Ignored for metric reporting (kept for API compatibility).

    Returns:
        dict: Non-weighted metrics for selection and diagnostics.
    """
    metrics = {}
    metrics.update(compute_classification_metrics(y_true_bin, y_prob, prefix="", sample_weight=None))
    if y_margin_true is not None and mu is not None:
        metrics.update(compute_regression_metrics(y_margin_true, mu, prefix="", sample_weight=None))
    if y_margin_true is not None and mu is not None and var is not None:
        metrics["nll"] = gaussian_nll(y_margin_true, mu, var, sample_weight=None)
    return metrics
