"""Scoring helpers: discrimination, calibration, and bootstrap intervals that resample whole weeks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "mean_pred": float(p.mean()),
        "observed": float(y.mean()),
        "n": int(len(y)),
    }


def row_log_loss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_ci(values: np.ndarray, groups: np.ndarray, stat=np.mean, n: int = 400, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% interval, resampling whole groups (arrival weeks) so correlated rows stay together."""
    rng = np.random.default_rng(seed)
    values, groups = np.asarray(values, float), np.asarray(groups)
    keys, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(keys))
    cnts = np.bincount(inv, minlength=len(keys)).astype(float)
    if stat is not np.mean:
        raise ValueError("cluster_ci only supports the mean")
    est = values.mean()
    draws = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        draws[i] = sums[pick].sum() / cnts[pick].sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def auc_ci(y, p, groups, n: int = 300, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    keys, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(keys))]
    est = roc_auc_score(y, p)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([members[k] for k in pick])
        if len(np.unique(y[idx])) == 2:
            draws.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def one_se_pick(order: list[str], losses: dict[str, np.ndarray], groups: np.ndarray) -> tuple[str, dict]:
    """Choose the simplest model whose mean loss is within one standard error of the best.

    `order` runs simplest to most complex. The standard error of each paired gap is taken over arrival weeks.
    """
    means = {k: float(v.mean()) for k, v in losses.items()}
    best = min(means, key=means.get)
    keys, inv = np.unique(groups, return_inverse=True)
    detail = {}
    chosen = None
    for name in order:
        diff = losses[name] - losses[best]
        wk = np.bincount(inv, weights=diff) / np.bincount(inv)
        se = float(wk.std(ddof=1) / np.sqrt(len(keys)))
        gap = means[name] - means[best]
        detail[name] = {"val_loss": means[name], "gap_to_best": gap, "se": se, "within_one_se": bool(gap <= se)}
        if chosen is None and gap <= se:
            chosen = name
    return chosen or best, detail
