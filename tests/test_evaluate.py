import numpy as np

from ghostroom import evaluate as E


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.2, 0.8], 500)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(400)], np.r_[np.ones(400), np.zeros(100)]])
    assert E.ece(y, p, bins=2) < 1e-9


def test_ece_large_for_overconfident_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    assert E.ece(y, np.where(y == 1, 0.99, 0.99)) > 0.4


def test_metrics_keys_and_ranges():
    y = np.array([0, 1, 0, 1, 1, 0])
    m = E.metrics(y, np.array([0.1, 0.9, 0.2, 0.7, 0.6, 0.3]))
    assert m["auc"] == 1.0
    assert 0 < m["log_loss"] < 1
    assert m["n"] == 6


def test_cluster_ci_brackets_the_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(2, 1, 600)
    g = np.repeat(np.arange(60), 10)
    est, lo, hi = E.cluster_ci(v, g)
    assert lo < est < hi
    assert hi - lo < 0.5


def test_auc_ci_brackets_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 800)
    p = np.clip(y * 0.3 + rng.random(800) * 0.7, 0, 1)
    est, lo, hi = E.auc_ci(y, p, np.repeat(np.arange(40), 20), n=100)
    assert lo <= est <= hi


def test_one_se_prefers_simpler_when_tied():
    rng = np.random.default_rng(3)
    base = rng.random(1000)
    groups = np.repeat(np.arange(50), 20)
    losses = {"simple": base + 0.0005 + rng.normal(0, 0.05, 1000), "complex": base}
    pick, detail = E.one_se_pick(["simple", "complex"], losses, groups)
    assert pick == "simple"
    assert detail["simple"]["within_one_se"]


def test_one_se_takes_complex_when_clearly_better():
    rng = np.random.default_rng(4)
    base = rng.random(1000)
    groups = np.repeat(np.arange(50), 20)
    losses = {"simple": base + 0.3 + rng.normal(0, 0.01, 1000), "complex": base}
    assert E.one_se_pick(["simple", "complex"], losses, groups)[0] == "complex"
