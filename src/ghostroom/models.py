"""Model candidates, ordered from simplest to most flexible."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imodels import FIGSClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import features as F

ORDER = ["channel_lookup", "rule_trees", "logistic", "boosting_small", "boosting_large"]
LOOKUP_KEYS = ["hotel", "market_segment", "deposit_type"]


class ChannelLookup(BaseEstimator, ClassifierMixin):
    """Smoothed cancellation rate for hotel x channel x deposit type. The baseline a revenue manager already has."""

    def __init__(self, strength: float = 20.0):
        self.strength = strength

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.prior_ = float(y.mean())
        key = X[LOOKUP_KEYS].astype(str).agg("|".join, axis=1)
        g = pd.DataFrame({"k": key.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        self.rate_ = ((g["sum"] + self.strength * self.prior_) / (g["count"] + self.strength)).to_dict()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        key = X[LOOKUP_KEYS].astype(str).agg("|".join, axis=1)
        p = key.map(self.rate_).fillna(self.prior_).to_numpy(float)
        return np.column_stack([1 - p, p])


class RuleTrees(BaseEstimator, ClassifierMixin):
    """A short sum of small trees (FIGS from imodels) that a person can read top to bottom."""

    def __init__(self, max_rules: int = 12):
        self.max_rules = max_rules

    def fit(self, X: pd.DataFrame, y):
        self.pre_ = F.linear_preprocessor(F.NUMERIC, F.CATEGORICAL).fit(X)
        self.names_ = [n.split("__", 1)[1] for n in self.pre_.get_feature_names_out()]
        self.model_ = FIGSClassifier(max_rules=self.max_rules, random_state=0)
        self.model_.fit(self.pre_.transform(X), np.asarray(y), feature_names=self.names_)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(self.pre_.transform(X))

    def describe(self) -> str:
        return str(self.model_)


def boosting(size: str = "small", seed: int = 0, n_jobs: int = 4, numeric=None, categorical=None) -> Pipeline:
    params = {
        "small": dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=100),
        "large": dict(n_estimators=500, learning_rate=0.04, num_leaves=63, min_child_samples=40),
    }[size]
    clf = LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs, verbose=-1, **params)
    return Pipeline([("prep", F.tree_preprocessor(numeric or F.NUMERIC, categorical or F.CATEGORICAL)), ("model", clf)])


def logistic(c: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", F.linear_preprocessor(F.NUMERIC, F.CATEGORICAL)),
        ("model", LogisticRegression(C=c, max_iter=3000)),
    ])


def make(name: str, seed: int = 0):
    return {
        "channel_lookup": lambda: ChannelLookup(),
        "rule_trees": lambda: RuleTrees(),
        "logistic": lambda: logistic(),
        "boosting_small": lambda: boosting("small", seed),
        "boosting_large": lambda: boosting("large", seed),
    }[name]()


def proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class Calibrated:
    """A fitted model plus an isotonic map on its probabilities. Exposes the base pipeline's steps for explanations."""

    def __init__(self, base, iso):
        self.base, self.iso = base, iso

    @property
    def named_steps(self):
        return self.base.named_steps

    def predict_proba(self, X):
        p = np.clip(self.iso.predict(self.base.predict_proba(X)[:, 1]), 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p, p])

    def decision_function(self, X):
        return self.base.decision_function(X)
