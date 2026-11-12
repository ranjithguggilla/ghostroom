"""Retraining under delayed labels.

A booking's label is only known on its outcome date: the day it is cancelled, or the day the guest leaves. A model
retrained on the first of a month can therefore only learn from bookings that were made earlier *and* have resolved.
This module replays that situation with real dates, and compares three ways of building the training table:

* final      every earlier booking with its eventual label (what a careless notebook does; it peeks at the future)
* matured    only earlier bookings whose outcome was already known on the retrain date
* pending0   earlier bookings, with unresolved ones counted as "did not cancel"

It also replays a streaming logistic model from `river` that learns each label on the day it arrives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E
from . import features as F
from . import models as M

REGIMES = ("final", "matured", "pending0")


def training_table(df: pd.DataFrame, t: pd.Timestamp, regime: str) -> tuple[np.ndarray, np.ndarray]:
    """Row mask and labels for a retrain on date t."""
    earlier = (df["booked"] < t).to_numpy()
    known = (df["outcome_date"] <= t).to_numpy()
    y = df["is_canceled"].to_numpy()
    if regime == "final":
        return earlier, y
    if regime == "matured":
        return earlier & known, y
    if regime == "pending0":
        return earlier, np.where(known, y, 0)
    raise ValueError(regime)


def month_starts(first: pd.Timestamp, last: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(first.replace(day=1), last, freq="MS"))


def replay_batch(df: pd.DataFrame, X: pd.DataFrame, eval_mask: np.ndarray, learner: str, regime: str) -> np.ndarray:
    """Predictions for every evaluation row from a model refit on the first of its booking month."""
    pred = np.full(len(df), np.nan)
    booked_month = df["booked"].dt.to_period("M").dt.to_timestamp()
    for t in month_starts(df.loc[eval_mask, "booked"].min(), df.loc[eval_mask, "booked"].max()):
        rows = eval_mask & (booked_month == t).to_numpy()
        if not rows.any():
            continue
        use, y = training_table(df, t, regime)
        if use.sum() < 2000 or len(np.unique(y[use])) < 2:
            continue
        model = M.make(learner).fit(X[use], y[use])
        pred[rows] = M.proba(model, X[rows])
    return pred


def river_model():
    from river import compose, linear_model, optim, preprocessing

    num = compose.Select(*F.NUMERIC) | preprocessing.StandardScaler()
    cat = compose.Select(*F.CATEGORICAL) | preprocessing.OneHotEncoder()
    return (num + cat) | linear_model.LogisticRegression(optimizer=optim.SGD(0.0005))


def replay_stream(df: pd.DataFrame, X: pd.DataFrame, eval_mask: np.ndarray) -> np.ndarray:
    """Streaming replay. Labels are learned on their outcome date; evaluation rows are scored on their booking date.

    On a shared date the labels are applied first, matching the batch rule `outcome_date <= t`.
    """
    cols = F.NUMERIC + F.CATEGORICAL
    records = X[cols].to_dict("records")
    for r in records:
        for c in F.NUMERIC:
            r[c] = float(r[c])
    y = df["is_canceled"].to_numpy()
    learn = pd.DataFrame({"t": df["outcome_date"].to_numpy(), "kind": 0, "i": np.arange(len(df))})
    score = pd.DataFrame({"t": df["booked"].to_numpy()[eval_mask], "kind": 1, "i": np.flatnonzero(eval_mask)})
    events = pd.concat([learn, score]).sort_values(["t", "kind", "i"], kind="stable")
    model = river_model()
    pred = np.full(len(df), np.nan)
    for kind, i in zip(events["kind"].to_numpy(), events["i"].to_numpy()):
        if kind == 0:
            model.learn_one(records[i], int(y[i]))
        else:
            pred[i] = model.predict_proba_one(records[i]).get(1, 0.5)
    return pred


def drift_alarms(df: pd.DataFrame, mask: np.ndarray, delta: float = 1e-6) -> list[str]:
    """Dates on which ADWIN flags a change in the cancellation rate of the labels as they arrive."""
    from river import drift

    sub = df.loc[mask].sort_values("outcome_date")
    det = drift.ADWIN(delta=delta)
    alarms = []
    for date, label in zip(sub["outcome_date"], sub["is_canceled"]):
        det.update(float(label))
        if det.drift_detected:
            alarms.append(str(date.date()))
    return alarms


def compare(df: pd.DataFrame, y: np.ndarray, eval_mask: np.ndarray, preds: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled metrics per method and a month-by-month table of log loss."""
    rows, monthly = [], {}
    month = df["booked"].dt.to_period("M").astype(str).to_numpy()
    wk = df["arrival"].dt.to_period("W").astype(str).to_numpy()
    for name, p in preds.items():
        ok = eval_mask & ~np.isnan(p)
        m = E.metrics(y[ok], p[ok])
        _, lo, hi = E.cluster_ci(E.row_log_loss(y[ok], p[ok]), wk[ok])
        m.update(method=name, loss_lo=lo, loss_hi=hi)
        rows.append(m)
        ll = pd.Series(E.row_log_loss(y[ok], p[ok]), index=month[ok]).groupby(level=0).mean()
        monthly[name] = ll
    return pd.DataFrame(rows).set_index("method"), pd.DataFrame(monthly)
