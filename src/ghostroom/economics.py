"""Overbooking replay.

For each hotel and arrival date the bookings arrive in the order they were made. A policy decides which to accept.
Guests who show up are served up to capacity; the rest are walked at a cost. Everything is replayed on real outcomes.

Assumptions (all adjustable): demand beyond what the hotel accepted would have existed and is simply lost; one
arrival-day cohort is treated on its own, so stays that overlap other nights are not modelled; a served guest pays
adr x nights; a walked guest pays nothing and costs `walk_mult` x one night's adr in relocation and goodwill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

ADR_CAP = 500.0


@dataclass
class Assumptions:
    capacity_quantile: float = 0.60  # capacity per hotel = this quantile of daily bookings in the training period
    walk_mult: float = 3.0


def capacities(train: pd.DataFrame, q: float) -> dict[str, int]:
    out = {}
    for hotel, g in train.groupby("hotel"):
        out[hotel] = int(round(g.groupby("arrival").size().quantile(q)))
    return out


def _day_profit(shown: np.ndarray, value: np.ndarray, adr: np.ndarray, cap: int, walk_mult: float) -> tuple[float, int]:
    """Serve the first `cap` guests who show; walk the rest."""
    idx = np.flatnonzero(shown)
    served, walked = idx[:cap], idx[cap:]
    return float(value[served].sum() - walk_mult * adr[walked].sum()), int(len(walked))


def simulate(df: pd.DataFrame, p: np.ndarray, cap: dict[str, int], policy: str, param: float, walk_mult: float) -> pd.DataFrame:
    """Per hotel-day outcome of one policy. Rows of df must already be restricted to the period being replayed."""
    d = df.assign(p=p, adr_c=df["adr"].clip(0, ADR_CAP), value=df["adr"].clip(0, ADR_CAP) * df["nights"].clip(lower=1))
    d = d.sort_values(["hotel", "arrival", "booked"], kind="stable")
    rows = []
    for (hotel, day), g in d.groupby(["hotel", "arrival"], sort=False):
        c = cap[hotel]
        shown_all = g["is_canceled"].to_numpy() == 0
        n = len(g)
        if policy == "none":
            take = min(n, c)
        elif policy == "flat":
            take = min(n, math.ceil(c * (1 + param)))
        elif policy == "model":
            exp_shows = np.cumsum(1 - g["p"].to_numpy())
            take = int(np.searchsorted(exp_shows, c * param, side="right"))
        elif policy == "oracle":
            cum = np.cumsum(shown_all)
            take = int(np.searchsorted(cum, c, side="right"))
        else:
            raise ValueError(policy)
        acc = slice(0, take)
        profit, walked = _day_profit(shown_all[acc], g["value"].to_numpy()[acc], g["adr_c"].to_numpy()[acc], c, walk_mult)
        rows.append((hotel, day, n, take, int(shown_all[acc].sum()), walked, profit))
    return pd.DataFrame(rows, columns=["hotel", "arrival", "demand", "accepted", "shows", "walked", "profit"])


def tune(df: pd.DataFrame, p: np.ndarray, cap: dict[str, int], walk_mult: float) -> dict[str, float]:
    """Pick the flat overbooking factor and the model cut-off with the best profit on a tuning period."""
    flat_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
    model_grid = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00]
    flat = {f: simulate(df, p, cap, "flat", f, walk_mult)["profit"].sum() for f in flat_grid}
    model = {k: simulate(df, p, cap, "model", k, walk_mult)["profit"].sum() for k in model_grid}
    return {"flat": max(flat, key=flat.get), "model": max(model, key=model.get)}


def evaluate_policies(df: pd.DataFrame, p: np.ndarray, cap: dict[str, int], knobs: dict[str, float], walk_mult: float) -> dict[str, pd.DataFrame]:
    return {
        "no overbooking": simulate(df, p, cap, "none", 0, walk_mult),
        "flat 10% rule of thumb": simulate(df, p, cap, "flat", 0.10, walk_mult),
        "flat overbooking": simulate(df, p, cap, "flat", knobs["flat"], walk_mult),
        "model overbooking": simulate(df, p, cap, "model", knobs["model"], walk_mult),
        "perfect foresight": simulate(df, p, cap, "oracle", 0, walk_mult),
    }


def summarise(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        rows[name] = {"profit": float(r["profit"].sum()), "walked": int(r["walked"].sum()),
                      "accepted": int(r["accepted"].sum()), "shows": int(r["shows"].sum())}
    return pd.DataFrame(rows).T


def paired_gain_ci(a: pd.DataFrame, b: pd.DataFrame, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Profit of policy a minus policy b, with a 95% interval from resampling whole arrival weeks."""
    m = a.merge(b, on=["hotel", "arrival"], suffixes=("_a", "_b"))
    m["gain"] = m["profit_a"] - m["profit_b"]
    wk = m["arrival"].dt.to_period("W").astype(str)
    per_week = m.groupby(wk)["gain"].sum().to_numpy()
    rng = np.random.default_rng(seed)
    draws = [per_week[rng.integers(0, len(per_week), len(per_week))].sum() for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(per_week.sum()), float(lo), float(hi)
