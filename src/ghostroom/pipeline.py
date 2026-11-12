"""End-to-end run: model comparison, leakage audit, delayed-label retraining, overbooking replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from . import asof, scoring
from . import data as D
from . import economics as Ec
from . import evaluate as E
from . import features as F
from . import models as M

ASOF_START = pd.Timestamp("2016-03-01")


def _fit(name, X, y, mask):
    return M.make(name).fit(X[mask], y[mask])


def compare_models(df, X, y, masks, wk):
    """Fit on train, score validation, pick by the one-SE rule, then refit on train+val for the test fold."""
    val_p, val_loss, rows = {}, {}, {}
    train_fits = {}
    for name in M.ORDER:
        train_fits[name] = _fit(name, X, y, masks["train"])
        val_p[name] = M.proba(train_fits[name], X[masks["val"]])
        val_loss[name] = E.row_log_loss(y[masks["val"]], val_p[name])
    chosen, detail = E.one_se_pick(M.ORDER, val_loss, wk[masks["val"]])
    trval = masks["train"] | masks["val"]
    test_p = {}
    for name in M.ORDER:
        refit = _fit(name, X, y, trval)
        test_p[name] = M.proba(refit, X[masks["test"]])
        rows[name] = {
            "val": E.metrics(y[masks["val"]], val_p[name]),
            "test": E.metrics(y[masks["test"]], test_p[name]),
            "selection": detail[name],
        }
    return chosen, rows, train_fits, val_p, test_p


def leak_audit(df, X, y, masks):
    """Add one block of after-booking information at a time and watch the validation score move."""
    def score(extra_numeric, drop_cat=(), drop_num=()):
        num = [c for c in F.NUMERIC if c not in drop_num] + list(extra_numeric)
        cat = [c for c in F.CATEGORICAL if c not in drop_cat]
        m = M.boosting("small", numeric=num, categorical=cat).fit(X[masks["train"]], y[masks["train"]])
        return E.metrics(y[masks["val"]], M.proba(m, X[masks["val"]]))

    out = {"booking_time_only": score([])}
    for block, cols in F.AFTER_BOOKING.items():
        out[f"plus_{block}"] = score(cols)
    out["plus_everything_after_booking"] = score([c for cols in F.AFTER_BOOKING.values() for c in cols])
    out["without_price"] = score([], drop_num=("adr", "adr_per_person"))
    out["without_deposit_type"] = score([], drop_cat=("deposit_type",))
    out["without_agent_and_country"] = score([], drop_cat=("agent_id", "country"))
    return out


def validation_design(df, X, y, masks, forward_test_auc):
    """The same learner under three ways of holding data out."""
    idx = np.arange(len(df))
    rng_tr, rng_te = train_test_split(idx, test_size=0.25, random_state=0)

    def run(tr, te):
        m = M.boosting("small").fit(X.iloc[tr], y[tr])
        return E.metrics(y[te], M.proba(m, X.iloc[te]))

    out = {"random_rows": run(rng_tr, rng_te)}
    raw = df.drop(columns=["split", "arrival", "booked", "outcome_date", "nights"])
    first = ~raw.duplicated().to_numpy()
    keep = idx[first]
    a, b = train_test_split(keep, test_size=0.25, random_state=0)
    out["random_rows_deduplicated"] = run(a, b)
    out["forward_in_time_test_auc"] = forward_test_auc
    out["duplicate_share"] = float(1 - first.mean())
    return out


def fit_isotonic(y_val, p_val):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)


def calibration_choice(y_val, p_val, y_test, p_test, order=None):
    """Isotonic is fitted on the earlier half of validation and judged on the later half."""
    if order is not None:
        y_val, p_val = y_val[order], p_val[order]
    n = len(y_val) // 2
    iso_half = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val[:n], y_val[:n])
    raw_half, iso_half_ece = E.ece(y_val[n:], p_val[n:]), E.ece(y_val[n:], iso_half.predict(p_val[n:]))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)
    return {
        "val_second_half_ece_raw": raw_half,
        "val_second_half_ece_isotonic": iso_half_ece,
        "ship_isotonic": bool(iso_half_ece < raw_half - 0.005),
        "test_ece_raw": E.ece(y_test, p_test),
        "test_ece_isotonic": E.ece(y_test, iso.predict(p_test)),
    }


def segment_audit(df, y, mask, p):
    d = df.loc[mask].assign(p=p, y=y[mask])
    out = {}
    for col in ("hotel", "market_segment", "deposit_type"):
        g = d.groupby(col).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
        out[col] = g.round(4).reset_index().to_dict("records")
    top = d["agent"].value_counts().head(5).index
    g = d[d["agent"].isin(top)].groupby("agent").agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    out["top_agents"] = g.round(4).reset_index().to_dict("records")
    return out


def agent_drift(df, y, masks, p_static_test):
    a9 = df["agent"] == 9
    q = df["arrival"].dt.to_period("Q").astype(str)
    obs = df[a9].groupby(q[a9])["is_canceled"].agg(["mean", "size"])
    return {"agent9_cancel_rate_by_quarter": {k: round(float(v), 4) for k, v in obs["mean"].items()},
            "agent9_bookings_by_quarter": {k: int(v) for k, v in obs["size"].items()},
            "agent9_share_of_test": float(a9[masks["test"]].mean()),
            "agent9_test_observed": float(y[masks["test"] & a9.to_numpy()].mean()),
            "agent9_test_predicted_static": float(p_static_test[a9[masks["test"]].to_numpy()].mean())}


def delayed_label_study(df, X, y, masks, shipped_train_p, fast=False, boosting_train_p=None):
    ev = masks["test"] & (df["booked"] >= ASOF_START).to_numpy()
    preds = {}
    static = np.full(len(df), np.nan)
    static[masks["test"]] = shipped_train_p
    preds["static, trained once on train"] = static
    if boosting_train_p is not None:
        static_b = np.full(len(df), np.nan)
        static_b[masks["test"]] = boosting_train_p
        preds["static boosting, trained once on train"] = static_b
    plan = [("boosting_small", "final", "monthly refit, final labels (peeks)"),
            ("boosting_small", "matured", "monthly refit, matured labels"),
            ("boosting_small", "pending0", "monthly refit, pending as no-cancel"),
            ("logistic", "matured", "logistic monthly refit, matured labels")]
    if fast:
        plan = plan[1:2]
    for learner, regime, label in plan:
        preds[label] = asof.replay_batch(df, X, ev, learner, regime)
    if not fast:
        preds["streaming logistic (river)"] = asof.replay_stream(df, X, ev)
    ok = ev.copy()
    for p in preds.values():
        ok &= ~np.isnan(p)
    table, monthly = asof.compare(df, y, ok, preds)
    alarms = {}
    if not fast:
        a9 = (df["agent"] == 9).to_numpy()
        alarms = {"agent_9": asof.drift_alarms(df, a9), "all_other_bookings": asof.drift_alarms(df, ~a9)}
    return table, monthly, alarms, preds, ok


def economics_study(df, y, masks, cap_q, p_val, p_test, p_alt=None):
    train = df[masks["train"]]
    result = {}
    for walk in (1.0, 3.0, 5.0):
        cap = Ec.capacities(train, cap_q)
        knobs = Ec.tune(df[masks["val"]], p_val, cap, walk)
        res = Ec.evaluate_policies(df[masks["test"]], p_test, cap, knobs, walk)
        summ = Ec.summarise(res)
        entry = {"capacity": cap, "knobs": knobs, "table": summ.round(0).to_dict("index")}
        entry["model_vs_flat"] = Ec.paired_gain_ci(res["model overbooking"], res["flat overbooking"])
        entry["model_vs_none"] = Ec.paired_gain_ci(res["model overbooking"], res["no overbooking"])
        entry["flat_vs_none"] = Ec.paired_gain_ci(res["flat overbooking"], res["no overbooking"])
        entry["share_of_foresight_gap_closed"] = float(
            (summ.loc["model overbooking", "profit"] - summ.loc["flat overbooking", "profit"])
            / (summ.loc["perfect foresight", "profit"] - summ.loc["flat overbooking", "profit"]))
        if p_alt is not None:
            alt = Ec.evaluate_policies(df[masks["test"]], p_alt, cap, knobs, walk)
            entry["alt_model_profit"] = float(alt["model overbooking"]["profit"].sum())
        result[f"walk_{walk:g}x"] = entry
        if walk == 3.0:
            result["_results"] = res
    return result


def capacity_sweep(df, masks, p_val, p_test):
    train = df[masks["train"]]
    rows = []
    for q in (0.4, 0.5, 0.6, 0.7, 0.8):
        cap = Ec.capacities(train, q)
        knobs = Ec.tune(df[masks["val"]], p_val, cap, 3.0)
        res = Ec.evaluate_policies(df[masks["test"]], p_test, cap, knobs, 3.0)
        s = Ec.summarise(res)["profit"]
        rows.append({"capacity_quantile": q, "city_rooms": cap["City Hotel"], "resort_rooms": cap["Resort Hotel"],
                     "flat_knob": knobs["flat"], "model_knob": knobs["model"],
                     "none": s["no overbooking"], "flat": s["flat overbooking"], "model": s["model overbooking"],
                     "foresight": s["perfect foresight"]})
    return rows


def run(out_dir="outputs", assets_dir="assets", df: pd.DataFrame | None = None, fast: bool = False) -> dict:
    t0 = time.time()
    out, assets = Path(out_dir), Path(assets_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    df = D.load() if df is None else df
    X, y, masks = F.build_frame(df), df["is_canceled"].to_numpy(), F.split_masks(df)
    wk = df["arrival"].dt.to_period("W").astype(str).to_numpy()
    res: dict = {"data": D.data_checks(df)}

    chosen, rows, train_fits, val_p, test_p = compare_models(df, X, y, masks, wk)
    res["models"], res["shipped"] = rows, chosen
    auc, lo, hi = E.auc_ci(y[masks["test"]], test_p[chosen], wk[masks["test"]], n=100 if fast else 300)
    res["shipped_test"] = {**rows[chosen]["test"], "auc_ci": [lo, hi]}
    ll, llo, lhi = E.cluster_ci(E.row_log_loss(y[masks["test"]], test_p[chosen]), wk[masks["test"]])
    res["shipped_test"]["log_loss_ci"] = [llo, lhi]
    best_test = min(rows, key=lambda k: rows[k]["test"]["log_loss"])
    res["best_on_test_in_hindsight"] = best_test

    # train-only predictions of the shipped learner on the test fold, used for calibration and as the static baseline
    p_test_train_only = M.proba(train_fits[chosen], X[masks["test"]])
    res["calibration"] = calibration_choice(y[masks["val"]], val_p[chosen], y[masks["test"]], p_test_train_only,
                                            order=np.argsort(df.loc[masks["val"], "arrival"].to_numpy(), kind="stable"))
    res["leakage_audit"] = leak_audit(df, X, y, masks)
    res["validation_design"] = validation_design(df, X, y, masks, rows[chosen]["test"]["auc"])
    res["segments"] = segment_audit(df, y, masks["test"], test_p[chosen])
    res["drift"] = agent_drift(df, y, masks, p_test_train_only)

    table, monthly, alarms, preds, ok = delayed_label_study(df, X, y, masks, p_test_train_only, fast,
                                                            M.proba(train_fits["boosting_small"], X[masks["test"]]))
    res["delayed_labels"] = {"rows": int(ok.sum()), "table": table.round(5).reset_index().to_dict("records"),
                             "drift_alarms": alarms}
    matured_key = "monthly refit, matured labels"
    p_alt = np.full(masks["test"].sum(), np.nan)
    pm = preds[matured_key][masks["test"]]
    p_alt = np.where(np.isnan(pm), test_p[chosen], pm)

    cap_q = 0.6
    econ = economics_study(df, y, masks, cap_q, val_p[chosen], test_p[chosen], p_alt)
    results_base = econ.pop("_results")
    res["economics"] = econ
    res["capacity_sweep"] = [] if fast else capacity_sweep(df, masks, val_p[chosen], test_p[chosen])

    # final model: refit on train+val, then wrapped with the isotonic map when the validation check asked for it
    final_model = _fit(chosen, X, y, masks["train"] | masks["val"])
    if res["calibration"]["ship_isotonic"]:
        final_model = M.Calibrated(final_model, fit_isotonic(y[masks["val"]], val_p[chosen]))
    p_final = M.proba(final_model, X[masks["test"]])
    res["final_test"] = E.metrics(y[masks["test"]], p_final)
    res["final_test"]["calibrated"] = bool(res["calibration"]["ship_isotonic"])
    res["final_test"]["log_loss_ci"] = list(E.cluster_ci(E.row_log_loss(y[masks["test"]], p_final), wk[masks["test"]])[1:])
    res["final_test"]["auc_ci"] = list(E.auc_ci(y[masks["test"]], p_final, wk[masks["test"]], n=100 if fast else 300)[1:])

    # a few readable examples from the test fold
    ex_idx = np.argsort(p_final)
    pick = [ex_idx[-1], ex_idx[len(ex_idx) // 2], ex_idx[0]]
    Xt = X[masks["test"]]
    res["examples"] = [{"probability": float(p_final[i]), "observed": int(y[masks["test"]][i]),
                        **scoring.reasons(final_model, Xt.iloc[[i]])[0]} for i in pick]
    res["rule_text"] = train_fits["rule_trees"].describe()

    joblib.dump({"model": final_model, "name": chosen}, out / "ghostroom_model.joblib")
    res["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    if not fast:
        from . import plots
        plots.make_all(df, y, masks, res, monthly, results_base, assets)
    return res


def retrain(out_dir="outputs", as_of=None, df: pd.DataFrame | None = None, learner: str = "boosting_small") -> dict:
    """Refit on the bookings whose outcome was already known on `as_of` (default: the last outcome in the file)."""
    df = D.load() if df is None else df
    t = pd.Timestamp(as_of) if as_of else df["outcome_date"].max()
    use, y = asof.training_table(df, t, "matured")
    model = M.make(learner).fit(F.build_frame(df)[use], y[use])
    info = {"name": learner, "as_of": str(t.date()), "rows": int(use.sum()), "cancel_rate": float(y[use].mean())}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **info}, out / "ghostroom_model_asof.joblib")
    return info
