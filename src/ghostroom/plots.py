"""README charts, all drawn from the numbers in metrics.json and the replay tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, BLUE, ORANGE, GREEN, GREY, RED = "#222831", "#2b6cb0", "#dd6b20", "#2f855a", "#a0aec0", "#c53030"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "figure.dpi": 130})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def leakage(res, path):
    a = res["leakage_audit"]
    labels = {"booking_time_only": "booking-time only", "plus_booking_changes": "+ booking changes",
              "plus_special_requests": "+ special requests", "plus_parking_request": "+ parking request",
              "plus_room_reassigned": "+ room reassigned", "plus_status_text": "+ final status text",
              "plus_everything_after_booking": "+ all of the above", "without_price": "no price",
              "without_deposit_type": "no deposit type", "without_agent_and_country": "no agent or country"}
    keys = list(labels)
    vals = [a[k]["auc"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cols = [BLUE if k == "booking_time_only" else RED if k.startswith("plus") else GREY for k in keys]
    ax.barh([labels[k] for k in keys][::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.03)
    ax.set_xlabel("validation AUC, same boosted model")
    ax.set_title("What each kind of information does to the score")
    _save(fig, path)


def model_comparison(res, path):
    names = list(res["models"])
    val = [res["models"][n]["val"]["log_loss"] for n in names]
    test = [res["models"][n]["test"]["log_loss"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, val, 0.4, label="validation", color=GREY)
    ax.bar(x + 0.2, test, 0.4, label="test", color=BLUE)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Validation picked one model, the test fold liked another")
    ax.legend(frameon=False)
    _save(fig, path)


def drift(df, res, path):
    q = df["arrival"].dt.to_period("Q").astype(str)
    a9 = df["agent"] == 9
    top = df.groupby(q)["is_canceled"].mean()
    ag = df[a9].groupby(q[a9])["is_canceled"].mean()
    rest = df[~a9].groupby(q[~a9])["is_canceled"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ag.index, ag.values, marker="o", color=ORANGE, label="agent 9 (largest online channel)")
    ax.plot(rest.index, rest.values, marker="o", color=BLUE, label="everyone else")
    ax.plot(top.index, top.values, color=GREY, ls="--", label="all bookings")
    for lab, start in (("validation", "2016Q3"), ("test", "2017Q1")):
        if start in ag.index:
            ax.axvline(list(ag.index).index(start) - 0.5, color=INK, lw=0.6)
            ax.text(list(ag.index).index(start) - 0.45, 0.97, lab, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylabel("share of bookings cancelled")
    ax.set_title("One channel's cancellation rate climbed from 28% to 46%")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def delayed(monthly, res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = {0: GREY, 1: "#4a5568", 2: GREEN, 3: ORANGE, 4: RED, 5: "#805ad5", 6: BLUE}
    for i, c in enumerate(monthly.columns):
        axes[0].plot(monthly.index, monthly[c], marker="o", ms=3, lw=1.4, label=c, color=colors[i % 7])
    axes[0].set_ylabel("log loss by booking month")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Test bookings scored on the day they were made")
    t = pd.DataFrame(res["delayed_labels"]["table"]).set_index("method")
    axes[1].barh(range(len(t)), t["log_loss"], color=[colors[i % 7] for i in range(len(t))])
    axes[1].set_yticks(range(len(t)), [m[:30] for m in t.index], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("pooled log loss")
    axes[1].set_title("Pooled")
    _save(fig, path)


def reliability(df, y, masks, p, path):
    yt = y[masks["test"]]
    order = np.argsort(p)
    xs, ys = [], []
    for idx in np.array_split(order, 10):
        xs.append(p[idx].mean())
        ys.append(yt[idx].mean())
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--")
    ax.plot(xs, ys, marker="o", color=BLUE)
    ax.set_xlabel("predicted cancellation probability")
    ax.set_ylabel("observed cancellation rate")
    ax.set_title("Shipped model on the test fold")
    _save(fig, path)


def overbooking(res, results, path):
    base = res["economics"]["walk_3x"]["table"]
    names = list(base)
    profit = [base[n]["profit"] / 1e6 for n in names]
    walked = [base[n]["walked"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pal = [GREY, "#e2b27a", ORANGE, BLUE, GREEN]
    axes[0].bar(range(len(names)), profit, color=pal)
    axes[0].set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names], fontsize=7)
    axes[0].set_ylim(min(profit) * 0.9, max(profit) * 1.02)
    axes[0].set_ylabel("six-month profit, millions")
    for i, v in enumerate(profit):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(range(len(names)), walked, color=pal)
    axes[1].set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names], fontsize=7)
    axes[1].set_ylabel("guests walked")
    fig.suptitle("Overbooking replay on the test fold (walk cost 3x one night)")
    _save(fig, path)


def sweep(res, path):
    s = pd.DataFrame(res["capacity_sweep"])
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(s["capacity_quantile"], (s["model"] - s["flat"]) / 1e3, marker="o", color=BLUE, label="model minus flat factor")
    ax.plot(s["capacity_quantile"], (s["model"] - s["none"]) / 1e3, marker="o", color=GREEN, label="model minus no overbooking")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("capacity as a quantile of training-period daily bookings")
    ax.set_ylabel("profit gain, thousands")
    ax.legend(frameon=False)
    ax.set_title("Gain by how tight capacity is")
    _save(fig, path)


def make_all(df, y, masks, res, monthly, results_base, assets: Path):
    assets = Path(assets)
    leakage(res, assets / "leakage_audit.png")
    model_comparison(res, assets / "model_comparison.png")
    drift(df, res, assets / "channel_drift.png")
    delayed(monthly, res, assets / "delayed_labels.png")
    chosen = res["shipped"]
    import joblib
    mdl = joblib.load(assets.parent / "outputs" / "ghostroom_model.joblib")["model"]
    from . import features as F
    from . import models as M
    p = M.proba(mdl, F.build_frame(df)[masks["test"]])
    reliability(df, y, masks, p, assets / "reliability.png")
    overbooking(res, results_base, assets / "overbooking.png")
    sweep(res, assets / "capacity_sweep.png")
    del chosen
