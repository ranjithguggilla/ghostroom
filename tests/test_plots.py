import numpy as np
import pandas as pd

from ghostroom import plots


def fake_result():
    audit = {k: {"auc": 0.8 + i / 100, "log_loss": 0.5} for i, k in enumerate(
        ["booking_time_only", "plus_booking_changes", "plus_special_requests", "plus_parking_request", "plus_room_reassigned",
         "plus_status_text", "plus_everything_after_booking", "without_price", "without_deposit_type", "without_agent_and_country"])}
    models = {n: {"val": {"log_loss": 0.5}, "test": {"log_loss": 0.45}} for n in ("channel_lookup", "logistic", "boosting_small")}
    table = [{"method": f"m{i}", "log_loss": 0.4 + i / 20} for i in range(3)]
    names = ["no overbooking", "flat 10% rule of thumb", "flat overbooking", "model overbooking", "perfect foresight"]
    econ = {"walk_3x": {"table": {n: {"profit": 1e6 + i * 1e5, "walked": i * 10} for i, n in enumerate(names)}}}
    sweep = [{"capacity_quantile": q, "model": 2.0 + q, "flat": 1.9 + q, "none": 1.0} for q in (0.4, 0.6)]
    return {"leakage_audit": audit, "models": models, "delayed_labels": {"table": table}, "economics": econ, "capacity_sweep": sweep}


def test_every_chart_is_written(tmp_path, frame):
    res = fake_result()
    plots.leakage(res, tmp_path / "a.png")
    plots.model_comparison(res, tmp_path / "b.png")
    plots.drift(frame.assign(agent=9.0), res, tmp_path / "c.png")
    monthly = pd.DataFrame({f"m{i}": np.linspace(0.4, 0.5, 4) for i in range(3)}, index=["2016-03", "2016-04", "2016-05", "2016-06"])
    plots.delayed(monthly, res, tmp_path / "d.png")
    p = np.random.default_rng(0).random(100)
    y = (np.random.default_rng(1).random(100) < p).astype(int)
    plots.reliability(None, y, {"test": np.ones(100, bool)}, p, tmp_path / "e.png")
    plots.overbooking(res, None, tmp_path / "f.png")
    plots.sweep(res, tmp_path / "g.png")
    plots.sweep({"capacity_sweep": []}, tmp_path / "h.png")
    for name in "abcdefg":
        assert (tmp_path / f"{name}.png").stat().st_size > 2000
    assert not (tmp_path / "h.png").exists()
