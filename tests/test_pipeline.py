import json

import joblib

from ghostroom import pipeline
from ghostroom.app import build
from ghostroom.cli import main


def test_fast_run_writes_metrics_and_model(tmp_path, frame):
    out = tmp_path / "outputs"
    res = pipeline.run(out_dir=out, assets_dir=tmp_path / "assets", df=frame, fast=True)
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["shipped"] in res["models"]
    assert set(res["models"]) >= {"channel_lookup", "logistic", "boosting_small"}
    assert res["shipped_test"]["auc"] > 0.6
    assert "calibrated" in res["final_test"]
    assert res["economics"]["walk_3x"]["table"]["perfect foresight"]["walked"] == 0
    assert res["leakage_audit"]["plus_everything_after_booking"]["auc"] >= res["leakage_audit"]["booking_time_only"]["auc"] - 0.02
    assert (out / "ghostroom_model.joblib").exists()
    assert "model" in joblib.load(out / "ghostroom_model.joblib")


def test_cli_show_and_score(tmp_path, frame, capsys):
    out = tmp_path / "o"
    pipeline.run(out_dir=out, assets_dir=tmp_path / "a", df=frame, fast=True)
    main(["show", "--out", str(out)])
    assert "shipped model" in capsys.readouterr().out
    main(["score", "--out", str(out), "lead_time=120", "deposit_type=Non Refund"])
    assert "probability" in capsys.readouterr().out


def test_calibration_choice_flags_a_bad_scale():
    import numpy as np
    rng = np.random.default_rng(0)
    p = rng.random(4000)
    y = (rng.random(4000) < p**2).astype(int)
    c = pipeline.calibration_choice(y[:2000], p[:2000], y[2000:], p[2000:])
    assert c["ship_isotonic"] and c["test_ece_isotonic"] < c["test_ece_raw"]


def test_gradio_app_builds(tmp_path, frame):
    out = tmp_path / "o"
    pipeline.run(out_dir=out, assets_dir=tmp_path / "a", df=frame, fast=True)
    assert build(out / "ghostroom_model.joblib") is not None


def test_retrain_uses_only_bookings_known_on_the_date(tmp_path, frame):
    info = pipeline.retrain(tmp_path, as_of="2016-06-01", df=frame, learner="logistic")
    known = (frame["booked"] < "2016-06-01") & (frame["outcome_date"] <= "2016-06-01")
    assert info["rows"] == int(known.sum())
    bundle = joblib.load(tmp_path / "ghostroom_model_asof.joblib")
    assert bundle["as_of"] == "2016-06-01"


def test_cli_retrain_runs(tmp_path, frame, monkeypatch, capsys):
    monkeypatch.setattr("ghostroom.data.load", lambda path=None: frame)
    main(["retrain", "--out", str(tmp_path), "--as-of", "2016-09-01"])
    assert "refit boosting_small" in capsys.readouterr().out
    main(["score", "--out", str(tmp_path), "--model", "ghostroom_model_asof.joblib", "lead_time=10"])
    assert "probability" in capsys.readouterr().out
