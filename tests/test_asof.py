import numpy as np
import pandas as pd

from ghostroom import asof
from ghostroom import features as F


def tiny() -> pd.DataFrame:
    # booked/outcome dates around a retrain date of 2016-06-01
    return pd.DataFrame({
        "booked": pd.to_datetime(["2016-05-01", "2016-05-10", "2016-05-20", "2016-06-05", "2016-04-01"]),
        "outcome_date": pd.to_datetime(["2016-05-15", "2016-07-01", "2016-05-25", "2016-06-10", "2016-08-01"]),
        "is_canceled": [1, 0, 1, 1, 1],
    })


def test_final_uses_every_earlier_booking_with_its_eventual_label():
    use, y = asof.training_table(tiny(), pd.Timestamp("2016-06-01"), "final")
    assert use.tolist() == [True, True, True, False, True]
    assert y.tolist() == [1, 0, 1, 1, 1]


def test_matured_drops_rows_not_resolved_by_the_retrain_date():
    use, _ = asof.training_table(tiny(), pd.Timestamp("2016-06-01"), "matured")
    assert use.tolist() == [True, False, True, False, False]


def test_pending_counts_unresolved_rows_as_stays():
    use, y = asof.training_table(tiny(), pd.Timestamp("2016-06-01"), "pending0")
    assert use.tolist() == [True, True, True, False, True]
    assert y.tolist()[:3] == [1, 0, 1] and y[4] == 0


def test_outcome_on_the_retrain_day_counts_as_known():
    d = tiny()
    d.loc[0, "outcome_date"] = pd.Timestamp("2016-06-01")
    use, _ = asof.training_table(d, pd.Timestamp("2016-06-01"), "matured")
    assert use[0]


def test_unknown_regime_raises():
    try:
        asof.training_table(tiny(), pd.Timestamp("2016-06-01"), "nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_month_starts_cover_the_range():
    out = asof.month_starts(pd.Timestamp("2016-03-17"), pd.Timestamp("2016-06-02"))
    assert [str(t.date()) for t in out] == ["2016-03-01", "2016-04-01", "2016-05-01", "2016-06-01"]


def test_batch_replay_only_scores_requested_rows(frame):
    X = F.build_frame(frame)
    ev = (frame["split"] == "test").to_numpy() & (frame["booked"] >= pd.Timestamp("2016-03-01")).to_numpy()
    pred = asof.replay_batch(frame, X, ev, "logistic", "matured")
    assert np.isnan(pred[~ev]).all()
    scored = ~np.isnan(pred)
    assert scored.sum() > 0.5 * ev.sum()
    assert ((pred[scored] >= 0) & (pred[scored] <= 1)).all()


def test_stream_replay_produces_probabilities(frame):
    X = F.build_frame(frame).iloc[:1500]
    sub = frame.iloc[:1500].reset_index(drop=True)
    ev = np.zeros(len(sub), bool)
    ev[-300:] = True
    pred = asof.replay_stream(sub, X.reset_index(drop=True), ev)
    assert np.isnan(pred[~ev]).all()
    assert ((pred[ev] >= 0) & (pred[ev] <= 1)).all()


def test_adwin_flags_a_step_change():
    n = 3000
    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-01", periods=n, freq="h")
    y = np.r_[rng.random(n // 2) < 0.2, rng.random(n // 2) < 0.7].astype(int)
    d = pd.DataFrame({"outcome_date": dates, "is_canceled": y})
    alarms = asof.drift_alarms(d, np.ones(n, bool))
    assert alarms and pd.Timestamp(alarms[0]) >= pd.Timestamp("2016-03-01")


def test_adwin_quiet_on_a_flat_stream():
    n = 3000
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"outcome_date": pd.date_range("2016-01-01", periods=n, freq="h"), "is_canceled": (rng.random(n) < 0.3).astype(int)})
    assert asof.drift_alarms(d, np.ones(n, bool)) == []
