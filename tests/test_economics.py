import numpy as np
import pandas as pd
import pytest

from ghostroom import economics as Ec


def day(cancel_flags, adr=100.0, nights=2):
    n = len(cancel_flags)
    return pd.DataFrame({
        "hotel": "H", "arrival": pd.Timestamp("2017-04-01"), "booked": pd.date_range("2017-01-01", periods=n),
        "is_canceled": cancel_flags, "adr": adr, "nights": nights,
    })


def test_capacity_is_a_training_quantile():
    tr = pd.DataFrame({"hotel": "H", "arrival": np.repeat(pd.date_range("2016-01-01", periods=10), np.arange(1, 11))})
    assert Ec.capacities(tr, 0.5)["H"] == 6 or Ec.capacities(tr, 0.5)["H"] == 5


def test_no_overbooking_accepts_capacity_and_never_walks():
    d = day([0] * 10)
    r = Ec.simulate(d, np.zeros(10), {"H": 6}, "none", 0, 3.0)
    assert r["accepted"].iloc[0] == 6 and r["walked"].iloc[0] == 0
    assert r["profit"].iloc[0] == pytest.approx(6 * 200)


def test_flat_overbooking_walks_when_everyone_shows():
    d = day([0] * 10)
    r = Ec.simulate(d, np.zeros(10), {"H": 5}, "flat", 0.4, 3.0)
    assert r["accepted"].iloc[0] == 7
    assert r["walked"].iloc[0] == 2
    assert r["profit"].iloc[0] == pytest.approx(5 * 200 - 2 * 3.0 * 100)


def test_flat_overbooking_gains_when_half_cancel():
    d = day([1, 0] * 6)
    none = Ec.simulate(d, np.zeros(12), {"H": 4}, "none", 0, 3.0)["profit"].iloc[0]
    flat = Ec.simulate(d, np.zeros(12), {"H": 4}, "flat", 1.0, 3.0)["profit"].iloc[0]
    assert flat > none


def test_oracle_never_walks_and_fills_the_house():
    d = day([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    r = Ec.simulate(d, np.zeros(10), {"H": 4}, "oracle", 0, 3.0)
    assert r["walked"].iloc[0] == 0 and r["shows"].iloc[0] == 4


def test_model_policy_accepts_more_when_cancellation_is_likely():
    d = day([0] * 12)
    low = Ec.simulate(d, np.full(12, 0.05), {"H": 6}, "model", 1.0, 3.0)["accepted"].iloc[0]
    high = Ec.simulate(d, np.full(12, 0.5), {"H": 6}, "model", 1.0, 3.0)["accepted"].iloc[0]
    assert high > low


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        Ec.simulate(day([0]), np.zeros(1), {"H": 1}, "bogus", 0, 3.0)


def test_adr_is_capped():
    d = day([0, 0], adr=9000.0, nights=1)
    r = Ec.simulate(d, np.zeros(2), {"H": 2}, "none", 0, 3.0)
    assert r["profit"].iloc[0] == pytest.approx(2 * Ec.ADR_CAP)


def test_paired_gain_is_zero_for_identical_policies():
    d = day([0, 1, 0, 1] * 3)
    r = Ec.simulate(d, np.zeros(12), {"H": 5}, "none", 0, 3.0)
    gain, lo, hi = Ec.paired_gain_ci(r, r, n=50)
    assert gain == 0 and lo == 0 and hi == 0


def test_tune_returns_grid_values(frame):
    sub = frame[frame["split"] == "val"].head(800)
    cap = {"City Hotel": 4, "Resort Hotel": 3}
    knobs = Ec.tune(sub, np.full(len(sub), 0.35), cap, 3.0)
    assert set(knobs) == {"flat", "model"}
    assert knobs["flat"] >= 0
