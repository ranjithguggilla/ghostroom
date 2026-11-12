import pandas as pd
import pytest

from ghostroom import features as F
from ghostroom import models as M
from ghostroom import scoring as S


@pytest.fixture(scope="module")
def fitted(frame):
    X = F.build_frame(frame)
    y = frame["is_canceled"].to_numpy()
    return {"logistic": M.make("logistic").fit(X, y), "boosting": M.make("boosting_small").fit(X, y)}


def test_raw_frame_round_trips_through_features():
    X = F.build_frame(S.raw_frame({"lead_time": 30, "arrival_date": "2017-08-01"}))
    assert X.shape[0] == 1
    assert X.loc[0, "lead_time"] == 30
    assert X.loc[0, "arrival_month"] == 8


def test_agent_none_becomes_missing():
    X = F.build_frame(S.raw_frame({"agent": None}))
    assert X.loc[0, "has_agent"] == 0


@pytest.mark.parametrize("kind", ["logistic", "boosting"])
def test_score_booking_shape(fitted, kind):
    out = S.score_booking(fitted[kind], {"lead_time": 200, "deposit_type": "Non Refund"})
    assert 0 <= out["probability"] <= 1
    assert out["band"] in {"low", "medium", "high"}
    assert out["explanation"]


def test_non_refundable_long_lead_scores_above_direct_short_lead(fitted):
    risky = S.score_booking(fitted["logistic"], {"lead_time": 300, "deposit_type": "Non Refund", "market_segment": "Groups"})
    safe = S.score_booking(fitted["logistic"], {"lead_time": 2, "market_segment": "Direct"})
    assert risky["probability"] > safe["probability"]


def test_logistic_contributions_reproduce_the_logit(fitted, frame):
    X = F.build_frame(frame).head(25)
    contrib = S.contributions(fitted["logistic"], X)
    est = fitted["logistic"].named_steps["model"]
    logit = contrib.sum(axis=1) + est.intercept_[0]
    expected = fitted["logistic"].decision_function(X)
    assert (logit - expected).abs().max() < 1e-6


def test_boosting_contributions_reproduce_the_margin(fitted, frame):
    X = F.build_frame(frame).head(25)
    contrib = S.contributions(fitted["boosting"], X)
    prep, est = fitted["boosting"].named_steps["prep"], fitted["boosting"].named_steps["model"]
    raw = est.predict(prep.transform(X), pred_contrib=True)
    assert abs((contrib.sum(axis=1) + raw[:, -1] - raw.sum(axis=1)).abs().max()) < 1e-6


def test_reasons_split_into_up_and_down(fitted, frame):
    X = F.build_frame(frame).head(5)
    r = S.reasons(fitted["logistic"], X, top=4)
    assert len(r) == 5
    assert all(set(x) == {"up", "down"} and len(x["up"]) + len(x["down"]) == 4 for x in r)


def test_band_edges():
    assert S.band(0.1) == "low" and S.band(0.3) == "medium" and S.band(0.6) == "high"


def test_default_booking_scores_without_error(fitted):
    assert isinstance(S.score_booking(fitted["logistic"], {}), dict)
    assert isinstance(pd.Timestamp(S.DEFAULT_BOOKING["arrival_date"]), pd.Timestamp)
