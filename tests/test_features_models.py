import numpy as np
import pytest

from ghostroom import features as F
from ghostroom import models as M


@pytest.fixture(scope="module")
def xy(frame):
    return F.build_frame(frame), frame["is_canceled"].to_numpy(), F.split_masks(frame)


def test_frame_has_every_declared_column(xy):
    X, _, _ = xy
    assert set(F.BOOKING_TIME) <= set(X.columns)
    for cols in F.AFTER_BOOKING.values():
        assert set(cols) <= set(X.columns)


def test_party_size_at_least_one_and_no_missing(xy):
    X, _, _ = xy
    assert (X["party_size"] >= 1).all()
    assert not X[F.NUMERIC].isna().any().any()


def test_booking_time_block_excludes_after_booking_columns():
    after = {c for cols in F.AFTER_BOOKING.values() for c in cols}
    assert after.isdisjoint(F.BOOKING_TIME)


def test_lookup_returns_prior_for_unseen_key(xy):
    X, y, m = xy
    mod = M.ChannelLookup().fit(X[m["train"]], y[m["train"]])
    unseen = X[m["val"]].iloc[:1].copy()
    unseen["market_segment"] = "never seen"
    assert M.proba(mod, unseen)[0] == pytest.approx(y[m["train"]].mean())


def test_lookup_smoothing_pulls_small_groups_to_prior(xy):
    X, y, m = xy
    strong = M.ChannelLookup(strength=1e6).fit(X[m["train"]], y[m["train"]])
    assert np.allclose(M.proba(strong, X[m["val"]]), y[m["train"]].mean(), atol=1e-3)


@pytest.mark.parametrize("name", M.ORDER)
def test_every_candidate_fits_and_beats_coin_flip(name, xy):
    X, y, m = xy
    mod = M.make(name).fit(X[m["train"]], y[m["train"]])
    p = M.proba(mod, X[m["val"]])
    assert p.shape == (m["val"].sum(),)
    assert ((p >= 0) & (p <= 1)).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[m["val"]], p) > 0.6


def test_rule_trees_describe_is_text(xy):
    X, y, m = xy
    mod = M.RuleTrees(max_rules=6).fit(X[m["train"]], y[m["train"]])
    assert "FIGS" in mod.describe()


def test_calibrated_wrapper_maps_probabilities_and_keeps_steps(xy):
    from sklearn.isotonic import IsotonicRegression
    X, y, m = xy
    base = M.make("logistic").fit(X[m["train"]], y[m["train"]])
    p = M.proba(base, X[m["val"]])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y[m["val"]])
    cal = M.Calibrated(base, iso)
    out = M.proba(cal, X[m["val"]])
    assert out.shape == p.shape and ((out > 0) & (out < 1)).all()
    assert list(cal.named_steps) == list(base.named_steps)
    assert np.allclose(cal.decision_function(X[m["val"]]), base.decision_function(X[m["val"]]))
