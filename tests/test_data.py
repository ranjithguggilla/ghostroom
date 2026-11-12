import hashlib

import pandas as pd
import pytest

from ghostroom import data as D


def test_booking_date_is_arrival_minus_lead(frame):
    assert ((frame["arrival"] - frame["booked"]).dt.days == frame["lead_time"]).all()


def test_nights_and_dates_types(frame):
    assert (frame["nights"] == frame["stays_in_weekend_nights"] + frame["stays_in_week_nights"]).all()
    assert pd.api.types.is_datetime64_any_dtype(frame["outcome_date"])


def test_splits_follow_arrival_date(frame):
    assert frame.loc[frame.split == "train", "arrival"].max() < D.VAL_START
    assert frame.loc[frame.split == "val", "arrival"].min() >= D.VAL_START
    assert frame.loc[frame.split == "val", "arrival"].max() < D.TEST_START
    assert frame.loc[frame.split == "test", "arrival"].min() >= D.TEST_START


def test_checks_report_clean_synthetic_dates(frame):
    c = D.data_checks(frame)
    assert c["cancel_before_booking"] == 0
    assert c["cancel_after_arrival"] == 0
    assert c["rows"] == len(frame)
    assert sum(c["split_rows"].values()) == len(frame)


def test_fetch_uses_override(tmp_path, monkeypatch):
    p = tmp_path / "x.csv"
    p.write_text("a\n1\n")
    monkeypatch.setenv("GHOSTROOM_DATA", str(p))
    assert D.fetch() == p


def test_fetch_rejects_wrong_checksum(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOSTROOM_DATA", raising=False)
    p = tmp_path / "hotels.csv"
    p.write_text("not the real file")
    assert hashlib.sha256(p.read_bytes()).hexdigest() != D.SHA256
    with pytest.raises(RuntimeError):
        D.fetch(p)
