"""Load the hotel bookings table and derive the three dates that matter.

Every row has an arrival date. The booking date is arrival minus lead time, and the outcome
date is the cancellation date for cancelled stays or the check-out date for completed ones.
Those three dates decide what a model is allowed to know at any moment.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"
SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
DEFAULT_PATH = Path("data/hotels.csv")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}

# split boundaries by arrival date
VAL_START = pd.Timestamp("2016-09-01")
TEST_START = pd.Timestamp("2017-03-01")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(path: Path = DEFAULT_PATH) -> Path:
    """Download the csv once and check its checksum. GHOSTROOM_DATA can point at a local copy."""
    override = os.environ.get("GHOSTROOM_DATA")
    if override:
        return Path(override)
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    if _sha(path) != SHA256:
        raise RuntimeError(f"{path} does not match the expected checksum; delete it and run again")
    return path


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    month = df["arrival_date_month"].map(MONTHS)
    df["arrival"] = pd.to_datetime(
        {"year": df["arrival_date_year"], "month": month, "day": df["arrival_date_day_of_month"]})
    df["booked"] = df["arrival"] - pd.to_timedelta(df["lead_time"], unit="D")
    df["outcome_date"] = pd.to_datetime(df["reservation_status_date"])
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df


def load(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(fetch(path or DEFAULT_PATH))
    df = add_dates(df)
    df["split"] = "train"
    df.loc[df["arrival"] >= VAL_START, "split"] = "val"
    df.loc[df["arrival"] >= TEST_START, "split"] = "test"
    return df.reset_index(drop=True)


def data_checks(df: pd.DataFrame) -> dict:
    """Facts about the table that the rest of the pipeline relies on."""
    canc = df[df["is_canceled"] == 1]
    return {
        "rows": int(len(df)),
        "cancel_rate": float(df["is_canceled"].mean()),
        "exact_duplicate_rows": int(df.duplicated(subset=[c for c in df.columns if c not in ("split",)]).sum()),
        "cancel_before_booking": int((canc["outcome_date"] < canc["booked"]).sum()),
        "cancel_after_arrival": int((canc["outcome_date"] > canc["arrival"]).sum()),
        "arrival_min": str(df["arrival"].min().date()),
        "arrival_max": str(df["arrival"].max().date()),
        "earliest_booking": str(df["booked"].min().date()),
        "split_rows": {k: int(v) for k, v in df["split"].value_counts().items()},
    }
