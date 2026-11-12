"""Feature blocks, split by what a hotel actually knows when the booking is confirmed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = [
    "lead_time", "nights", "stays_in_weekend_nights", "adults", "children", "babies", "party_size",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "days_in_waiting_list",
    "adr", "adr_per_person", "arrival_week", "arrival_dow", "arrival_month", "has_agent", "has_company",
]
CATEGORICAL = [
    "hotel", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type",
    "deposit_type", "customer_type", "agent_id",
]
BOOKING_TIME = NUMERIC + CATEGORICAL

# information that accrues after the booking is made; used only in the leakage audit
AFTER_BOOKING = {
    "booking_changes": ["booking_changes"],
    "special_requests": ["total_of_special_requests"],
    "parking_request": ["required_car_parking_spaces"],
    "room_reassigned": ["room_reassigned"],
    "status_text": ["status_checked_out", "status_no_show"],
}


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only. Nothing here is fitted, so it cannot leak across splits."""
    out = pd.DataFrame(index=df.index)
    for c in ("lead_time", "nights", "stays_in_weekend_nights", "adults", "babies", "is_repeated_guest", "adr"):
        out[c] = df[c]
    # a few guests have dozens of earlier stays or months on a waiting list; cap them so one row cannot drive a linear fit
    out["previous_cancellations"] = df["previous_cancellations"].clip(upper=5)
    out["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].clip(upper=10)
    out["days_in_waiting_list"] = np.log1p(df["days_in_waiting_list"].clip(lower=0))
    out["children"] = df["children"].fillna(0)
    out["party_size"] = (out["adults"] + out["children"] + out["babies"]).clip(lower=1)
    out["adr"] = out["adr"].clip(lower=0, upper=500)
    out["adr_per_person"] = out["adr"] / out["party_size"]
    out["arrival_week"] = df["arrival_date_week_number"]
    out["arrival_dow"] = df["arrival"].dt.dayofweek
    out["arrival_month"] = df["arrival"].dt.month
    out["has_agent"] = df["agent"].notna().astype(int)
    out["has_company"] = df["company"].notna().astype(int)
    for c in ("hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"):
        out[c] = df[c].astype(str)
    out["country"] = df["country"].fillna("unknown").astype(str)
    out["agent_id"] = df["agent"].fillna(-1).astype(int).astype(str)
    # after-booking columns, kept for the audit
    out["booking_changes"] = df["booking_changes"]
    out["total_of_special_requests"] = df["total_of_special_requests"]
    out["required_car_parking_spaces"] = df["required_car_parking_spaces"]
    out["room_reassigned"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    out["status_checked_out"] = (df["reservation_status"] == "Check-Out").astype(int)
    out["status_no_show"] = (df["reservation_status"] == "No-Show").astype(int)
    return out


def linear_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False), categorical),
    ])


def tree_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=50,
                               encoded_missing_value=-1), categorical),
    ])


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {k: (df["split"] == k).to_numpy() for k in ("train", "val", "test")}
