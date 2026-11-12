"""Score one booking and say why, in words a front-desk or revenue manager can read."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import features as F

DEFAULT_BOOKING = {
    "hotel": "City Hotel", "arrival_date": "2017-07-15", "lead_time": 60, "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
    "adults": 2, "children": 0, "babies": 0, "meal": "BB", "country": "PRT", "market_segment": "Online TA",
    "distribution_channel": "TA/TO", "is_repeated_guest": 0, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A", "deposit_type": "No Deposit", "agent": 9, "company": None, "customer_type": "Transient",
    "days_in_waiting_list": 0, "adr": 100.0,
}

LABELS = {
    "lead_time": "booking made far ahead or close in", "deposit_type": "deposit terms", "market_segment": "sales channel",
    "country": "guest country", "agent_id": "booking agent", "adr": "nightly price", "previous_cancellations": "earlier cancellations",
    "hotel": "which hotel", "customer_type": "customer type", "arrival_week": "time of year", "arrival_month": "time of year",
    "nights": "length of stay", "party_size": "party size", "adr_per_person": "price per guest", "distribution_channel": "distribution channel",
    "stays_in_weekend_nights": "weekend nights", "is_repeated_guest": "returning guest", "reserved_room_type": "room type",
    "previous_bookings_not_canceled": "earlier completed stays", "has_agent": "booked through an agent", "meal": "meal plan",
    "arrival_dow": "weekday of arrival", "adults": "number of adults", "children": "children", "babies": "babies",
    "days_in_waiting_list": "time on waiting list", "has_company": "company booking",
}


def raw_frame(booking: dict) -> pd.DataFrame:
    b = {**DEFAULT_BOOKING, **booking}
    arrival = pd.Timestamp(b["arrival_date"])
    row = {
        "hotel": b["hotel"], "lead_time": int(b["lead_time"]), "arrival_date_year": arrival.year,
        "arrival_date_month": arrival.strftime("%B"), "arrival_date_week_number": int(arrival.isocalendar().week),
        "arrival_date_day_of_month": arrival.day, "stays_in_weekend_nights": int(b["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(b["stays_in_week_nights"]), "adults": int(b["adults"]), "children": float(b["children"]),
        "babies": int(b["babies"]), "meal": b["meal"], "country": b["country"], "market_segment": b["market_segment"],
        "distribution_channel": b["distribution_channel"], "is_repeated_guest": int(b["is_repeated_guest"]),
        "previous_cancellations": int(b["previous_cancellations"]),
        "previous_bookings_not_canceled": int(b["previous_bookings_not_canceled"]),
        "reserved_room_type": b["reserved_room_type"], "assigned_room_type": b["reserved_room_type"], "booking_changes": 0,
        "deposit_type": b["deposit_type"], "agent": np.nan if b["agent"] in (None, "") else float(b["agent"]),
        "company": np.nan if b["company"] in (None, "") else float(b["company"]), "days_in_waiting_list": int(b["days_in_waiting_list"]),
        "customer_type": b["customer_type"], "adr": float(b["adr"]), "required_car_parking_spaces": 0, "total_of_special_requests": 0,
        "reservation_status": "Check-Out", "reservation_status_date": arrival.strftime("%Y-%m-%d"), "is_canceled": 0,
    }
    return D.add_dates(pd.DataFrame([row]))


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature push on the log-odds of cancelling, grouped back to the input columns."""
    prep, est = model.named_steps["prep"], model.named_steps["model"]
    Z = prep.transform(X)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    if hasattr(est, "coef_"):
        raw = np.asarray(Z) * est.coef_[0]
    else:
        raw = est.predict(Z, pred_contrib=True)[:, :-1]
    cats = sorted(F.CATEGORICAL, key=len, reverse=True)
    group = []
    for n in names:
        group.append(next((c for c in cats if n == c or n.startswith(c + "_")), n))
    frame = pd.DataFrame(raw, columns=group, index=X.index)
    return frame.T.groupby(level=0).sum().T


def reasons(model, X: pd.DataFrame, top: int = 3) -> list[dict]:
    contrib = contributions(model, X)
    out = []
    for i in range(len(X)):
        c = contrib.iloc[i].sort_values(key=np.abs, ascending=False).head(top)
        out.append({"up": [], "down": []})
        for col, v in c.items():
            label = LABELS.get(col, col.replace("_", " "))
            value = X.iloc[i].get(col, "")
            if isinstance(value, float) and value == int(value):
                value = int(value)
            text = f"{label} ({value})"
            out[-1]["up" if v > 0 else "down"].append(text)
    return out


def band(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"


def score_booking(model, booking: dict) -> dict:
    X = F.build_frame(raw_frame(booking))
    p = float(model.predict_proba(X)[:, 1][0])
    why = reasons(model, X)[0]
    parts = []
    if why["up"]:
        parts.append("pushes toward cancelling: " + ", ".join(why["up"]))
    if why["down"]:
        parts.append("pushes toward staying: " + ", ".join(why["down"]))
    return {"probability": round(p, 4), "band": band(p), "expected_show": round(1 - p, 4), "explanation": "; ".join(parts)}
