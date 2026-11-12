import numpy as np
import pandas as pd
import pytest

from ghostroom import data as D


def synthetic(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """A small table with the same columns as the hotel file and a cancellation signal worth learning."""
    rng = np.random.default_rng(seed)
    arrival = pd.Timestamp("2015-07-01") + pd.to_timedelta(rng.integers(0, 790, n), unit="D")
    lead = np.minimum(rng.exponential(70, n).astype(int), 600)
    hotel = rng.choice(["City Hotel", "Resort Hotel"], n, p=[0.65, 0.35])
    segment = rng.choice(["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"], n, p=[0.45, 0.2, 0.15, 0.15, 0.05])
    deposit = rng.choice(["No Deposit", "Non Refund"], n, p=[0.88, 0.12])
    agent = rng.choice([9.0, 240.0, 14.0, np.nan], n, p=[0.35, 0.2, 0.1, 0.35])
    logit = -1.6 + lead / 90 + (segment == "Groups") * 0.8 + (segment == "Direct") * -0.7 + (agent == 9) * 0.5 + (deposit == "Non Refund") * 6
    canceled = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    nights_w, nights_e = rng.integers(0, 6, n), rng.integers(0, 3, n)
    nights = nights_w + nights_e
    booked = arrival - pd.to_timedelta(lead, unit="D")
    cancel_day = booked + pd.to_timedelta((rng.random(n) * lead).astype(int), unit="D")
    outcome = np.where(canceled == 1, cancel_day, arrival + pd.to_timedelta(nights, unit="D"))
    df = pd.DataFrame({
        "hotel": hotel, "is_canceled": canceled, "lead_time": lead,
        "arrival_date_year": arrival.year, "arrival_date_month": arrival.strftime("%B"),
        "arrival_date_week_number": arrival.isocalendar().week.to_numpy(), "arrival_date_day_of_month": arrival.day,
        "stays_in_weekend_nights": nights_e, "stays_in_week_nights": nights_w, "adults": rng.integers(1, 4, n),
        "children": rng.choice([0.0, 1.0, np.nan], n, p=[0.85, 0.14, 0.01]), "babies": 0,
        "meal": rng.choice(["BB", "HB", "SC"], n), "country": rng.choice(["PRT", "GBR", "ESP", "FRA", None], n),
        "market_segment": segment, "distribution_channel": rng.choice(["TA/TO", "Direct", "Corporate"], n),
        "is_repeated_guest": rng.integers(0, 2, n), "previous_cancellations": rng.integers(0, 2, n),
        "previous_bookings_not_canceled": 0, "reserved_room_type": rng.choice(["A", "D", "E"], n),
        "assigned_room_type": rng.choice(["A", "D", "E"], n), "booking_changes": rng.integers(0, 3, n) * (1 - canceled),
        "deposit_type": deposit, "agent": agent, "company": np.nan, "days_in_waiting_list": 0,
        "customer_type": rng.choice(["Transient", "Contract"], n), "adr": rng.normal(100, 25, n).clip(20),
        "required_car_parking_spaces": rng.integers(0, 2, n) * (1 - canceled),
        "total_of_special_requests": rng.integers(0, 3, n) * (1 - canceled),
        "reservation_status": np.where(canceled == 1, "Canceled", "Check-Out"),
        "reservation_status_date": pd.to_datetime(outcome).strftime("%Y-%m-%d"),
    })
    df = D.add_dates(df)
    df["split"] = "train"
    df.loc[df["arrival"] >= D.VAL_START, "split"] = "val"
    df.loc[df["arrival"] >= D.TEST_START, "split"] = "test"
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def frame():
    return synthetic()
