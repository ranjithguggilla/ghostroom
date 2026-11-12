"""Gradio front end: describe a booking, get a cancellation probability and the reasons."""

from __future__ import annotations

from pathlib import Path

import joblib

from .scoring import DEFAULT_BOOKING, score_booking


def build(model_path: str | Path = "outputs/ghostroom_model.joblib"):
    import gradio as gr

    model = joblib.load(model_path)["model"]

    def predict(hotel, arrival_date, lead_time, nights_weekend, nights_week, adults, children, market_segment, deposit_type,
                agent, country, adr, previous_cancellations, customer_type):
        out = score_booking(model, {
            "hotel": hotel, "arrival_date": arrival_date, "lead_time": lead_time, "stays_in_weekend_nights": nights_weekend,
            "stays_in_week_nights": nights_week, "adults": adults, "children": children, "market_segment": market_segment,
            "deposit_type": deposit_type, "agent": agent, "country": country, "adr": adr,
            "previous_cancellations": previous_cancellations, "customer_type": customer_type,
        })
        return out["probability"], out["band"], out["explanation"]

    d = DEFAULT_BOOKING
    with gr.Blocks(title="ghostroom") as demo:
        gr.Markdown("# ghostroom\nHow likely is this booking to be cancelled before arrival?")
        with gr.Row():
            with gr.Column():
                hotel = gr.Dropdown(["City Hotel", "Resort Hotel"], value=d["hotel"], label="hotel")
                arrival = gr.Textbox(value=d["arrival_date"], label="arrival date (YYYY-MM-DD)")
                lead = gr.Slider(0, 700, value=d["lead_time"], step=1, label="days booked ahead")
                wk_end = gr.Slider(0, 6, value=1, step=1, label="weekend nights")
                wk_day = gr.Slider(0, 14, value=2, step=1, label="week nights")
                adults = gr.Slider(1, 6, value=2, step=1, label="adults")
                kids = gr.Slider(0, 4, value=0, step=1, label="children")
                seg = gr.Dropdown(["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate", "Complementary", "Aviation"],
                                  value=d["market_segment"], label="market segment")
                dep = gr.Dropdown(["No Deposit", "Non Refund", "Refundable"], value=d["deposit_type"], label="deposit type")
                agent = gr.Number(value=d["agent"], label="agent id (blank for none)")
                country = gr.Textbox(value=d["country"], label="guest country code")
                adr = gr.Number(value=d["adr"], label="average daily rate")
                prev = gr.Slider(0, 5, value=0, step=1, label="earlier cancellations")
                cust = gr.Dropdown(["Transient", "Transient-Party", "Contract", "Group"], value=d["customer_type"], label="customer type")
                btn = gr.Button("score booking")
            with gr.Column():
                prob = gr.Number(label="probability of cancelling")
                band = gr.Textbox(label="risk band")
                why = gr.Textbox(label="why", lines=5)
        btn.click(predict, [hotel, arrival, lead, wk_end, wk_day, adults, kids, seg, dep, agent, country, adr, prev, cust], [prob, band, why])
    return demo


def launch():
    build().launch()
