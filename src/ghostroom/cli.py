"""Command line: run the study, show headline numbers, score one booking, launch the demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ghostroom", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch data, run every study, write outputs/ and assets/")
    r.add_argument("--out", default="outputs")
    r.add_argument("--assets", default="assets")
    s = sub.add_parser("show", help="print the headline numbers from outputs/metrics.json")
    s.add_argument("--out", default="outputs")
    c = sub.add_parser("score", help="score one booking given as key=value pairs")
    c.add_argument("--out", default="outputs")
    c.add_argument("--model", default="ghostroom_model.joblib", help="file name inside --out")
    c.add_argument("fields", nargs="*", help="e.g. lead_time=120 deposit_type='Non Refund' agent=9")
    t = sub.add_parser("retrain", help="refit on bookings whose outcome was known on a given date")
    t.add_argument("--out", default="outputs")
    t.add_argument("--as-of", default=None, help="YYYY-MM-DD, default is the last outcome date in the file")
    sub.add_parser("app", help="launch the Gradio demo")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from .pipeline import run
        run(args.out, args.assets)
        print(f"wrote {args.out}/metrics.json")
    elif args.cmd == "show":
        m = json.loads((Path(args.out) / "metrics.json").read_text())
        t = m["final_test"]
        print(f"shipped model: {m['shipped']}" + (" with isotonic calibration" if t["calibrated"] else ""))
        print(f"test AUC {t['auc']:.4f}  log loss {t['log_loss']:.4f}  ECE {t['ece']:.4f}")
        for row in m["delayed_labels"]["table"]:
            print(f"  {row['method']:<44} log loss {row['log_loss']:.4f}  AUC {row['auc']:.4f}")
        e = m["economics"]["walk_3x"]
        for k, v in e["table"].items():
            print(f"  {k:<20} profit {v['profit']:,.0f}  walked {v['walked']:,.0f}")
    elif args.cmd == "score":
        import joblib

        from .scoring import score_booking
        bundle = joblib.load(Path(args.out) / args.model)
        fields = {}
        for kv in args.fields:
            k, v = kv.split("=", 1)
            fields[k] = float(v) if v.replace(".", "", 1).isdigit() else v
        print(json.dumps(score_booking(bundle["model"], fields), indent=2))
    elif args.cmd == "retrain":
        from .pipeline import retrain
        info = retrain(args.out, args.as_of)
        print(f"refit {info['name']} on {info['rows']:,} bookings known as of {info['as_of']} (cancel rate {info['cancel_rate']:.3f})")
    elif args.cmd == "app":
        from .app import launch
        launch()


if __name__ == "__main__":
    main()
