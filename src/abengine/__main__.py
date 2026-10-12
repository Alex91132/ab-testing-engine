"""
Командная строка:

    python -m abengine generate --db data/shop.duckdb            # синтетические логи магазина
    python -m abengine report   --db data/shop.duckdb --out docs  # отчёты по всем экспериментам
    python -m abengine report   --db data/shop.duckdb --experiment checkout_redesign --out docs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb


def cmd_generate(args):
    from .dataset import generate, write_duckdb
    t0 = time.time()
    tables, _ = generate(n_users=args.users, seed=args.seed)
    write_duckdb(tables, args.db)
    sizes = ", ".join(f"{k}: {len(v):,}".replace(",", " ") for k, v in tables.items())
    print(f"{args.db}: {sizes} ({time.time() - t0:.0f} с)")


def cmd_report(args):
    from .analysis import analyze
    from .metrics import daily_metrics, experiment_ids, experiment_info, user_metrics
    from .report import render, render_index

    out = Path(args.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    with duckdb.connect(args.db, read_only=True) as con:
        ids = [args.experiment] if args.experiment else experiment_ids(con)
        analyses = []
        for exp in ids:
            a = analyze(user_metrics(con, exp), daily_metrics(con, exp), experiment_info(con, exp))
            path = out / "reports" / f"{exp}.html"
            path.write_text(render(a), encoding="utf-8")
            print(f"{exp:22s} {a.decision['status']:8s} → {path}")
            analyses.append(a)
    if not args.experiment:
        (out / "index.html").write_text(render_index(analyses), encoding="utf-8")
        print(f"список → {out / 'index.html'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abengine", description="Анализ A/B-экспериментов")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="сгенерировать синтетические логи в DuckDB")
    g.add_argument("--db", default="data/shop.duckdb")
    g.add_argument("--users", type=int, default=150_000)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)
    r = sub.add_parser("report", help="собрать HTML-отчёты")
    r.add_argument("--db", default="data/shop.duckdb")
    r.add_argument("--experiment", help="один эксперимент (по умолчанию — все + страница-список)")
    r.add_argument("--out", default="docs")
    r.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
