"""
Проверка методов симуляциями → results/validation_*.csv и сводка results/validation.csv.

    python scripts/run_validation.py                 # 2000 A/A + 1000 A/B, ~10 минут на 8 ядрах
    python scripts/run_validation.py --aa 40 --ab 20 # быстрая проверка
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from abengine.simulate import Effect  # noqa: E402
from abengine.validate import rejection_rates, run  # noqa: E402

AB_EFFECT = Effect(buy=1.10, basket=1.06, ctr=1.04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=2000, help="A/A-повторов")
    ap.add_argument("--ab", type=int, default=1000, help="A/B-повторов")
    ap.add_argument("--users", type=int, default=20_000)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    aa = run(args.aa, args.users, Effect(), seed0=0)
    aa.to_csv(out / "validation_aa.csv", index=False)
    print(f"A/A: {len(aa)} экспериментов за {time.time() - t0:.0f} с", flush=True)

    t0 = time.time()
    ab = run(args.ab, args.users, AB_EFFECT, seed0=100_000)
    ab.to_csv(out / "validation_ab.csv", index=False)
    print(f"A/B: {len(ab)} экспериментов за {time.time() - t0:.0f} с", flush=True)

    summary = pd.DataFrame({"aa_false_positive": rejection_rates(aa), "ab_power": rejection_rates(ab)})
    summary.index.name = "procedure"
    summary.to_csv(out / "validation.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
