"""
Проверка методов симуляциями. Тысячи экспериментов на свежих популяциях:

* A/A (эффекта нет) — доля ложных срабатываний должна быть ≈ α = 5 %:
  правильные методы её держат, «t-тест по заказам» и ежедневное подглядывание — нет;
* A/B с известным эффектом — мощность: насколько CUPED помогает поймать эффект,
  во что обходится mSPRT по сравнению с одним взглядом в конце.

Пользовательские суммы считаются из матриц «пользователь × день» так же, как их считает
SQL в metrics.py (это проверяет тест test_sql_matches_numpy).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as st

from . import stats
from .simulate import Effect, Experiment, effect_multipliers, first_true, make_population, simulate_period

ALPHA = 0.05


def _sums(period):
    """Суммы по пользователю за период — как в USER_METRICS_SQL (только оплаченные заказы)."""
    paid = period.orders & ~period.cancelled
    return dict(
        buyer=(paid.sum(axis=1) > 0).astype(float),
        revenue=np.where(paid, period.revenue, 0).sum(axis=1),
        orders=paid.sum(axis=1).astype(float),
        views=period.views.sum(axis=1).astype(float),
        clicks=period.clicks.sum(axis=1).astype(float),
    )


OBF_LOOKS = (7, 14)                      # еженедельные запланированные взгляды
TAU_REL = 0.10                           # mSPRT: τ ≈ ожидаемый эффект (здесь ~10 % от уровня конверсии)
_OBF_CACHE: dict = {}


def one_experiment(seed: int, n_users: int, effect: Effect, exp_days: int = 14, pre_days: int = 28) -> dict:
    """Один эксперимент: p-value всех проверяемых процедур."""
    rng = np.random.default_rng(seed)
    pop = make_population(n_users, rng)
    variant = rng.integers(0, 2, n_users)              # случайное назначение 50/50
    exp_cfg = [Experiment("x", "x", "x", effect)]
    holder = {}

    def mult(visits):
        holder["expo"] = first_true(visits)
        return effect_multipliers({"x": variant}, exp_cfg, holder["expo"], exp_days)

    pre = simulate_period(pop, pre_days, rng)
    cur = simulate_period(pop, exp_days, rng, mult=mult)
    expo = holder["expo"]
    inn = expo >= 0
    c, t = inn & (variant == 0), inn & (variant == 1)
    y, x = _sums(cur), _sums(pre)
    out = {}

    # --- пользовательские метрики
    out["conv_welch"] = stats.mean_test(y["buyer"][c], y["buyer"][t]).p_value
    out["conv_cuped"] = stats.cuped(y["buyer"][c], y["buyer"][t], x["buyer"][c], x["buyer"][t]).p_value
    out["arpu_welch"] = stats.mean_test(y["revenue"][c], y["revenue"][t]).p_value
    out["arpu_cuped"] = stats.cuped(y["revenue"][c], y["revenue"][t], x["revenue"][c], x["revenue"][t]).p_value

    # --- отношения: дельта-метод против «единица = заказ / просмотр»
    out["aov_delta"] = stats.ratio_test(y["revenue"][c], y["orders"][c], y["revenue"][t], y["orders"][t]).p_value
    ok = cur.orders & ~cur.cancelled
    per_order_c = cur.revenue[c][ok[c]]
    per_order_t = cur.revenue[t][ok[t]]
    out["aov_naive"] = stats.naive_unit_test(per_order_c, per_order_t).p_value
    out["ctr_delta"] = stats.ratio_test(y["clicks"][c], y["views"][c], y["clicks"][t], y["views"][t]).p_value
    k_c, n_c, k_t, n_t = y["clicks"][c].sum(), y["views"][c].sum(), y["clicks"][t].sum(), y["views"][t].sum()
    p_pool = (k_c + k_t) / (n_c + n_t)
    z = (k_t / n_t - k_c / n_c) / np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    out["ctr_naive"] = float(2 * st.norm.sf(abs(z)))     # z-тест долей по просмотрам

    # --- подглядывание: конверсия каждый день
    cum_orders = np.cumsum(cur.orders & ~cur.cancelled, axis=1)
    diffs, ses, fixed = [], [], []
    for d in range(exp_days):
        cd, td = c & (expo <= d), t & (expo <= d)
        b = (cum_orders[:, d] > 0).astype(float)
        r = stats.mean_test(b[cd], b[td])
        diffs.append(r.diff), ses.append(r.se), fixed.append(r.p_value)
    tau = TAU_REL * float(y["buyer"][c].mean())
    p_seq, _, _ = stats.msprt(np.array(diffs), np.array(ses), tau)
    out["peek_fixed_any"] = float(min(fixed))          # «остановимся, как только p < α»
    out["peek_msprt_any"] = float(p_seq.min())
    # О'Брайен — Флеминг: взгляды на 7-й и 14-й день. Доля информации — по числу пользователей:
    # у накопительной метрики дисперсия оценки со временем может и расти, 1/SE² для этого не годится
    looks = [d - 1 for d in OBF_LOOKS]
    n_look = np.array([(inn & (expo <= d)).sum() for d in looks], dtype=float)
    frac = tuple(np.round(n_look / n_look[-1], 2))
    if frac not in _OBF_CACHE:
        _OBF_CACHE[frac] = stats.obf_boundaries(frac)
    bounds = _OBF_CACHE[frac]
    zs = np.abs(np.array(diffs)[looks] / np.array(ses)[looks])
    out["peek_obf_any"] = 0.0 if (zs > bounds).any() else 1.0   # 0 — отвергли на каком-то взгляде

    # --- SRM на честном сплите и семейство из 4 вторичных метрик
    out["srm"] = stats.srm_test(int(c.sum()), int(t.sum()))[1]
    family = [out["arpu_cuped"], out["aov_delta"], out["ctr_delta"], out["conv_cuped"]]
    out["family_min_raw"] = float(min(family))
    out["family_min_holm"] = float(stats.holm(family).min())
    return out


def _run(task):
    seed, n_users, effect = task
    return one_experiment(seed, n_users, effect)


def run(n_reps: int, n_users: int = 20_000, effect: Effect = Effect(), seed0: int = 0,
        workers: int | None = None) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    tasks = [(seed0 + i, n_users, effect) for i in range(n_reps)]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_run, tasks, chunksize=max(1, n_reps // (workers * 4))))
    return pd.DataFrame(rows)


def rejection_rates(df: pd.DataFrame) -> pd.Series:
    """Доля повторов с p < α (для SRM — p < 0,001)."""
    rates = {c: (df[c] < (0.001 if c == "srm" else ALPHA)).mean() for c in df.columns}
    return pd.Series(rates)
