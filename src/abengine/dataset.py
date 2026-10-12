"""
Учебный набор данных интернет-магазина в DuckDB: пользователи, активность по дням,
заказы, эксперименты и назначения — так, как они лежат в хранилище событий.

    python -m abengine generate --out data/shop.duckdb
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .simulate import (PLATFORMS, Effect, Experiment, assign, effect_multipliers, first_true,
                       make_population, simulate_period)

PRE_START = date(2026, 8, 10)       # понедельник: 4 недели до экспериментов
EXP_START = date(2026, 9, 7)        # понедельник: 3 недели экспериментов
PRE_DAYS, EXP_DAYS = 28, 21

EXPERIMENTS = [
    Experiment("checkout_redesign", "Новый чекаут в один экран", "salt-checkout-2026",
               Effect(buy=1.06), primary_metric="conversion",
               hypothesis="Оформление заказа в один экран вместо трёх поднимет конверсию в покупку"),
    Experiment("recommendations_v2", "Блок рекомендаций v2", "salt-recs-2026",
               Effect(ctr=1.06, basket=1.01, cancel=1.20, novelty=1.0), primary_metric="ctr",
               hypothesis="Новая модель рекомендаций повысит кликабельность карточек товара"),
    Experiment("aa_platform_check", "A/A-проверка платформы", "salt-aa-2026",
               primary_metric="conversion",
               hypothesis="Группы ничем не отличаются: проверка сплитования и расчёта метрик"),
    Experiment("promo_banner", "Баннер акции на главной", "salt-promo-2026",
               Effect(buy=1.01), primary_metric="arpu", log_loss=0.15,
               hypothesis="Баннер с акцией увеличит выручку на пользователя"),
]


def generate(n_users: int = 150_000, seed: int = 7, experiments=EXPERIMENTS):
    """Возвращает таблицы как DataFrame'ы и истинные эффекты (для проверки отчёта)."""
    rng = np.random.default_rng(seed)
    pop = make_population(n_users, rng)
    variants = {e.key: assign(pop.user_id, e.salt, e.traffic) for e in experiments}

    pre = simulate_period(pop, PRE_DAYS, rng, start_weekday=PRE_START.weekday())
    holder = {}

    def mult(visits):
        holder["exposure"] = first_true(visits)
        return effect_multipliers(variants, experiments, holder["exposure"], EXP_DAYS)

    exp = simulate_period(pop, EXP_DAYS, rng, start_weekday=EXP_START.weekday(), mult=mult)
    exposure = holder["exposure"]

    users = pd.DataFrame({"user_id": pop.user_id, "platform": np.take(PLATFORMS, pop.platform)})

    activity, orders = [], []
    for period, start in ((pre, PRE_START), (exp, EXP_START)):
        u, d = np.nonzero(period.visits)
        days = pd.to_datetime(start) + pd.to_timedelta(d, unit="D")
        activity.append(pd.DataFrame({"user_id": pop.user_id[u], "date": days.date,
                                      "views": period.views[u, d], "clicks": period.clicks[u, d]}))
        u, d = np.nonzero(period.orders)
        days = pd.to_datetime(start) + pd.to_timedelta(d, unit="D")
        orders.append(pd.DataFrame({"user_id": pop.user_id[u], "order_date": days.date,
                                    "revenue": period.revenue[u, d],
                                    "status": np.where(period.cancelled[u, d], "cancelled", "completed")}))
    activity = pd.concat(activity, ignore_index=True)
    orders = pd.concat(orders, ignore_index=True).sort_values(["order_date", "user_id"], ignore_index=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))

    assignments, meta = [], []
    low_activity = pop.visit_p < np.quantile(pop.visit_p, 0.4)
    for e in experiments:
        v = variants[e.key]
        keep = (v >= 0) & (exposure >= 0)
        if e.log_loss:   # баг логирования: у малоактивных в treatment часть назначений теряется
            lost = (v == 1) & low_activity & (rng.random(n_users) < e.log_loss)
            keep &= ~lost
        idx = np.flatnonzero(keep)
        assignments.append(pd.DataFrame({
            "experiment_id": e.key, "user_id": pop.user_id[idx],
            "variant": np.where(v[idx] == 1, "treatment", "control"),
            "first_exposure": [EXP_START + timedelta(days=int(x)) for x in exposure[idx]]}))
        meta.append({"experiment_id": e.key, "title": e.title, "hypothesis": e.hypothesis,
                     "primary_metric": e.primary_metric, "salt": e.salt,
                     "start_date": EXP_START, "end_date": EXP_START + timedelta(days=EXP_DAYS - 1),
                     "traffic": e.traffic})
    assignments = pd.concat(assignments, ignore_index=True)
    experiments_df = pd.DataFrame(meta)
    truth = {e.key: asdict(e.effect) | {"log_loss": e.log_loss} for e in experiments}
    return {"users": users, "daily_activity": activity, "orders": orders,
            "experiments": experiments_df, "assignments": assignments}, truth


SCHEMA = """
CREATE TABLE users (user_id INTEGER PRIMARY KEY, platform VARCHAR NOT NULL);
CREATE TABLE daily_activity (user_id INTEGER NOT NULL, date DATE NOT NULL, views INTEGER NOT NULL,
                             clicks INTEGER NOT NULL, PRIMARY KEY (user_id, date));
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, order_date DATE NOT NULL,
                     revenue DECIMAL(12, 2) NOT NULL, status VARCHAR NOT NULL);
CREATE TABLE experiments (experiment_id VARCHAR PRIMARY KEY, title VARCHAR NOT NULL, hypothesis VARCHAR,
                          primary_metric VARCHAR NOT NULL, salt VARCHAR NOT NULL,
                          start_date DATE NOT NULL, end_date DATE NOT NULL, traffic DOUBLE NOT NULL);
CREATE TABLE assignments (experiment_id VARCHAR NOT NULL, user_id INTEGER NOT NULL, variant VARCHAR NOT NULL,
                          first_exposure DATE NOT NULL, PRIMARY KEY (experiment_id, user_id));
"""


def write_duckdb(tables: dict[str, pd.DataFrame], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with duckdb.connect(str(path)) as con:
        con.execute(SCHEMA)
        for name in ("users", "daily_activity", "orders", "experiments", "assignments"):
            df = tables[name]
            con.register("src", df)
            con.execute(f"INSERT INTO {name} SELECT {', '.join(df.columns)} FROM src")
            con.unregister("src")
