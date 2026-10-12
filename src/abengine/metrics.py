"""
Метрики: определения и SQL, который сводит события к пользовательским суммам.

Единица анализа — пользователь, попавший в эксперимент (первый заход в период
эксперимента). Для каждого считаются суммы за период эксперимента — от дня первого
захода до конца — и те же суммы за 28 дней до эксперимента (ковариаты для CUPED).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    kind: str                       # 'mean' — среднее по пользователям, 'ratio' — Σnum / Σden
    num: str
    den: str | None = None
    higher_is_better: bool = True
    fmt: str = "num"                # 'pct' | 'rub' | 'num'
    description: str = ""


METRICS = {m.key: m for m in [
    Metric("conversion", "Конверсия в покупку", "mean", "buyer", fmt="pct",
           description="Доля пользователей хотя бы с одним оплаченным заказом"),
    Metric("arpu", "Выручка на пользователя", "mean", "revenue", fmt="rub",
           description="Выручка оплаченных заказов / пользователи в эксперименте"),
    Metric("aov", "Средний чек", "ratio", "revenue", "orders", fmt="rub",
           description="Выручка / число оплаченных заказов"),
    Metric("ctr", "CTR карточек товара", "ratio", "clicks", "views", fmt="pct",
           description="Клики / просмотры карточек"),
    Metric("cancel_rate", "Доля отменённых заказов", "ratio", "cancelled", "all_orders",
           higher_is_better=False, fmt="pct", description="Отменённые / все заказы"),
]}

USER_METRICS_SQL = """
WITH e AS (
    SELECT start_date, end_date FROM experiments WHERE experiment_id = $exp
), a AS (
    SELECT user_id, variant, first_exposure FROM assignments WHERE experiment_id = $exp
), act AS (
    SELECT a.user_id,
           sum(d.views)  FILTER (WHERE d.date >= a.first_exposure)               AS views,
           sum(d.clicks) FILTER (WHERE d.date >= a.first_exposure)               AS clicks,
           sum(d.views)  FILTER (WHERE d.date <  e.start_date)                   AS pre_views,
           sum(d.clicks) FILTER (WHERE d.date <  e.start_date)                   AS pre_clicks
    FROM a CROSS JOIN e
    JOIN daily_activity d ON d.user_id = a.user_id
     AND d.date BETWEEN e.start_date - INTERVAL 28 DAY AND e.end_date
    GROUP BY a.user_id
), ord AS (
    SELECT a.user_id,
           count(*)                FILTER (WHERE o.order_date >= a.first_exposure)                           AS all_orders,
           count(*)                FILTER (WHERE o.order_date >= a.first_exposure AND o.status = 'completed') AS orders,
           count(*)                FILTER (WHERE o.order_date >= a.first_exposure AND o.status = 'cancelled') AS cancelled,
           sum(o.revenue)          FILTER (WHERE o.order_date >= a.first_exposure AND o.status = 'completed') AS revenue,
           count(*)                FILTER (WHERE o.order_date <  e.start_date)                               AS pre_all_orders,
           count(*)                FILTER (WHERE o.order_date <  e.start_date AND o.status = 'completed')    AS pre_orders,
           count(*)                FILTER (WHERE o.order_date <  e.start_date AND o.status = 'cancelled')    AS pre_cancelled,
           sum(o.revenue)          FILTER (WHERE o.order_date <  e.start_date AND o.status = 'completed')    AS pre_revenue
    FROM a CROSS JOIN e
    JOIN orders o ON o.user_id = a.user_id
     AND o.order_date BETWEEN e.start_date - INTERVAL 28 DAY AND e.end_date
    GROUP BY a.user_id
)
SELECT a.user_id, a.variant, u.platform,
       date_diff('day', e.start_date, a.first_exposure)      AS exposure_day,
       coalesce(act.views, 0)          AS views,
       coalesce(act.clicks, 0)         AS clicks,
       coalesce(ord.all_orders, 0)     AS all_orders,
       coalesce(ord.orders, 0)         AS orders,
       coalesce(ord.cancelled, 0)      AS cancelled,
       coalesce(ord.revenue, 0)::DOUBLE AS revenue,
       (coalesce(ord.orders, 0) > 0)::INTEGER AS buyer,
       coalesce(act.pre_views, 0)      AS pre_views,
       coalesce(act.pre_clicks, 0)     AS pre_clicks,
       coalesce(ord.pre_all_orders, 0) AS pre_all_orders,
       coalesce(ord.pre_orders, 0)     AS pre_orders,
       coalesce(ord.pre_cancelled, 0)  AS pre_cancelled,
       coalesce(ord.pre_revenue, 0)::DOUBLE AS pre_revenue,
       (coalesce(ord.pre_orders, 0) > 0)::INTEGER AS pre_buyer
FROM a CROSS JOIN e
JOIN users u USING (user_id)
LEFT JOIN act USING (user_id)
LEFT JOIN ord USING (user_id)
ORDER BY a.user_id
"""

DAILY_SQL = """
-- суммы по пользователю и дню эксперимента (только дни после первого захода)
WITH e AS (SELECT start_date, end_date FROM experiments WHERE experiment_id = $exp),
a AS (SELECT user_id, variant, first_exposure FROM assignments WHERE experiment_id = $exp),
act AS (
    SELECT d.user_id, date_diff('day', e.start_date, d.date) AS day, d.views, d.clicks
    FROM daily_activity d JOIN a USING (user_id) CROSS JOIN e
    WHERE d.date BETWEEN a.first_exposure AND e.end_date
), ord AS (
    SELECT o.user_id, date_diff('day', e.start_date, o.order_date) AS day,
           count(*) AS all_orders,
           count(*) FILTER (WHERE o.status = 'completed') AS orders,
           count(*) FILTER (WHERE o.status = 'cancelled') AS cancelled,
           coalesce(sum(o.revenue) FILTER (WHERE o.status = 'completed'), 0)::DOUBLE AS revenue
    FROM orders o JOIN a USING (user_id) CROSS JOIN e
    WHERE o.order_date BETWEEN a.first_exposure AND e.end_date
    GROUP BY 1, 2
)
SELECT user_id, day, coalesce(act.views, 0) AS views, coalesce(act.clicks, 0) AS clicks,
       coalesce(ord.all_orders, 0) AS all_orders, coalesce(ord.orders, 0) AS orders,
       coalesce(ord.cancelled, 0) AS cancelled, coalesce(ord.revenue, 0) AS revenue
FROM act FULL JOIN ord USING (user_id, day)
ORDER BY user_id, day
"""


def user_metrics(con: duckdb.DuckDBPyConnection, experiment_id: str) -> pd.DataFrame:
    return con.execute(USER_METRICS_SQL, {"exp": experiment_id}).df()


def daily_metrics(con: duckdb.DuckDBPyConnection, experiment_id: str) -> pd.DataFrame:
    return con.execute(DAILY_SQL, {"exp": experiment_id}).df()


def experiment_info(con: duckdb.DuckDBPyConnection, experiment_id: str) -> dict:
    row = con.execute("SELECT * FROM experiments WHERE experiment_id = ?", [experiment_id]).df()
    if row.empty:
        raise KeyError(f"нет эксперимента {experiment_id!r}")
    return row.iloc[0].to_dict()


def experiment_ids(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute("SELECT experiment_id FROM experiments ORDER BY start_date, experiment_id").fetchall()]
