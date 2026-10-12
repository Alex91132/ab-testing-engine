"""
Генератор поведения пользователей интернет-магазина «пользователь × день».

Каждый пользователь получает устойчивые черты: насколько часто заходит, как часто
кликает по карточкам, как часто покупает, какой у него средний чек. Поэтому поведение
до эксперимента предсказывает поведение во время — на этом держится CUPED.

За день пользователь либо заходит, либо нет; если зашёл — смотрит товары, кликает и
с некоторой вероятностью оформляет заказ (не больше одного в день); часть заказов
отменяется. Эффект эксперимента — множители к вероятностям и чеку у группы treatment
начиная с дня первого захода в период эксперимента; новизна — затухающая добавка.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

PLATFORMS = ("web", "android", "ios")
PLATFORM_SHARE = (0.45, 0.35, 0.20)
PLATFORM_ACTIVITY = (1.0, 1.15, 1.3)


@dataclass(frozen=True)
class Effect:
    """Истинный эффект для treatment: множители (1.0 — без изменений)."""
    buy: float = 1.0          # вероятность заказа в день визита
    ctr: float = 1.0          # вероятность клика на просмотр
    basket: float = 1.0       # средний чек
    cancel: float = 1.0       # вероятность отмены заказа
    novelty: float = 0.0      # доп. доля эффекта в первые дни: 1.0 — вдвое сильнее, затухает за ~3 дня


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    salt: str
    effect: Effect = Effect()
    primary_metric: str = "conversion"
    hypothesis: str = ""
    traffic: float = 1.0      # доля пользователей в эксперименте
    log_loss: float = 0.0     # доля «потерянных» записей о назначении у малоактивных в treatment (баг)


@dataclass
class Population:
    user_id: np.ndarray
    platform: np.ndarray
    visit_p: np.ndarray
    views_mean: np.ndarray
    ctr: np.ndarray
    buy_p: np.ndarray
    basket_mu: np.ndarray
    cancel_p: np.ndarray

    @property
    def n(self) -> int:
        return len(self.user_id)


@dataclass
class Period:
    """Матрицы «пользователь × день» за период."""
    visits: np.ndarray        # bool
    views: np.ndarray         # int
    clicks: np.ndarray        # int
    orders: np.ndarray        # bool: был ли заказ
    revenue: np.ndarray       # руб., 0 без заказа
    cancelled: np.ndarray     # bool
    meta: dict = field(default_factory=dict)


def make_population(n: int, rng: np.random.Generator) -> Population:
    platform = rng.choice(len(PLATFORMS), size=n, p=PLATFORM_SHARE)
    activity = rng.gamma(0.8, 1.0, n) * np.take(PLATFORM_ACTIVITY, platform)
    return Population(
        user_id=np.arange(1, n + 1),
        platform=platform,
        visit_p=np.clip(1 - np.exp(-0.25 * activity), 0.003, 0.9),
        views_mean=rng.gamma(3.0, 1.6, n),
        ctr=rng.beta(2.0, 16.0, n),
        buy_p=rng.beta(0.5, 8.0, n),
        basket_mu=rng.normal(np.log(2200), 0.6, n),
        cancel_p=rng.beta(1.5, 22.0, n),
    )


def assign(user_id: np.ndarray, salt: str, traffic: float = 1.0) -> np.ndarray:
    """
    Детерминированное назначение по хешу: md5(salt:user_id) → корзина 0..9999.
    Корзины [0, traffic·10⁴) делятся пополам: чётные — control (0), нечётные — treatment (1);
    остальные пользователи в эксперимент не попадают (−1). Разные соли — независимые эксперименты.
    """
    buckets = np.fromiter(
        (int(hashlib.md5(f"{salt}:{u}".encode()).hexdigest()[:8], 16) % 10_000 for u in user_id),
        dtype=np.int64, count=len(user_id))
    return np.where(buckets < traffic * 10_000, buckets % 2, -1)


def weekday_factor(days: np.ndarray, start_weekday: int) -> np.ndarray:
    """Выходные чуть активнее будних (start_weekday: 0 — понедельник)."""
    wd = (start_weekday + days) % 7
    return np.where(wd >= 5, 1.15, 1.0)


def simulate_period(pop: Population, n_days: int, rng: np.random.Generator, start_weekday: int = 0,
                    mult: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None) -> Period:
    """
    mult(visits) → множители эффекта формы (N, n_days) для 'buy', 'ctr', 'basket', 'cancel'
    (1 там, где эффекта нет). Эффект зависит от дня первого захода, поэтому множители
    считаются уже после того, как разыграны визиты. Без mult — период без экспериментов.
    """
    n = pop.n
    shape = (n, n_days)
    one = np.ones(shape)
    days = np.arange(n_days)

    visit_p = np.clip(pop.visit_p[:, None] * weekday_factor(days, start_weekday)[None, :], 0, 0.95)
    visits = rng.random(shape) < visit_p
    given = mult(visits) if mult else {}
    m = {k: given.get(k, one) for k in ("buy", "ctr", "basket", "cancel")}
    views = np.where(visits, 1 + rng.poisson(pop.views_mean[:, None] * one), 0)
    ctr = np.clip(pop.ctr[:, None] * m["ctr"], 0, 0.95)
    clicks = rng.binomial(views, ctr)
    buy_p = np.clip(pop.buy_p[:, None] * m["buy"], 0, 0.9)
    orders = visits & (rng.random(shape) < buy_p)
    basket = np.exp(pop.basket_mu[:, None] + rng.normal(0, 0.6, shape)) * m["basket"]
    revenue = np.where(orders, np.round(basket, 2), 0.0)
    cancel_p = np.clip(pop.cancel_p[:, None] * m["cancel"], 0, 0.9)
    cancelled = orders & (rng.random(shape) < cancel_p)
    return Period(visits, views, clicks, orders, revenue, cancelled)


def effect_multipliers(variants: dict[str, np.ndarray], experiments: list[Experiment],
                       exposure_day: np.ndarray, n_days: int) -> dict[str, np.ndarray]:
    """Перемножает эффекты всех экспериментов для каждого пользователя и дня (после первого захода)."""
    n = len(exposure_day)
    days = np.arange(n_days)[None, :]
    since = days - exposure_day[:, None]                        # дней с первого захода
    exposed = (exposure_day[:, None] >= 0) & (since >= 0)
    out = {k: np.ones((n, n_days)) for k in ("buy", "ctr", "basket", "cancel")}
    for exp in experiments:
        treated = (variants[exp.key] == 1)[:, None] & exposed
        boost = 1 + exp.effect.novelty * np.exp(-np.maximum(since, 0) / 3.0)
        for k in out:
            base = getattr(exp.effect, k)
            if base != 1.0:
                out[k] = np.where(treated, out[k] * (1 + (base - 1) * boost), out[k])
    return out


def first_true(mask: np.ndarray) -> np.ndarray:
    """Индекс первого True по строке, −1 если нет."""
    has = mask.any(axis=1)
    return np.where(has, mask.argmax(axis=1), -1)
