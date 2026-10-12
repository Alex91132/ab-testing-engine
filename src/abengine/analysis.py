"""
Анализ одного эксперимента: проверки здоровья → метрики → последовательный анализ →
срезы → решение. Порядок важен: при сломанном сплите (SRM) результаты метрик
не интерпретируются, сколько бы звёздочек они ни дали.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import stats
from .fmt import ci, num, p_eq, pct
from .metrics import METRICS, Metric

ALPHA = 0.05
SRM_ALPHA = 0.001          # порог SRM строже: проверка идёт в каждом эксперименте, ложная тревога дорога
GUARDRAILS = ("cancel_rate",)
SEQ_TAU = 0.05             # mSPRT: масштаб ожидаемого эффекта — 5 % от уровня метрики в control


@dataclass
class MetricResult:
    metric: Metric
    role: str                          # primary | secondary | guardrail
    raw: stats.TestResult              # без снижения дисперсии
    final: stats.TestResult            # CUPED (для отношений — CUPED + линеаризация)
    p_adj: float | None = None         # с поправкой Холма (только secondary)
    mde: float = 0.0                   # относительный MDE при текущем объёме (мощность 80 %)

    @property
    def p(self) -> float:
        return self.p_adj if self.p_adj is not None else self.final.p_value

    @property
    def significant(self) -> bool:
        return self.p < ALPHA

    @property
    def good(self) -> bool:
        """Изменение в желаемую сторону."""
        return (self.final.diff > 0) == self.metric.higher_is_better


@dataclass
class Analysis:
    info: dict
    n: dict
    srm: tuple[float, float]
    balance: stats.TestResult
    results: list[MetricResult]
    sequential: pd.DataFrame
    segments: pd.DataFrame
    decision: dict = field(default_factory=dict)

    @property
    def primary(self) -> MetricResult:
        return next(r for r in self.results if r.role == "primary")


def _split(df: pd.DataFrame, col: str):
    return df.loc[df.variant == "control", col].to_numpy(float), df.loc[df.variant == "treatment", col].to_numpy(float)


def test_metric(df: pd.DataFrame, m: Metric, alpha: float = ALPHA):
    """(без CUPED, с CUPED) для метрики по пользовательской таблице."""
    if m.kind == "mean":
        y_c, y_t = _split(df, m.num)
        x_c, x_t = _split(df, "pre_" + m.num)
        return stats.mean_test(y_c, y_t, alpha), stats.cuped(y_c, y_t, x_c, x_t, alpha)
    n_c, n_t = _split(df, m.num)
    d_c, d_t = _split(df, m.den)
    pn_c, pn_t = _split(df, "pre_" + m.num)
    pd_c, pd_t = _split(df, "pre_" + m.den)
    # дельта-метод берёт всех пользователей эксперимента: у кого знаменатель 0, тот просто не добавляет веса
    raw = stats.ratio_test(n_c, d_c, n_t, d_t, alpha)
    return raw, stats.ratio_cuped(n_c, d_c, n_t, d_t, pn_c, pd_c, pn_t, pd_t, alpha)


def sequential(daily: pd.DataFrame, users: pd.DataFrame, m: Metric, n_days: int, tau_rel: float = SEQ_TAU):
    """
    Накопленный результат по дням: на день d — пользователи, зашедшие до d включительно,
    и их суммы за дни ≤ d; тот же метод, что в итоговой таблице (CUPED). Для каждого дня —
    фиксированный тест (как будто смотрим один раз) и всегда-валидные p-value / интервал mSPRT.
    """
    ids = users.user_id.to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    expo = users.exposure_day.to_numpy()
    cols = {"orders"} if m.num == "buyer" else {m.num} | ({m.den} if m.den else set())
    rows = pos.loc[daily.user_id].to_numpy()
    cube = {}
    for c in cols:
        arr = np.zeros((len(ids), n_days))
        np.add.at(arr, (rows, daily.day.to_numpy()), daily[c].to_numpy(float))
        cube[c] = np.cumsum(arr, axis=1)
    out = []
    for d in range(n_days):
        part = users.loc[expo <= d].copy()
        inn = expo <= d
        if m.num == "buyer":
            part["buyer"] = (cube["orders"][inn, d] > 0).astype(float)
        for c in cols:
            part[c] = cube[c][inn, d]
        _, r = test_metric(part, m)
        out.append(dict(day=d + 1, users=int(inn.sum()), control=r.control, diff=r.diff, se=r.se,
                        lift=r.lift, lift_lo=r.lift_ci[0], lift_hi=r.lift_ci[1], p_fixed=r.p_value))
    seq = pd.DataFrame(out)
    tau = tau_rel * abs(seq.control.iloc[-1])
    p, lo, hi = stats.msprt(seq["diff"].to_numpy(), seq.se.to_numpy(), tau)
    seq["p_msprt"] = p
    seq["av_lift_lo"] = lo / seq.control
    seq["av_lift_hi"] = hi / seq.control
    return seq


def segments(df: pd.DataFrame, m: Metric, by: str = "platform") -> pd.DataFrame:
    rows = []
    for value, part in df.groupby(by):
        _, final = test_metric(part, m)
        rows.append(dict(segment=value, users=len(part), lift=final.lift, lo=final.lift_ci[0],
                         hi=final.lift_ci[1], p=final.p_value))
    return pd.DataFrame(rows)


def decide(a: Analysis) -> dict:
    srm_p = a.srm[1]
    if srm_p < SRM_ALPHA:
        return dict(status="invalid", title="Результатам нельзя доверять",
                    text=f"Соотношение групп {num(a.n['control'])} : {num(a.n['treatment'])} вместо 50/50 "
                         f"(SRM, {p_eq(srm_p)}). Пользователи теряются неслучайно — эффект смещён. "
                         "Найти причину (логирование, редиректы, боты), исправить и перезапустить.")
    bad = [r for r in a.results if r.role == "guardrail" and r.final.p_value < ALPHA and not r.good]
    if bad:
        names = ", ".join(r.metric.title.lower() for r in bad)
        return dict(status="stop", title="Не раскатывать: пострадала защитная метрика",
                    text=f"Статистически значимо ухудшилась {names}. Выигрыш по основной метрике "
                         "не компенсирует риск — разобраться в причинах до повторного запуска.")
    p = a.primary
    lift = f"{pct(p.final.lift, 1, True)} {ci(*p.final.lift_ci)}"
    if p.significant and p.good:
        return dict(status="ship", title="Раскатывать",
                    text=f"{p.metric.title}: {lift}, {p_eq(p.final.p_value)}. Защитные метрики не ухудшились.")
    if p.significant:
        return dict(status="stop", title="Не раскатывать: основная метрика ухудшилась",
                    text=f"{p.metric.title}: {lift}, {p_eq(p.final.p_value)}.")
    return dict(status="neutral", title="Эффект не обнаружен",
                text=f"{p.metric.title}: {lift}, {p_eq(p.final.p_value)}. При таком объёме тест "
                     f"заметил бы эффект от {pct(p.mde)} (MDE при мощности 80 %). Если меньший эффект "
                     "важен — нужен больший трафик, иначе решение принимается по другим соображениям.")


def analyze(users: pd.DataFrame, daily: pd.DataFrame, info: dict, n_days: int = 21) -> Analysis:
    n = users.variant.value_counts().to_dict()
    n = {"control": int(n.get("control", 0)), "treatment": int(n.get("treatment", 0))}
    srm = stats.srm_test(n["control"], n["treatment"])
    b_c, b_t = _split(users, "pre_revenue")
    balance = stats.mean_test(b_c, b_t)

    primary = info["primary_metric"]
    results = []
    for key, m in METRICS.items():
        role = "primary" if key == primary else "guardrail" if key in GUARDRAILS else "secondary"
        raw, final = test_metric(users, m)
        mde = 2.8 * final.se / abs(final.control) if final.control else float("nan")
        results.append(MetricResult(m, role, raw, final, mde=mde))
    sec = [r for r in results if r.role == "secondary"]
    for r, p in zip(sec, stats.holm([r.final.p_value for r in sec])):
        r.p_adj = float(p)
    results.sort(key=lambda r: ("primary", "secondary", "guardrail").index(r.role))

    a = Analysis(info, n, srm, balance, results,
                 sequential(daily, users, METRICS[primary], n_days),
                 segments(users, METRICS[primary]))
    a.decision = decide(a)
    return a
