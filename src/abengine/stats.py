"""
Статистика A/B-тестов. Единица анализа — пользователь (по нему шло назначение):
все метрики сводятся к пользовательским суммам, наблюдения разных пользователей независимы.

* mean_test        — Welch t-тест для пользовательских метрик (конверсия, выручка на пользователя);
* cuped            — снижение дисперсии ковариатой из периода до эксперимента (Deng et al., 2013);
* ratio_test       — метрика-отношение Σчислитель/Σзнаменатель (средний чек, CTR) дельта-методом;
* naive_unit_test  — t-тест по заказам/просмотрам как по независимым — НЕВЕРНО, для сравнения;
* srm_test         — проверка соотношения групп (Sample Ratio Mismatch), χ²;
* holm, benjamini_hochberg — поправки на множественные сравнения;
* obf_boundaries   — групповой последовательный дизайн О'Брайена — Флеминга (запланированные взгляды);
* msprt            — последовательный тест со «всегда валидными» p-value (Johari et al., 2017).

Относительный эффект lift = Δ / уровень control; его интервал — интервал Δ, делённый на
уровень control. Так p-value и интервал для lift всегда согласованы (интервал не накрывает 0
ровно тогда, когда p < α); неопределённость самого уровня control при малых эффектах
пренебрежимо мала.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as st


@dataclass
class TestResult:
    method: str
    control: float            # оценка метрики в control (среднее или отношение)
    treatment: float
    diff: float               # treatment − control
    se: float
    ci: tuple[float, float]   # интервал для diff
    lift: float               # treatment / control − 1
    lift_ci: tuple[float, float]
    p_value: float
    n: tuple[int, int]        # пользователей в control, treatment
    extra: dict = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _z(alpha: float) -> float:
    return float(st.norm.ppf(1 - alpha / 2))


def _result(method, m_c, m_t, var_mc, var_mt, n_c, n_t, alpha, df=None, extra=None) -> TestResult:
    diff = m_t - m_c
    se = float(np.sqrt(var_mc + var_mt))
    if df is None:
        p = 2 * st.norm.sf(abs(diff) / se) if se > 0 else 1.0
        q = _z(alpha)
    else:
        p = 2 * st.t.sf(abs(diff) / se, df) if se > 0 else 1.0
        q = float(st.t.ppf(1 - alpha / 2, df))
    lo, hi = diff - q * se, diff + q * se
    return TestResult(method, float(m_c), float(m_t), float(diff), se, (lo, hi),
                      float(diff / m_c), (lo / m_c, hi / m_c), float(p), (int(n_c), int(n_t)), extra or {})


def mean_test(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05, method: str = "Welch t") -> TestResult:
    """Welch t-тест разности средних (дисперсии групп могут различаться)."""
    n_c, n_t = len(y_c), len(y_t)
    v_c, v_t = y_c.var(ddof=1) / n_c, y_t.var(ddof=1) / n_t
    df = (v_c + v_t) ** 2 / (v_c ** 2 / (n_c - 1) + v_t ** 2 / (n_t - 1)) if v_c + v_t > 0 else n_c + n_t - 2
    return _result(method, y_c.mean(), y_t.mean(), v_c, v_t, n_c, n_t, alpha, df=df)


def cuped(y_c, y_t, x_c, x_t, alpha: float = 0.05) -> TestResult:
    """
    CUPED: Ŷ = Y − θ(X − X̄), θ = cov(Y, X) / var(X) по обеим группам вместе.
    X — та же метрика до эксперимента: от назначения она не зависит, поэтому оценка
    эффекта остаётся несмещённой, а дисперсия падает в (1 − ρ²) раз.
    """
    y = np.concatenate([y_c, y_t])
    x = np.concatenate([x_c, x_t])
    vx = x.var(ddof=1)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    xm = x.mean()
    a_c, a_t = y_c - theta * (x_c - xm), y_t - theta * (x_t - xm)
    res = mean_test(a_c, a_t, alpha, method="CUPED")
    raw = mean_test(y_c, y_t, alpha)
    pooled_raw = np.concatenate([y_c - y_c.mean(), y_t - y_t.mean()]).var()
    pooled_adj = np.concatenate([a_c - a_c.mean(), a_t - a_t.mean()]).var()
    res.extra.update(theta=theta, variance_reduction=1 - pooled_adj / pooled_raw if pooled_raw > 0 else 0.0,
                     raw=raw)
    return res


def ratio_moments(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Отношение Σnum/Σden и дисперсия его оценки по дельта-методу (единица — пользователь)."""
    n = len(num)
    mn, md = num.mean(), den.mean()
    vn, vd = num.var(ddof=1), den.var(ddof=1)
    cov = np.cov(num, den, ddof=1)[0, 1]
    r = mn / md
    var_r = (vn / md ** 2 - 2 * mn * cov / md ** 3 + mn ** 2 * vd / md ** 4) / n
    return float(r), float(var_r)


def ratio_test(num_c, den_c, num_t, den_t, alpha: float = 0.05) -> TestResult:
    """
    Метрика-отношение (средний чек = выручка / заказы, CTR = клики / просмотры).
    Заказы одного пользователя зависимы, поэтому дисперсию считаем по пользователям:
    R = Σx/Σy, Var(R) ≈ (σx²/μy² − 2μx·σxy/μy³ + μx²σy²/μy⁴) / n.
    """
    r_c, v_c = ratio_moments(num_c, den_c)
    r_t, v_t = ratio_moments(num_t, den_t)
    return _result("Дельта-метод", r_c, r_t, v_c, v_t, len(num_c), len(num_t), alpha)


def linearize(num: np.ndarray, den: np.ndarray, r_c: float, den_mean_c: float) -> np.ndarray:
    """
    Линеаризация отношения (Budylin et al., 2018): L = (num − R_c·den) / mean(den_c).
    Среднее L в группе — приближение (R − R_c), и к L применимо всё, что работает
    для обычных средних, включая CUPED.
    """
    return (num - r_c * den) / den_mean_c


def ratio_cuped(num_c, den_c, num_t, den_t, pre_num_c, pre_den_c, pre_num_t, pre_den_t, alpha: float = 0.05):
    """CUPED для отношения: линеаризуем метрику и её значение до эксперимента, затем CUPED."""
    r_c = num_c.sum() / den_c.sum()
    dm = den_c.mean()
    pr_c = pre_num_c.sum() / max(pre_den_c.sum(), 1e-12)
    pdm = max(pre_den_c.mean(), 1e-12)
    res = cuped(linearize(num_c, den_c, r_c, dm), linearize(num_t, den_t, r_c, dm),
                linearize(pre_num_c, pre_den_c, pr_c, pdm), linearize(pre_num_t, pre_den_t, pr_c, pdm), alpha)
    # среднее L — приближение разности отношений: переводим результат в масштаб R
    lo, hi = res.ci
    return TestResult("CUPED + линеаризация", float(r_c), float(r_c + res.diff), res.diff, res.se,
                      (lo, hi), float(res.diff / r_c), (lo / r_c, hi / r_c), res.p_value, res.n, res.extra)


def naive_unit_test(values_c: np.ndarray, values_t: np.ndarray, alpha: float = 0.05) -> TestResult:
    """t-тест по заказам (или просмотрам), как будто они независимы. Занижает дисперсию — только для сравнения."""
    return mean_test(values_c, values_t, alpha, method="t-тест по заказам (неверно)")


def srm_test(n_control: int, n_treatment: int, expected_share: float = 0.5) -> tuple[float, float]:
    """χ²-тест соотношения групп. p < 0.001 — эксперимент сломан, результаты не интерпретировать."""
    total = n_control + n_treatment
    exp = np.array([total * (1 - expected_share), total * expected_share])
    chi2, p = st.chisquare([n_control, n_treatment], exp)
    return float(chi2), float(p)


def holm(p_values) -> np.ndarray:
    """Поправка Холма (FWER): шаговая версия Бонферрони, равномерно мощнее неё."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(p_values) -> np.ndarray:
    """Поправка Бенджамини — Хохберга (FDR)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for i, idx in enumerate(order):
        rank = m - i
        running = min(running, p[idx] * m / rank)
        adj[idx] = running
    return adj


def msprt(diffs: np.ndarray, ses: np.ndarray, tau: float, alpha: float = 0.05):
    """
    Последовательный тест mSPRT с нормальной смесью (Johari, Pekelis, Walsh, 2017).
    На каждом «подглядывании» k по оценке разности θ̂ₖ и её дисперсии sₖ²:
        Λₖ = √(sₖ²/(sₖ²+τ²)) · exp(τ²θ̂ₖ² / (2sₖ²(sₖ²+τ²))),
    всегда-валидное p-value pₖ = min(1, minⱼ≤ₖ 1/Λⱼ). Остановка при pₖ < α сохраняет
    ошибку I рода ≤ α при любом числе подглядываний. τ — масштаб ожидаемого эффекта.
    Возвращает (p-value, нижняя и верхняя границы всегда-валидного интервала) по шагам.
    """
    d = np.asarray(diffs, dtype=float)
    s2 = np.asarray(ses, dtype=float) ** 2
    t2 = tau ** 2
    log_lam = 0.5 * np.log(s2 / (s2 + t2)) + t2 * d ** 2 / (2 * s2 * (s2 + t2))
    p = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_lam)))
    half = np.sqrt(s2 * (s2 + t2) / t2 * (2 * np.log(1 / alpha) + np.log((s2 + t2) / s2)))
    lo = np.maximum.accumulate(d - half)
    hi = np.minimum.accumulate(d + half)
    return p, lo, hi


def obf_boundaries(fractions, alpha: float = 0.05, n_sim: int = 400_000, seed: int = 20260912) -> np.ndarray:
    """
    Границы О'Брайена — Флеминга для заранее запланированных взглядов: |Zₖ| > c / √tₖ,
    tₖ — доля набранной информации (выборки) на k-м взгляде. Константа c подбирается так,
    чтобы под H0 вероятность пересечь хоть одну границу была ровно α. Траектория
    Z-статистик — броуновское движение: corr(Zᵢ, Zⱼ) = √(tᵢ/tⱼ); вероятность считается
    Монте-Карло с фиксированным зерном. Ранние границы очень строгие, последняя — почти 1,96.
    """
    t = np.asarray(fractions, dtype=float)
    if not (np.all(np.diff(t) > 0) and 0 < t[0] and t[-1] == 1):
        raise ValueError(f"доли информации должны строго возрастать до 1: {t}")
    rng = np.random.default_rng(seed)
    steps = np.diff(np.concatenate([[0.0], t]))
    w = np.cumsum(rng.standard_normal((n_sim, len(t))) * np.sqrt(steps), axis=1)
    z = np.abs(w / np.sqrt(t))
    scaled = (z * np.sqrt(t)).max(axis=1)            # пересечение ⇔ max_k |Zₖ|·√tₖ > c
    c = float(np.quantile(scaled, 1 - alpha))
    return c / np.sqrt(t)
