import numpy as np
import pytest
from scipy import stats as st

from abengine import stats


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


def test_welch_matches_scipy(rng):
    a, b = rng.lognormal(3, 1, 3000), rng.lognormal(3.05, 1.2, 2500)
    r = stats.mean_test(a, b)
    ref = st.ttest_ind(b, a, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert r.diff == pytest.approx(b.mean() - a.mean())


def test_lift_ci_consistent_with_p_value(rng):
    """Интервал для lift не накрывает 0 ровно тогда, когда p < α."""
    for _ in range(50):
        a, b = rng.normal(10, 3, 800), rng.normal(10 + rng.normal(0, 0.3), 3, 800)
        r = stats.mean_test(a, b)
        assert (r.lift_ci[0] > 0 or r.lift_ci[1] < 0) == (r.p_value < 0.05)


def test_cuped_matches_regression_and_reduces_variance(rng):
    n = 20_000
    x = rng.gamma(2, 50, n)
    treat = rng.integers(0, 2, n).astype(bool)
    y = 0.8 * x + rng.normal(0, 40, n) + 5 * treat
    r = stats.cuped(y[~treat], y[treat], x[~treat], x[treat])
    # оценка эффекта ≈ коэффициент при treat в регрессии y ~ treat + x
    design = np.column_stack([np.ones(n), treat, x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert r.diff == pytest.approx(beta[1], abs=0.05)
    rho = np.corrcoef(x, y)[0, 1]
    assert r.extra["variance_reduction"] == pytest.approx(rho ** 2, abs=0.02)
    assert r.se < r.extra["raw"].se


def test_delta_method_matches_bootstrap(rng):
    n = 4000
    den = rng.poisson(5, n) + 1
    num = rng.binomial(den, rng.beta(2, 18, n))
    _, var_r = stats.ratio_moments(num.astype(float), den.astype(float))
    boot = []
    for _ in range(1500):
        i = rng.integers(0, n, n)
        boot.append(num[i].sum() / den[i].sum())
    assert np.sqrt(var_r) == pytest.approx(np.std(boot), rel=0.08)


def test_linearization_agrees_with_delta(rng):
    n = 30_000
    den_c, den_t = rng.poisson(6, n) + 1, rng.poisson(6, n) + 1
    num_c = rng.binomial(den_c, 0.10).astype(float)
    num_t = rng.binomial(den_t, 0.106).astype(float)
    delta = stats.ratio_test(num_c, den_c.astype(float), num_t, den_t.astype(float))
    r_c, dm = num_c.sum() / den_c.sum(), den_c.mean()
    lin = stats.mean_test(stats.linearize(num_c, den_c, r_c, dm), stats.linearize(num_t, den_t, r_c, dm))
    assert lin.diff == pytest.approx(delta.diff, rel=0.03)
    assert lin.se == pytest.approx(delta.se, rel=0.03)


def test_srm():
    assert stats.srm_test(50_000, 50_100)[1] > 0.5
    assert stats.srm_test(50_000, 51_500)[1] < 1e-5


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(stats.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(stats.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_msprt_properties():
    diffs = np.array([0.0, 0.5, 1.0, 1.2, 0.2])
    ses = np.array([1.0, 0.7, 0.5, 0.4, 0.35])
    p, lo, hi = stats.msprt(diffs, ses, tau=1.0)
    assert np.all(np.diff(p) <= 1e-12)                 # p-value не растёт
    assert p[0] == 1.0
    assert np.all(np.diff(lo) >= 0) and np.all(np.diff(hi) <= 0)   # интервал только сужается
    assert np.all((lo <= hi))


def test_obf_matches_published_boundaries():
    """Jennison & Turnbull, табл. 2.3: K = 3, α = 0,05 → 3,471 / 2,454 / 2,004."""
    b = stats.obf_boundaries([1 / 3, 2 / 3, 1])
    assert np.allclose(b, [3.471, 2.454, 2.004], atol=0.01)
    with pytest.raises(ValueError):
        stats.obf_boundaries([0.5, 0.4, 1])
