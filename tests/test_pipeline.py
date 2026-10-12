import duckdb
import numpy as np
import pandas as pd
import pytest

from abengine.analysis import analyze
from abengine.dataset import EXP_START, generate, write_duckdb
from abengine.metrics import daily_metrics, experiment_ids, experiment_info, user_metrics
from abengine.simulate import assign


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tables, truth = generate(n_users=80_000, seed=3)   # меньше — потеря 3 % treatment не дотягивает до порога SRM
    path = tmp_path_factory.mktemp("db") / "shop.duckdb"
    write_duckdb(tables, path)
    con = duckdb.connect(str(path), read_only=True)
    yield con, tables
    con.close()


def test_assignment_deterministic_balanced_independent():
    ids = np.arange(1, 50_001)
    a1, a2 = assign(ids, "salt-a"), assign(ids, "salt-a")
    b = assign(ids, "salt-b")
    assert np.array_equal(a1, a2)
    assert abs(a1.mean() - 0.5) < 0.01
    assert abs(np.corrcoef(a1, b)[0, 1]) < 0.02       # разные соли — независимые сплиты
    part = assign(ids, "salt-c", traffic=0.2)
    assert abs((part >= 0).mean() - 0.2) < 0.01


def test_sql_metrics_match_pandas(db):
    """USER_METRICS_SQL против независимого пересчёта по сырым таблицам."""
    con, t = db
    um = user_metrics(con, "checkout_redesign").set_index("user_id")
    a = t["assignments"].query("experiment_id == 'checkout_redesign'").set_index("user_id")
    o = t["orders"].join(a.first_exposure, on="user_id", how="inner")
    o = o[(o.order_date >= o.first_exposure) & (o.status == "completed")]
    rev = o.groupby("user_id").revenue.sum().reindex(um.index, fill_value=0)
    assert np.allclose(um.revenue, rev, atol=0.01)
    pre = t["orders"][(t["orders"].order_date < EXP_START) & (t["orders"].status == "completed")]
    pre_rev = pre.groupby("user_id").revenue.sum().reindex(um.index, fill_value=0)
    assert np.allclose(um.pre_revenue, pre_rev, atol=0.01)
    d = t["daily_activity"].join(a.first_exposure, on="user_id", how="inner")
    views = d[d.date >= d.first_exposure].groupby("user_id").views.sum().reindex(um.index, fill_value=0)
    assert np.array_equal(um.views.to_numpy(), views.to_numpy())


def test_daily_sums_add_up_to_user_totals(db):
    con, _ = db
    um = user_metrics(con, "recommendations_v2").set_index("user_id")
    dm = daily_metrics(con, "recommendations_v2").groupby("user_id").sum()
    for col in ("views", "clicks", "orders", "cancelled"):
        assert np.array_equal(dm[col].reindex(um.index, fill_value=0).to_numpy(), um[col].to_numpy())
    assert np.allclose(dm.revenue.reindex(um.index, fill_value=0), um.revenue)


def test_broken_experiment_is_flagged_and_aa_is_not(db):
    con, _ = db
    results = {}
    for exp in experiment_ids(con):
        a = analyze(user_metrics(con, exp), daily_metrics(con, exp), experiment_info(con, exp))
        results[exp] = a
    assert results["promo_banner"].decision["status"] == "invalid"
    assert results["aa_platform_check"].srm[1] > 0.001
    assert results["aa_platform_check"].decision["status"] != "invalid"
    primary = results["checkout_redesign"].primary
    assert primary.metric.key == "conversion" and primary.role == "primary"
    seq = results["checkout_redesign"].sequential
    assert len(seq) == 21 and seq.lift.iloc[-1] == pytest.approx(primary.final.lift)
