"""
HTML-отчёт по эксперименту и страница-список. Один самодостаточный файл без JS и
внешних библиотек: графики — SVG, нарисованный здесь же на CSS-переменных, поэтому
отчёт одинаково читается в светлой и тёмной теме и открывается из любого места.
"""

from __future__ import annotations

import html
import math
from datetime import date

from .analysis import ALPHA, SRM_ALPHA, Analysis, MetricResult
from .fmt import num, p_eq, pct, pval

STATUS = {
    "ship": ("Раскатывать", "good"),
    "stop": ("Не раскатывать", "bad"),
    "invalid": ("Невалидный эксперимент", "bad"),
    "neutral": ("Эффект не обнаружен", "neutral"),
}
ROLE = {"primary": "основная", "secondary": "вторичная", "guardrail": "защитная"}
SEGMENT = {"android": "Android", "ios": "iOS", "web": "Веб"}

CSS = """
:root{--bg:#fff;--text:#1a1a2e;--muted:#6b7280;--card:#f4f5f7;--border:#e5e7eb;--accent:#2563eb;
--good:#15803d;--good-bg:#dcfce7;--bad:#b91c1c;--bad-bg:#fee2e2;--neutral:#4b5563;--neutral-bg:#e5e7eb;
--band:rgba(37,99,235,.14);--band2:rgba(107,114,128,.16)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1117;--text:#e5e7eb;--muted:#9ca3af;
--card:#1a1d27;--border:#2a2e3a;--accent:#60a5fa;--good:#4ade80;--good-bg:rgba(74,222,128,.14);--bad:#f87171;
--bad-bg:rgba(248,113,113,.14);--neutral:#d1d5db;--neutral-bg:rgba(209,213,219,.12);--band:rgba(96,165,250,.18);
--band2:rgba(156,163,175,.18)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 56px}
a{color:var(--accent)} h1{font-size:1.7rem;line-height:1.25;margin:4px 0 6px} h2{font-size:1.15rem;margin:28px 0 10px}
.muted{color:var(--muted)} .small{font-size:.85rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:12px 0}
.decision{border-left:5px solid var(--neutral)} .decision.good{border-left-color:var(--good)} .decision.bad{border-left-color:var(--bad)}
.decision h2{margin:0 0 4px}
.badge{display:inline-block;padding:1px 9px;border-radius:999px;font-size:.8rem;font-weight:600;white-space:nowrap}
.badge.good{background:var(--good-bg);color:var(--good)} .badge.bad{background:var(--bad-bg);color:var(--bad)}
.badge.neutral{background:var(--neutral-bg);color:var(--neutral)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi b{display:block;font-size:1.25rem}
.scroll{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.8rem} th:first-child,td:first-child{text-align:left}
td.l{text-align:left;white-space:normal} td.metric{min-width:240px}
.pos{color:var(--good)} .neg{color:var(--bad)}
svg{width:100%;height:auto;display:block} svg text{fill:var(--text);font-size:12px;font-family:inherit}
svg .m{fill:var(--muted)} .legend{font-size:.82rem;color:var(--muted);margin-top:4px}
.legend i{display:inline-block;width:14px;height:10px;border-radius:2px;margin:0 5px 0 12px;vertical-align:middle}
footer{margin-top:36px;font-size:.82rem;color:var(--muted)}
.chart{overflow-x:auto}
@media (max-width:700px){.chart svg{min-width:720px}}
"""


# --------------------------------------------------------------------------- форматирование
def value(m, x: float) -> str:
    if m.fmt == "pct":
        return pct(x, 2)
    if m.fmt == "rub":
        return num(x, 0) + " ₽"
    return num(x, 2)


def esc(s) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- SVG
def _ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    span = hi - lo or 1
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 12))
        t += step
    return out


def forest_svg(results: list[MetricResult], width: int = 900) -> str:
    """Относительный эффект и 95 %-интервал по каждой метрике."""
    row, left, right, top = 40, 230, 30, 14
    h = top + row * len(results) + 40
    vals = [v for r in results for v in r.final.lift_ci] + [0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    X = lambda v: left + (v - lo) / (hi - lo) * (width - left - right)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="Эффект по метрикам с 95 % интервалами">']
    for t in _ticks(lo, hi):
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - 34}" stroke="var(--border)"/>'
                     f'<text x="{X(t):.1f}" y="{h - 16}" text-anchor="middle" class="m">{pct(t, 0, True)}</text>')
    parts.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top - 4}" y2="{h - 34}" stroke="var(--muted)" stroke-width="1.5"/>')
    for i, r in enumerate(results):
        y = top + row * i + row / 2
        sig = r.significant
        color = "var(--muted)" if not sig else ("var(--good)" if r.good else "var(--bad)")
        a, b = r.final.lift_ci
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end">{esc(r.metric.title)}</text>'
                     f'<line x1="{X(a):.1f}" x2="{X(b):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
                     f'<circle cx="{X(r.final.lift):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def sequential_svg(seq, width: int = 900, height: int = 300) -> str:
    """Накопленный эффект основной метрики по дням: обычный 95 %-интервал и всегда-валидный (mSPRT)."""
    m = dict(l=56, r=16, t=12, b=40)
    days = seq.day.tolist()
    vals = list(seq.lift_lo) + list(seq.lift_hi) + [0]
    av = [v for v in list(seq.av_lift_lo) + list(seq.av_lift_hi) if math.isfinite(v) and abs(v) < 1]
    vals += av
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 0.01
    lo, hi = lo - pad, hi + pad
    X = lambda d: m["l"] + (d - days[0]) / max(days[-1] - days[0], 1) * (width - m["l"] - m["r"])  # noqa: E731
    Y = lambda v: height - m["b"] - (min(max(v, lo), hi) - lo) / (hi - lo) * (height - m["t"] - m["b"])  # noqa: E731

    def band(a, b, fill):
        pts = [f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(days, a)]
        pts += [f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(reversed(days), reversed(list(b)))]
        return f'<polygon points="{" ".join(pts)}" fill="{fill}"/>'

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Накопленный эффект по дням">']
    for t in _ticks(lo, hi, 6):
        parts.append(f'<line x1="{m["l"]}" x2="{width - m["r"]}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--border)"/>'
                     f'<text x="{m["l"] - 8}" y="{Y(t) + 4:.1f}" text-anchor="end" class="m">{pct(t, 0, True)}</text>')
    for d in days:
        if d == 1 or d % 7 == 0:
            parts.append(f'<text x="{X(d):.1f}" y="{height - m["b"] + 18}" text-anchor="middle" class="m">{d}</text>')
    parts.append(f'<text x="{(m["l"] + width - m["r"]) / 2}" y="{height - 4}" text-anchor="middle" class="m">день эксперимента</text>')
    av_lo = [max(v, lo) for v in seq.av_lift_lo]
    av_hi = [min(v, hi) for v in seq.av_lift_hi]
    parts.append(band(av_lo, av_hi, "var(--band2)"))
    parts.append(band(seq.lift_lo, seq.lift_hi, "var(--band)"))
    parts.append(f'<line x1="{m["l"]}" x2="{width - m["r"]}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--muted)" stroke-width="1.2"/>')
    line = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in zip(days, seq.lift))
    parts.append(f'<polyline points="{line}" fill="none" stroke="var(--accent)" stroke-width="2.4"/>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- страницы
def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>'
            f'<style>{CSS}</style></head><body><main>{body}</main></body></html>')


def _metric_rows(a: Analysis) -> str:
    out = []
    for r in a.results:
        f, m = r.final, r.metric
        cls = "" if not r.significant else ("pos" if r.good else "neg")
        if r.significant:
            verdict = ("good", "лучше") if r.good else ("bad", "хуже")
        else:
            verdict = ("neutral", "нет разницы")
        p_cell = pval(f.p_value) + (f'<br><span class="muted small">Холм: {pval(r.p_adj)}</span>' if r.p_adj is not None else "")
        vr = f.extra.get("variance_reduction")
        out.append(
            f'<tr><td class="l metric"><b>{esc(m.title)}</b> <span class="badge neutral">{ROLE[r.role]}</span>'
            f'<br><span class="muted small">{esc(m.description)}</span></td><td>{value(m, r.raw.control)}</td><td>{value(m, r.raw.treatment)}</td>'
            f'<td class="{cls}"><b>{pct(f.lift, 2, True)}</b><br><span class="small">'
            f'[{pct(f.lift_ci[0], 2, True)}; {pct(f.lift_ci[1], 2, True)}]</span></td>'
            f'<td>{p_cell}</td><td>{pct(vr, 0) if vr is not None else "—"}</td>'
            f'<td>{pct(r.mde, 1)}</td><td><span class="badge {verdict[0]}">{verdict[1]}</span></td></tr>')
    return "".join(out)


def render(a: Analysis) -> str:
    info, d = a.info, a.decision
    label, cls = STATUS[d["status"]]
    chi2, srm_p = a.srm
    srm_ok = srm_p >= SRM_ALPHA
    total = a.n["control"] + a.n["treatment"]
    bal = a.balance
    seq = a.sequential
    first_fixed = seq.day[seq.p_fixed < ALPHA].min()
    first_msprt = seq.day[seq.p_msprt < ALPHA].min()
    seq_note = (
        f"Если бы решение принимали, глядя на результат каждый день, обычный тест впервые показал бы p < 0,05 на "
        f"{int(first_fixed)}-й день. " if first_fixed == first_fixed else
        "Обычный тест ни в один из дней не показал p < 0,05. ")
    seq_note += (f"Всегда-валидный mSPRT, рассчитанный на ежедневные взгляды, — на {int(first_msprt)}-й день."
                 if first_msprt == first_msprt else
                 f"Всегда-валидный mSPRT значимости не достиг ({p_eq(seq.p_msprt.iloc[-1])} на последний день).")
    seg_rows = "".join(
        f'<tr><td>{esc(SEGMENT.get(s.segment, s.segment))}</td><td>{num(s.users)}</td><td class="{"pos" if s.lo > 0 else "neg" if s.hi < 0 else ""}">'
        f'{pct(s.lift, 2, True)}</td><td>[{pct(s.lo, 2, True)}; {pct(s.hi, 2, True)}]</td><td>{pval(s.p)}</td></tr>'
        for s in a.segments.itertuples())
    p = a.primary
    body = f"""
<p class="small"><a href="../index.html">← все эксперименты</a></p>
<p class="muted small">{esc(info['experiment_id'])} · {info['start_date']:%d.%m.%Y} — {info['end_date']:%d.%m.%Y}</p>
<h1>{esc(info['title'])}</h1>
<p class="muted">Гипотеза: {esc(info.get('hypothesis') or '—')}</p>

<div class="card decision {cls}"><h2>{esc(d['title'])}</h2><div>{esc(d['text'])}</div></div>

<div class="grid">
 <div class="card kpi"><span class="muted small">Пользователей</span><b>{num(total)}</b>
  <span class="small">control {num(a.n['control'])} · treatment {num(a.n['treatment'])}</span></div>
 <div class="card kpi"><span class="muted small">{esc(p.metric.title)} (основная)</span><b class="{'pos' if p.significant and p.good else 'neg' if p.significant else ''}">{pct(p.final.lift, 2, True)}</b>
  <span class="small">95 % ДИ [{pct(p.final.lift_ci[0], 2, True)}; {pct(p.final.lift_ci[1], 2, True)}], {p_eq(p.final.p_value)}</span></div>
 <div class="card kpi"><span class="muted small">Сплит 50/50 (SRM)</span><b class="{'' if srm_ok else 'neg'}">{'в порядке' if srm_ok else 'нарушен'}</b>
  <span class="small">χ² = {num(chi2, 1)}, {p_eq(srm_p)}</span></div>
 <div class="card kpi"><span class="muted small">Баланс до старта (выручка)</span><b class="{'' if bal.p_value >= 0.01 else 'neg'}">{'в порядке' if bal.p_value >= 0.01 else 'перекос'}</b>
  <span class="small">разница {pct(bal.lift, 1, True)}, {p_eq(bal.p_value)}</span></div>
</div>

<h2>Метрики</h2>
<div class="card scroll"><table>
<thead><tr><th>Метрика</th><th>Control</th><th>Treatment</th><th>Изменение, 95 % ДИ</th><th>p-value</th>
<th>CUPED:<br>−дисперсия</th><th>MDE</th><th>Итог</th></tr></thead>
<tbody>{_metric_rows(a)}</tbody></table>
<p class="muted small">Изменение и интервал — с CUPED (ковариата — та же метрика за 28 дней до старта); для отношений —
линеаризация + CUPED, без снижения дисперсии — дельта-метод. Вторичные метрики — с поправкой Холма; защитные проверяются без
поправки, чтобы не пропустить вред. MDE — минимальный эффект, который тест заметил бы с мощностью 80 %.</p></div>
<div class="card"><div class="chart">{forest_svg(a.results)}</div>
<div class="legend"><i style="background:var(--good)"></i>лучше <i style="background:var(--bad)"></i>хуже
<i style="background:var(--muted)"></i>разница не значима</div></div>

<h2>{esc(p.metric.title)} по дням</h2>
<div class="card"><div class="chart">{sequential_svg(seq)}</div>
<div class="legend"><i style="background:var(--accent)"></i>накопленный эффект
<i style="background:var(--band)"></i>95 % ДИ на этот день <i style="background:var(--band2)"></i>всегда-валидный ДИ (mSPRT)</div>
<p class="small">{seq_note} Итог принимается по плановой длительности ({len(seq)} дней); узкий синий интервал честен только
для одного взгляда в конце. Серый — интервал, которым можно пользоваться при ежедневном мониторинге.</p></div>

<h2>Срезы по платформам</h2>
<div class="card scroll"><table><thead><tr><th>Платформа</th><th>Пользователей</th><th>{esc(p.metric.title)}</th>
<th>95 % ДИ</th><th>p-value</th></tr></thead><tbody>{seg_rows}</tbody></table>
<p class="muted small">Разведочный анализ: без поправок на множественность, решения по срезам не принимаются — только гипотезы
для следующих экспериментов.</p></div>

<footer>Отчёт собран {date.today():%d.%m.%Y} движком <a href="https://github.com/Alex91132/ab-testing-engine">abengine</a>.
Данные синтетические.</footer>
"""
    return page(f"{info['title']} — отчёт A/B", body)


def render_index(analyses: list[Analysis]) -> str:
    rows = []
    for a in analyses:
        label, cls = STATUS[a.decision["status"]]
        p = a.primary
        rows.append(
            f'<tr><td class="l"><a href="reports/{esc(a.info["experiment_id"])}.html"><b>{esc(a.info["title"])}</b></a>'
            f'<br><span class="muted small">{esc(a.info["experiment_id"])}</span></td>'
            f'<td>{num(a.n["control"] + a.n["treatment"])}</td><td class="l">{esc(p.metric.title)}</td>'
            f'<td>{pct(p.final.lift, 2, True)}</td><td>{pval(p.final.p_value)}</td>'
            f'<td><span class="badge {cls}">{label}</span></td></tr>')
    body = f"""
<h1>Эксперименты</h1>
<p class="muted">Отчёты собраны движком <a href="https://github.com/Alex91132/ab-testing-engine">abengine</a>
по синтетическим логам интернет-магазина: 150 000 пользователей, 4 недели истории и 3 недели экспериментов.
Четыре эксперимента идут одновременно с независимым хешированием пользователей.</p>
<div class="card scroll"><table><thead><tr><th>Эксперимент</th><th>Пользователей</th><th>Основная метрика</th>
<th>Эффект</th><th>p-value</th><th>Решение</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="small">Что внутри отчёта: проверка сплита (SRM) и баланса групп до старта, CUPED, дельта-метод для метрик-отношений,
поправка Холма, последовательный анализ (mSPRT), срезы. Методы проверены на 3 000 симулированных экспериментов —
<a href="https://github.com/Alex91132/ab-testing-engine#проверка-методов">результаты в README</a>.</p>
<footer>Александр Федоров · <a href="https://alex91132.github.io/">портфолио</a></footer>
"""
    return page("Отчёты A/B-экспериментов", body)
