import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""구성요소 영향도: 하나씩 끄고 결과가 얼마나 달라지는지 잰다.

실행: PYTHONPATH=. python scripts/ablate.py
'●'는 짝지은 95% 신뢰구간을 넘은 것(= 진짜 효과), '○'는 몬테카를로 오차 안(= 빼도 되는 복잡도).
모델을 키울 때마다 돌려서, 값을 못 하는 복잡도가 쌓이지 않게 한다.
"""
from dataclasses import replace

from retire_sim import engine_tax, metrics, mortality
from retire_sim.config import CareMarkov, CareShock, Household, NPSRules, Person, SimConfig
from retire_sim.economy_v2 import EconomyV2
from retire_sim.pension import estimate_nps_monthly
from retire_sim.tax import TaxConfig

ECO, TC = EconomyV2.enhanced(), TaxConfig(overseas_share=0.2)
BASE_CFG = SimConfig(care=CareMarkov())
QX = mortality.default_qx()


def couple_60():
    return Household(members=[Person(60, "M", nps_monthly=110, private_pension_annual=600, private_pension_start=60),
                              Person(58, "F", nps_monthly=50)],
                     liquid_assets=50_000, stock_weight=0.4, annual_spending=3600,
                     essential=3000, lifestyle=600, legacy_target=10_000)


def couple_30():
    n1 = estimate_nps_monthly(5000, 3, 30, 60); n2 = estimate_nps_monthly(4000, 2, 29, 59)
    return Household(members=[Person(30, "M", nps_monthly=n1, salary=5000), Person(29, "F", nps_monthly=n2, salary=4000)],
                     liquid_assets=5000, stock_weight=0.5, annual_spending=3600, essential=3600, lifestyle=1200,
                     legacy_target=10_000, retire_age=60, saving_rate=0.15, pension_balance=1000, retirement_contrib=True)


# (이름, 경제 덮어쓰기, 설정 덮어쓰기, 세금 덮어쓰기)
ABL = [
    ("경제: 폭락 점프",        dict(jump_lambda=0.0), {}, {}),
    ("경제: 파라미터 불확실성",  dict(erp_sd=0.0, pi_theta_sd=0.0, q_theta_sd=0.0), {}, {}),
    ("경제: 기간프리미엄",      dict(term_premium=0.0), {}, {}),
    ("경제: 금리-주식 상관",    dict(rho_rs=0.0), {}, {}),
    ("경제: 운용보수 0.5%",    dict(fee=0.0), {}, {}),
    ("경제: 피셔 연결",        dict(fisher=False), {}, {}),
    ("수명: 코호트 개선",      {}, dict(mortality_improvement=False), {}),
    ("수명: 장수 불확실성",     {}, dict(longevity_sigma_level=0.0, longevity_sigma_trend=0.0), {}),
    ("수명: 사별 효과",        {}, dict(widow_excess=(0.0, 0.0)), {}),
    ("간병: 비용 인플레이션",    {}, dict(care_real_growth=0.0), {}),
    ("간병: 2단계 마르코프",    {}, dict(care=CareShock()), {}),
    ("제도: 유족연금",         {}, dict(nps=NPSRules(survivor_on=False)), {}),
    ("제도: 기초연금",         {}, dict(nps=NPSRules(basic_on=False)), {}),
    ("세금: 전체",            {}, {}, dict(enabled=False)),
]


def run(hh, eco=None, cfg=None, tc=None):
    return engine_tax.run(hh, cfg or BASE_CFG, e=eco or ECO, tc=tc or TC, qx_table=QX)


for name, fn in (("60대 부부", couple_60), ("30대 부부", couple_30)):
    hh = fn(); base = run(hh)
    b = engine_tax.summarize(base)["기본생활 유지 확률"]
    print(f"\n== {name}  기준 {b * 100:.1f}%")
    rows = []
    for lab, ekw, ckw, tkw in ABL:
        eco = replace(ECO, **ekw) if ekw else None
        if ckw:
            kw = dict(ckw)
            cfg = SimConfig(care=kw.pop("care", CareMarkov()), **kw)
        else:
            cfg = None
        tc = replace(TC, **tkw) if tkw else None
        r = run(hh, eco, cfg, tc)
        s = engine_tax.summarize(r); d = metrics.compare(r, base)
        rows.append((abs(d["차이"]), lab, s["기본생활 유지 확률"] - b, d["±"], s["CTE90 부족액(실질)"]))
    for _, lab, dd, ci, cte in sorted(rows, reverse=True):
        mark = "●" if abs(dd) > ci else "○"
        print(f"  {mark} {lab:22s} {dd * 100:+6.2f}%p (±{ci * 100:4.2f})  CTE90 {cte:,.0f}")
