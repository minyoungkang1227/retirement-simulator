import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""README·docs 의 예시 수치를 현재 모델로 다시 계산한다.

실행: PYTHONPATH=. python scripts/readme_numbers.py
웹 앱과 같은 설정(EconomyV2.enhanced, CareMarkov, 해외주식 20%, 공통 난수)을 쓴다.
모델을 고칠 때마다 이 스크립트를 돌려 README 표를 갱신하면 숫자가 낡지 않는다.
"""
from dataclasses import replace

import numpy as np

from retire_sim import engine_tax, metrics, mortality, __version__
from retire_sim.config import Person, Household, SimConfig, CareMarkov
from retire_sim.economy_v2 import EconomyV2
from retire_sim.pension import estimate_nps_monthly
from retire_sim.tax import TaxConfig, HouseConfig

ECO, TC = EconomyV2.enhanced(), TaxConfig(overseas_share=0.2)
CFG = SimConfig(n_paths=10_000, seed=42, care=CareMarkov())
QX = mortality.default_qx()


def run(hh, eco=None, cfg=None, house=None, qx=None):
    return engine_tax.run(hh, cfg or CFG, e=eco or ECO, tc=TC, house=house or HouseConfig(), qx_table=qx or QX)


def show(res):
    s = engine_tax.summarize(res)
    return s["기본생활 유지 확률"], s.get("여행·취미 평균 충족률", 0), s.get("유산 목표 달성 확률", 0)


def diff(res, base):
    c = metrics.compare(res, base)            # 고갈확률 차이 → 유지 확률은 부호 반대
    return -c["차이"], c["±"]


def couple_52():
    """52·50세 부부, 60세 은퇴, 금융자산 5억, 월 저축 150만 원."""
    return Household(members=[Person(52, "M", nps_monthly=110, nps_start_age=65, nps_years=25),
                              Person(50, "F", nps_monthly=50, nps_start_age=65, nps_years=20)],
                     liquid_assets=50_000, stock_weight=0.5, annual_spending=4800,
                     essential=3600, lifestyle=1200, legacy_target=10_000,
                     retire_age=60, annual_saving=1800)


def couple_30():
    n1 = estimate_nps_monthly(5000, 3, 30, 60); n2 = estimate_nps_monthly(4000, 2, 29, 59)
    return Household(members=[Person(30, "M", nps_monthly=n1, salary=5000, nps_years=33),
                              Person(29, "F", nps_monthly=n2, salary=4000, nps_years=32)],
                     liquid_assets=5000, stock_weight=0.5, annual_spending=4800,
                     essential=3600, lifestyle=1200, legacy_target=10_000,
                     retire_age=60, saving_rate=0.15, pension_balance=1000, retirement_contrib=True)


def couple_65():
    return Household(members=[Person(65, "M", nps_monthly=110, nps_start_age=65, nps_years=25),
                              Person(63, "F", nps_monthly=50, nps_start_age=65, nps_years=20)],
                     liquid_assets=30_000, stock_weight=0.5, annual_spending=4200,
                     essential=3000, lifestyle=1200, legacy_target=30_000)


print(f"모델 v{__version__} · 경로 {CFG.n_paths:,}개 · 시드 {CFG.seed}\n")

# ── A. 52·50세 부부 ──
hh = couple_52(); base = run(hh); b = show(base)
print("【A】 52·50세 부부, 60세 은퇴, 금융자산 5억, 월 저축 150만 원")
print(f"  기본생활 유지 {b[0]*100:.0f}% · 여행·취미 충족 {b[1]*100:.0f}% · 1억 남기기 {b[2]*100:.0f}%")
print("  무엇을 바꾸면:")
for lab, h2 in [("은퇴 2년 늦추기", replace(hh, retire_age=62)),
                ("생활비 10% 줄이기", replace(hh, essential=3240, annual_spending=4440)),
                ("투자 성향 성장형(70%)", replace(hh, stock_weight=0.7)),
                ("투자 성향 보수형(30%)", replace(hh, stock_weight=0.3)),
                ("월 저축 50만 원 늘리기", replace(hh, annual_saving=2400))]:
    d, ci = diff(run(h2), base)
    print(f"    {lab:22s} {d*100:+6.1f}%p (±{ci*100:.1f})")
print("  위기 상황:")
ra = hh.retire_age - hh.members[0].age
for lab, kw in [("은퇴 첫해 주가 −40%", dict(stock_shocks={ra: -0.40})),
                ("은퇴 10년 뒤 주가 −40%", dict(stock_shocks={ra + 10: -0.40})),
                ("3년 연속 물가 6%", dict(infl_shocks={0: .06, 1: .06, 2: .06}))]:
    d, ci = diff(run(hh, eco=replace(ECO, **kw)), base)
    print(f"    {lab:22s} {d*100:+6.1f}%p")
qlow = {k: np.minimum(v * 0.8, 1.0) for k, v in QX.items()}
for k in qlow: qlow[k][-1] = 1.0
d, _ = diff(run(hh, qx=qlow), base); print(f"    {'예상보다 오래 삶(사망률 −20%)':22s} {d*100:+6.1f}%p")
c2 = CareMarkov(incidence=tuple((a, x * 2) for a, x in CareMarkov().incidence))   # 앱의 '간병 위험 2배'와 같은 방식
d, _ = diff(run(hh, cfg=replace(CFG, care=c2)), base); print(f"    {'간병 위험 2배':22s} {d*100:+6.1f}%p")

# ── B. 30·29세 맞벌이 ──
hh = couple_30(); base = run(hh); b = show(base)
print(f"\n【B】 30·29세 맞벌이, 연 소득 9,000만 원, 저축률 15%, 퇴직연금 적립")
print(f"  기본생활 유지 {b[0]*100:.0f}% · 여행·취미 충족 {b[1]*100:.0f}% · 1억 남기기 {b[2]*100:.0f}%")
sal_tot = sum(m.salary for m in hh.members)
rate_up = 0.15 + 600 / sal_tot                      # 월 50만 원 = 연 600만 원을 저축률로 환산
for lab, h2 in [(f"월 50만 원 더 저축(저축률 {rate_up*100:.0f}%)", replace(hh, saving_rate=rate_up)),
                ("은퇴 2년 늦추기", replace(hh, retire_age=62)),
                ("생활비 10% 줄이기", replace(hh, essential=3240, annual_spending=4440)),
                ("퇴직연금이 없다면", replace(hh, retirement_contrib=False, pension_balance=0))]:
    d, ci = diff(run(h2), base)
    print(f"    {lab:22s} {d*100:+6.1f}%p (±{ci*100:.1f})")

# ── C. 집은 있고 현금은 적은 가구 ──
hh = couple_65(); house = HouseConfig(official=50_000, market=70_000, n_houses=1, years_held=15)
base = run(hh, house=house); b = show(base)
print(f"\n【C】 65·63세 부부, 금융자산 3억, 집 시세 7억(공시 5억)")
print(f"  그대로:                기본생활 {b[0]*100:3.0f}% · 여행 {b[1]*100:3.0f}% · 3억 남기기 {b[2]*100:3.0f}%")
for lab, hc in [("주택연금 지금 가입(63세)", replace(house, reverse_mortgage_age=63)),
                ("주택연금 70세 가입", replace(house, reverse_mortgage_age=70)),
                ("70세에 절반 가격 집으로", replace(house, downsize_age=70, downsize_ratio=0.5))]:
    r = run(hh, house=hc); v = show(r); d, _ = diff(r, base)
    extra = (f"월 {r.get('rm_monthly', 0):.0f}만 원" if r.get("rm_monthly") else
             (f"현금 {r.get('ds_cash', 0)/10000:.1f}억 확보" if r.get("ds_cash") else "—"))
    print(f"  {lab:22s} 기본생활 {v[0]*100:3.0f}% ({d*100:+5.1f}%p) · 여행 {v[1]*100:3.0f}% · 3억 남기기 {v[2]*100:3.0f}% · {extra}")

# ── D. 가장 큰 불확실성 ──
print("\n【D】 주식 위험프리미엄 ±1σ(5.2% ± 1.4%p) — 52·50세 부부 기준")
hh = couple_52()
for lab, erp in [("3.8%", 0.038), ("5.2%", 0.052), ("6.6%", 0.066)]:
    print(f"    ERP {lab}: 기본생활 유지 {show(run(hh, eco=replace(ECO, erp=erp)))[0]*100:.0f}%")
