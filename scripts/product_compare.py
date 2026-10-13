import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""연금보험 상품 비교: 같은 돈을 연금보험·펀드·예금에 넣으면 어떻게 달라지는가.

실행: PYTHONPATH=. python scripts/product_compare.py

결론(기본 설정): 연금보험은 **모으는 데는 가장 불리**하고(사업비 + 시장금리보다 낮은 적립이율),
**오래 사는 위험을 넘기는 데** 값이 있다. 적립금만 비교하면 연금보험을 과소평가하게 된다.
"""
import numpy as np

from retire_sim import mortality
from retire_sim.config import Person
from retire_sim.economy_v2 import EconomyV2, generate
from retire_sim.product import AnnuityProduct, compare_vehicles, guarantee_value

ECO = EconomyV2.enhanced()
QX = mortality.default_qx()
AGE, T, N = 45, 55, 10_000                      # 45세 → 100세, 1만 경로
eco = generate(ECO, T, N, np.random.default_rng(42))
me = [Person(AGE, "M")]
prod = AnnuityProduct(monthly_premium=50, pay_years=20, start_age=65)

c = compare_vehicles(prod, me, QX, eco)
cap = T - (c["개시 나이"] - AGE)                 # 계산 상한(그 이상은 '끝까지')
fmt = lambda t, u="": f"{t[0]:>8,.0f} {u}/{t[1]:>8,.0f} {u}/{t[2]:>8,.0f} {u}"


def yrs(t):
    f = lambda v: ("끝까지" if v >= cap else f"{v:.0f}년")
    return f"{f(t[0]):>8} /{f(t[1]):>8} /{f(t[2]):>8}"


print(f"{AGE}세 남성 · 월 {prod.monthly_premium:.0f}만 원 {prod.pay_years}년 납입 · {prod.start_age}세 연금개시")
print(f"낸 돈 {c['낸 돈(명목 중앙값)']:,.0f}만 원 중 사업비 {c['사업비(명목 중앙값)']:,.0f}만 원"
      f" ({c['사업비(명목 중앙값)'] / c['낸 돈(명목 중앙값)'] * 100:.1f}%)\n")
print("                      하위 10%  /   중앙값   /  상위 10%")
for k, v in c["연금개시 적립금"].items():
    print(f"  {prod.start_age}세 적립금  {k:5s} {fmt(v)} 만원")
print(f"  연금보험 첫해 연금액  {fmt(c['연금보험 첫해 연금액'])} 만원/년")
print()
for k, v in c["같은 금액을 꺼낼 때 버티는 햇수"].items():
    print(f"  같은 금액을 꺼내면  {k:5s} {yrs(v)}")
print("\n  살아 있는 동안 돈이 바닥날 확률")
for k, v in c["살아 있는 동안 바닥날 확률"].items():
    print(f"    {k:6s} {v * 100:5.1f}%")
print("  → 적립금만 보면 연금보험이 가장 적다. 그런데 바닥날 위험은 연금보험만 0이다.")
print("     연금보험이 파는 것은 수익률이 아니라 '오래 살 위험'을 떠안아 주는 일이다.\n")

g = guarantee_value(prod, me, QX, eco["rate"], eco["cpi"])
print("최저보증이율의 가치 (보증 있음 − 없음, 첫해 연금액 기준)")
print(f"  평균        {g['평균 연금액 증가']:6.1f} 만원/년")
print(f"  하위 10% 경로 {g['하위 10% 경로에서 증가']:6.1f} 만원/년   ← 금리가 낮게 흘러간 경우")
print(f"  보증이 한 번이라도 작동한 경로 {g['보증이 작동한 경로 비율'] * 100:.0f}%")
print("  → 금리에 대한 풋옵션이다. 평균으로 보면 작고, 나쁜 경로에서만 값을 한다.\n")

print("상품 스펙을 바꿔 보면 (첫해 연금액 중앙값)")
from dataclasses import replace
for lab, pr in [("기준", prod),
                ("사업비 없는 상품", replace(prod, acq_rate_early=0, acq_rate_late=0, admin_rate_pay=0, admin_rate_post=0)),
                ("공시이율 +0.5%p", replace(prod, credit_spread=prod.credit_spread + 0.005)),
                ("최저보증 1.5% 균일", replace(prod, guarantee=((10 ** 9, 0.015),))),
                ("보증지급 20년", replace(prod, guarantee_years=20)),
                ("70세 개시", replace(prod, start_age=70))]:
    from retire_sim.product import run as prun
    r = prun(pr, me, QX, eco["rate"], eco["cpi"])
    print(f"  {lab:16s} {np.median(r['first_payment']):7,.0f} 만원/년  (개시 {pr.start_age}세)")
