import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""'어떤 상품을 고르느냐' vs '언제부터 받느냐' — 어느 쪽이 더 큰 선택인가.

실행: PYTHONPATH=. python scripts/product_spread.py

실명 상품을 모으지 않고 **공시된 사업비 범위의 양 끝**으로 상품 선택의 폭을 잡는다.
공시이율·최저보증이율은 상품별 범위가 공개돼 있지 않아 대표값을 쓴다(그만큼 상품 쪽 폭은 과소평가).
그래도 결론의 방향은 바뀌지 않는다 — 개시 나이 쪽이 압도적으로 크다.
"""
import numpy as np

from retire_sim import mortality, product
from retire_sim.config import Person
from retire_sim.economy_v2 import EconomyV2, generate

ec = EconomyV2.enhanced()
eco = generate(ec, 60, 5000, np.random.default_rng(42))
qx = mortality.default_qx()

for age, sex, prem, years in ((45, "M", 50, 20), (35, "F", 30, 25)):
    prod = product.AnnuityProduct(monthly_premium=prem, pay_years=years, start_age=65,
                                  credit_spread=0.023 - (ec.q0 + ec.pi0), guarantee_years=10)
    me = [Person(age=age, sex=sex)]
    out = product.choice_spread(prod, me, qx, eco, stock_weight=0.4)
    print(f"\n== {age}세 {'남' if sex == 'M' else '여'} · 월 {prem}만 원 {years}년 납입")
    for k, v in out["상품 선택"].items():
        print(f"   상품 사업비 {k:5s}   첫해 연금 {v:8,.0f}만 원")
    for k, v in out["개시 나이"].items():
        print(f"   개시 {k}세        첫해 연금 {v:8,.0f}만 원")
    print(f"   → 폭: 상품 선택 {out['폭']['상품 선택'] * 100:5.1f}%  vs  "
          f"개시 나이 {out['폭']['개시 나이'] * 100:5.1f}%")
print("\n사업비 범위 출처: 보험연구원 「변액연금 최저보증 및 사업비 부과 현황 조사」 월납 기준")
print("  계약체결비용 5.6~6.9% · 계약관리비용 4.2~9%")
