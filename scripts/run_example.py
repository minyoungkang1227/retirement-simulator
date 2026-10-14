import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""라이브러리를 코드에서 쓰는 가장 짧은 예제 — 60세 부부, 금융자산 5억, 생활비 월 300만 원.

    PYTHONPATH=. python scripts/run_example.py

화면(Streamlit) 없이 엔진만 돌려 보고 싶을 때 여기서 시작하면 된다.
README 그림은 scripts/make_readme_images.py 가 따로 만든다.
"""
import numpy as np
from retire_sim.config import Person, Household, SimConfig, CareMarkov
from retire_sim.economy_v2 import EconomyV2
from retire_sim import engine_tax, mortality

hh = Household(
    members=[Person(age=60, sex="M", nps_monthly=110, nps_start_age=65,
                    private_pension_annual=600, private_pension_start=60),
             Person(age=58, sex="F", nps_monthly=50, nps_start_age=65)],
    liquid_assets=50_000, stock_weight=0.4, annual_spending=3600)
cfg = SimConfig(care=CareMarkov())

q = mortality.default_qx()
print(f"[사망률 체크] 60세 남 기대여명 {mortality.life_expectancy(q['M'], 60):.1f}년, "
      f"58세 여 {mortality.life_expectancy(q['F'], 58):.1f}년")

res = engine_tax.run(hh, cfg, e=EconomyV2.enhanced())
for k, v in engine_tax.summarize(res).items():
    print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

# 시나리오 비교: 국민연금 연기수령
for start in (60, 65, 70):
    for m in hh.members: m.nps_start_age = start
    p = engine_tax.summarize(engine_tax.run(hh, cfg, e=EconomyV2.enhanced()))["고갈확률"]
    print(f"국민연금 {start}세 개시 → 고갈확률 {p:.1%}")
for m in hh.members: m.nps_start_age = 65
