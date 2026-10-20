import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""재산분 건보료(v45 법정 산식)가 결과를 얼마나 바꾸는지 짝지은 비교로 측정.

    PYTHONPATH=. python scripts/measure_hi_property.py

같은 시드·같은 난수로 '재산분 0' vs '별표4 자동계산'을 돌려 차이와 신뢰구간을 낸다.
"""
import numpy as np

from retire_sim import engine_tax, mortality
from retire_sim.config import CareMarkov, Household, Person, SimConfig
from retire_sim.economy_v2 import EconomyV2
from retire_sim.tax import HI_POINT_VALUE, HouseConfig, TaxConfig, hi_property_monthly, hi_property_points

E, T, Q = EconomyV2.enhanced(), TaxConfig(overseas_share=0.2), mortality.default_qx()
C = SimConfig(n_paths=10000, care=CareMarkov())


def hh_with_house(official, market):
    return Household(members=[Person(60, "M", nps_monthly=110, private_pension_annual=600, private_pension_start=60),
                              Person(58, "F", nps_monthly=50)],
                     liquid_assets=50_000, stock_weight=0.4, annual_spending=3600,
                     essential=3000, lifestyle=600, legacy_target=10_000)


print("① 점수표 손검산 (공제 1억 차감 후 → 점수 → 월 보험료, 만원)")
for base in (8000, 10000, 12250, 15000, 20000, 30000, 50000, 100000, 200000):
    pts = float(hi_property_points(np.array([float(base)]))[0])
    pm = float(hi_property_monthly(np.array([float(base)]), T)[0])
    print(f"  과세표준 {base:>7,}만원 → 공제후 {max(base - 10000, 0):>7,} → {pts:>6.0f}점 → 월 {pm:>6.2f}만원 (연 {pm * 12:>7.1f})")

print(f"\n  점수당 금액 {HI_POINT_VALUE}원 · 장기요양 포함 배수 {1 + T.ltc_ratio:.4f}")

print("\n② 짝지은 비교: 재산분 0 vs 별표4 자동계산")
for official, market, label in ((50000, 70000, "공시 5억 / 시세 7억"),
                                (90000, 120000, "공시 9억 / 시세 12억"),
                                (140000, 190000, "공시 14억 / 시세 19억")):
    hh = hh_with_house(official, market)
    off = HouseConfig(official=official, market=market, n_houses=1, years_held=20, hi_property_monthly=0.0)
    auto = HouseConfig(official=official, market=market, n_houses=1, years_held=20, hi_property_monthly=None)
    ra = engine_tax.run(hh, C, e=E, tc=T, qx_table=Q, house=off)
    rb = engine_tax.run(hh, C, e=E, tc=T, qx_table=Q, house=auto)
    sa, sb = engine_tax.summarize(ra), engine_tax.summarize(rb)
    oka = (ra["depleted_at"] < 0).astype(float)
    okb = (rb["depleted_at"] < 0).astype(float)
    d = (okb - oka).mean() * 100
    se = (okb - oka).std(ddof=1) / np.sqrt(len(oka)) * 100
    print(f"  {label}: {sa['기본생활 유지 확률'] * 100:.1f}% → {sb['기본생활 유지 확률'] * 100:.1f}% "
          f"({d:+.2f}%p, 95% CI [{d - 1.96 * se:+.2f}, {d + 1.96 * se:+.2f}])  "
          f"20년 건보료 {sa['20년 건보료']:,.0f} → {sb['20년 건보료']:,.0f}만원")
