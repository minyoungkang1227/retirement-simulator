"""회귀 테스트: 코드를 고쳐도 대표 가구의 결과가 크게 달라지지 않는지 자동 확인.
실행: pytest -q    (커밋할 때 GitHub Actions가 자동 실행)
"""
import os

from dataclasses import replace

import numpy as np
import pytest

from retire_sim.config import Person, Household, SimConfig, CareMarkov, AddOns
from retire_sim.economy_v2 import EconomyV2, generate, gaussian2_price, vasicek_price, integrated_cov
from retire_sim import engine_tax, metrics, mortality
from retire_sim.tax import TaxConfig, HouseConfig, reverse_mortgage_monthly
from retire_sim.pension import estimate_nps_monthly

ECO = EconomyV2.enhanced()
CFG = SimConfig(care=CareMarkov())


def _run(hh, **kw):
    kw.setdefault("e", ECO); kw.setdefault("tc", TaxConfig(overseas_share=0.2))
    return engine_tax.run(hh, CFG, **kw)


def couple_60():
    return Household(members=[Person(60, "M", nps_monthly=110, private_pension_annual=600, private_pension_start=60),
                              Person(58, "F", nps_monthly=50)],
                     liquid_assets=50_000, stock_weight=0.4, annual_spending=3600, essential=3000, lifestyle=600,
                     legacy_target=10_000)


def couple_30():
    n1 = estimate_nps_monthly(5000, 3, 30, 60); n2 = estimate_nps_monthly(4000, 2, 29, 59)
    return Household(members=[Person(30, "M", nps_monthly=n1, salary=5000), Person(29, "F", nps_monthly=n2, salary=4000)],
                     liquid_assets=5000, stock_weight=0.5, annual_spending=3600, essential=3600, lifestyle=1200,
                     legacy_target=10_000, retire_age=60, saving_rate=0.15, pension_balance=1000, retirement_contrib=True)


# ── 1. 대표 가구 결과 (범위로 고정: 분산 감소·리팩터링은 허용, 로직 변화는 잡아냄) ──
@pytest.mark.parametrize("hh_fn,key,lo,hi", [
    (couple_60, "기본생활 유지 확률", 0.65, 0.75),
    (couple_30, "기본생활 유지 확률", 0.73, 0.83),
    (couple_30, "여행·취미 평균 충족률", 0.87, 0.97),
])
def test_headline(hh_fn, key, lo, hi):
    s = engine_tax.summarize(_run(hh_fn()))
    assert lo <= s[key] <= hi, f"{key}={s[key]:.3f} (허용 {lo}~{hi})"


# ── 2. 방향성: 당연히 이래야 하는 관계 ──
def test_directions():
    base = _run(couple_30())
    b = engine_tax.summarize(base)["기본생활 유지 확률"]
    more_save = engine_tax.summarize(_run(couple_30().__class__(**{**couple_30().__dict__, "saving_rate": 0.25})))["기본생활 유지 확률"]
    less_spend = engine_tax.summarize(_run(couple_30().__class__(**{**couple_30().__dict__, "essential": 3000, "annual_spending": 3000})))["기본생활 유지 확률"]
    later = engine_tax.summarize(_run(couple_30().__class__(**{**couple_30().__dict__, "retire_age": 65})))["기본생활 유지 확률"]
    assert more_save > b and less_spend > b and later > b


def test_shock_order():
    """같은 폭락도 은퇴 직후가 더 치명적이어야 한다 (수익률 순서 위험)."""
    hh = couple_60()
    early = engine_tax.summarize(_run(hh, e=EconomyV2.enhanced(stock_shocks={0: -0.4})))["기본생활 유지 확률"]
    late = engine_tax.summarize(_run(hh, e=EconomyV2.enhanced(stock_shocks={10: -0.4})))["기본생활 유지 확률"]
    assert early < late


# ── 3. 이론값 검증 ──
def test_economy_moments():
    d = generate(EconomyV2.enhanced(erp_sd=0, pi_theta_sd=0, q_theta_sd=0, jump_lambda=0, fee=0), 60, 20_000,
                 np.random.default_rng(0))
    e = EconomyV2.enhanced()
    assert abs(d["rate"][:, -1].mean() - (e.q_theta + e.pi_theta)) < 0.004
    assert abs(d["infl"][:, -1].mean() - e.pi_theta) < 0.004
    # 장기 표준편차도 OU 정상분포 이론값 σ/√(2κ)와 맞아야 한다 (전에는 scripts/validate.py 가 보던 몫)
    for key, sig, kap in (("infl", e.pi_sigma, e.pi_kappa), ("q", e.q_sigma, e.q_kappa)):
        if key in d:
            assert abs(d[key][:, -1].std() - sig / np.sqrt(2 * kap)) < 0.004, key


def test_bond_price_reduces_to_vasicek():
    e = EconomyV2()
    v = vasicek_price(5.0, np.array([0.025]), e)[0]
    g = gaussian2_price(5.0, np.array([0.025]), np.array([0.0]), e.r_kappa, e.r_theta, e.r_sigma, 1.0, 0.0, 1e-9, 0.0)[0]
    assert abs(v - g) < 1e-9


def test_integrated_cov_symmetric_psd():
    M = integrated_cov(0.2, 0.008, 0.4, 0.01, 0.3, -0.2, -0.1)
    assert np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > -1e-12


def test_life_expectancy_matches_table():
    q = mortality.default_qx()
    assert abs(mortality.life_expectancy(q["M"], 60) - 23.7) < 0.2
    assert abs(mortality.life_expectancy(q["F"], 60) - 28.4) < 0.2


def test_care_preserves_mortality():
    q = mortality.default_qx()
    al, ic, sv = mortality.simulate_life(q["M"], 60, 45, 40_000, np.random.default_rng(0), CareMarkov(), return_severe=True)
    assert abs((al.sum(1).mean() - 0.5) - 23.7) < 0.5
    assert 0.05 < sv[:, 5:].sum() / max(ic[:, 5:].sum(), 1) < 0.25       # 중증 비율 ≈ 12.7%


def test_reverse_mortgage_table():
    assert abs(reverse_mortgage_monthly(70, 30_000) - 92.3) < 1.5        # 공사 표: 70세·3억 → 월 92.3만원
    assert reverse_mortgage_monthly(80, 50_000) > reverse_mortgage_monthly(70, 50_000)


# ── 4. 계산 신뢰성 ──
def test_paired_ci_smaller_than_independent():
    hh = couple_60()
    a, b = _run(hh), _run(Household(**{**hh.__dict__, "stock_weight": 0.6}))
    c = metrics.compare(b, a)
    assert c["±"] < c["독립 가정 시 ±"]


def test_reproducible():
    s1 = engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]
    s2 = engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]
    assert s1 == s2


# ── 5. 기능이 꺼져 있을 때 결과가 바뀌지 않아야 ──
def test_addons_noop():
    hh = couple_60()
    a = engine_tax.summarize(_run(hh))["기본생활 유지 확률"]
    b = engine_tax.summarize(_run(hh, addons=AddOns()))["기본생활 유지 확률"]
    assert a == b


def test_house_strategies_help_cash_poor():
    hh = Household(members=[Person(65, "M", nps_monthly=130), Person(63, "F", nps_monthly=60)],
                   liquid_assets=30_000, stock_weight=0.5, annual_spending=3000, essential=3000, lifestyle=1200)
    H = HouseConfig(official=50_000, market=70_000, n_houses=1, years_held=20, hi_property_monthly=15)
    base = engine_tax.summarize(_run(hh, house=H))["기본생활 유지 확률"]
    rm = engine_tax.summarize(_run(hh, house=HouseConfig(**{**H.__dict__, "reverse_mortgage_age": 70})))["기본생활 유지 확률"]
    assert rm > base + 0.2


# ── 6. 집값 위험 (v26) ──
def test_house_index_distribution():
    from retire_sim.economy_v2 import generate
    d = generate(EconomyV2.enhanced(), 30, 20_000, np.random.default_rng(0))
    H = d["house"]
    assert H.shape[1] == 31 and np.allclose(H[:, 0], 1.0)
    lr = np.diff(np.log(H), axis=1)
    assert 0.08 < lr.std() < 0.12                                   # 연 변동성 ≈ 10%
    assert 0.0 < np.log(np.median(H[:, 30])) / 30 < 0.03            # 실질 상승률 ≈ 1.25%
    assert np.percentile(H[:, 30], 5) < 0.8 < 1.3 < np.percentile(H[:, 30], 95)


def test_house_risk_off_reproduces_v25():
    """집값 파라미터를 끄면 v25와 같아야 한다(집값 = 물가만큼)."""
    flat = EconomyV2.enhanced(house_sigma=0, house_real_growth=0, house_real_sd=0)
    d = generate(flat, 20, 5000, np.random.default_rng(0))
    assert np.allclose(d["house"], 1.0)


# ── 7. 사망률 개선·지표·연금 상품 (v27) ──
def test_mortality_improvement():
    q = mortality.default_qx()
    assert mortality.life_expectancy(q["M"], 60, improve=True) > mortality.life_expectancy(q["M"], 60) + 1.0
    p1 = mortality.project_qx(q["M"], 10)
    assert (p1[30:100] < q["M"][30:100]).all() and p1[-1] == 1.0


def test_cte_monotone_and_definition():
    from retire_sim.engine_tax import cte
    x = np.arange(100.0)
    assert cte(x, 0.9) == pytest.approx(x[90:].mean())
    s = engine_tax.summarize(_run(couple_60()))
    assert s["평균 부족액(실질)"] <= s["CTE70 부족액(실질)"] <= s["CTE90 부족액(실질)"]


def test_certainty_equivalent_positive_and_responsive():
    rich = Household(**{**couple_60().__dict__, "liquid_assets": 200_000})
    a = engine_tax.summarize(_run(couple_60()))["확실성등가 소비(연, 1인)"]
    b = engine_tax.summarize(_run(rich))["확실성등가 소비(연, 1인)"]
    assert 0 < a < b


def test_annuity_variants_pricing_order():
    """같은 보험료라면: 보증기간·체증·부부형이 붙을수록 처음 받는 금액이 작아야 한다."""
    hh = couple_60()
    def pay(**kw):
        kw.setdefault("annuity_start_age", 65)
        return _run(hh, addons=AddOns(annuity_premium=10_000, **kw))["annuity_pay"]
    plain = pay()
    assert pay(annuity_guarantee_years=20) < plain
    assert pay(annuity_escalation=0.02) < plain
    assert pay(annuity_joint=True) < plain
    assert plain < pay(annuity_start_age=80)                       # 늦게 받을수록 금액은 커진다
    assert pay(annuity_joint=True) < pay(annuity_joint=True, annuity_survivor_ratio=0.7)


# ── v28 제도 정밀화: 유족연금 · 기초연금 ──
def test_survivor_rules():
    from retire_sim.pension import survivor_pct, survivor_split, survivor_annual
    assert (survivor_pct(5), survivor_pct(10), survivor_pct(19.9), survivor_pct(20), survivor_pct(35)) == (0.4, 0.5, 0.5, 0.6, 0.6)
    own, sv = survivor_split(np.array([100.0, 10.0, 0.0]), np.array([60.0, 60.0, 60.0]), 0.30)
    # own 100: 100+18=118 > 60 → 본인연금+30%; own 10: 10+18=28 < 60 → 유족연금만; own 0 → 유족연금만
    assert np.allclose(own, [100, 0, 0]) and np.allclose(sv, [18, 60, 60])
    assert abs(survivor_annual(Person(65, "M", nps_monthly=100, nps_years=25)) - 720) < 1e-9


def test_basic_pension_rules():
    """기초연금 산정: 자격 판정 → 연계감액 → 부부감액 → 소득역전방지."""
    from retire_sim.pension import basic_pension_annual, basic_pension_one
    from retire_sim.config import NPSRules
    r = NPSRules(); one = np.ones(1); z = np.zeros(1); B = r.basic_amount
    yes, no = one > 0, one < 0

    def call(income=0.0, n=1, nps=0.0, a=0.0, sv=False, asset=0.0, fin=0.0, rules=r):
        inc = np.array([float(income)])
        return basic_pension_annual(inc, np.array([float(asset)]), np.array([float(fin)]),
                                    [yes] * n, [np.array([float(nps)])] * n, [np.array([float(a)])] * n,
                                    [sv * one > 0] * n, rules, one)[0]

    assert abs(call() - B * 12) < 1e-6                                   # 단독·국민연금 없음 → 기준연금액
    assert abs(call(n=2) - 2 * B * 0.8 * 12) < 1e-6                      # 부부 모두 수급 → 각 20% 감액
    assert call(income=300.0) == 0                                       # 선정기준액(단독 247만) 초과 → 0
    assert basic_pension_annual(z, z, z, [no], [z], [z], [no], r, one)[0] == 0   # 64세는 대상 아님

    # 연계감액: 국민연금이 기준연금액의 150% 이하면 A급여와 무관하게 전액
    assert basic_pension_one(np.array([1.5 * B]), np.array([3 * B]), B)[0] == B
    # A급여가 0이면 산식②가 기준연금액을 넘어 상한에 걸리므로 국민연금이 아무리 많아도 감액 없음
    assert basic_pension_one(np.array([5 * B]), np.array([0.0]), B)[0] == B
    # 두 산식 중 큰 값: A=1.2B, 국민연금=1.9B → 산식① 0.6B, 산식② 0.7B → 0.7B
    mid = basic_pension_one(np.array([1.9 * B]), np.array([1.2 * B]), B)[0]
    assert abs(mid - 0.7 * B) < 1e-9 and 0.5 * B < mid < B
    # 하한은 부가연금액(기준연금액의 50%)
    assert abs(basic_pension_one(np.array([5 * B]), np.array([5 * B]), B)[0] - 0.5 * B) < 1e-9
    assert call(nps=5 * B, a=5 * B, sv=True) == B * 12                   # 유족연금 수급자는 연계감액 없음

    # 지역별 기본재산액: 공제가 작은 농어촌은 같은 재산이라도 소득인정액이 커져 탈락할 수 있다
    # 재산 8.5억: 대도시는 (8.5 − 1.35)억 환산 월 238만 < 247만 → 수급, 농어촌은 (8.5 − 0.725)억 → 259만 → 탈락
    assert call(asset=85000.0) > 0
    assert call(asset=85000.0, rules=NPSRules(basic_region="농어촌")) == 0
    assert call(asset=85000.0, rules=NPSRules(basic_region="중소도시")) == 0


def test_a_benefit_and_region():
    from retire_sim.pension import a_benefit_monthly, NPS_A, NPS_C
    from retire_sim.config import NPSRules
    assert a_benefit_monthly(Person(65, "M", nps_years=9)) == 0                  # 10년 미만은 0
    assert abs(a_benefit_monthly(Person(65, "M", nps_years=20)) - NPS_C * NPS_A / 12) < 1e-9
    assert a_benefit_monthly(Person(65, "M", nps_years=40)) > a_benefit_monthly(Person(65, "M", nps_years=20))
    assert a_benefit_monthly(Person(65, "M", nps_years=50)) == a_benefit_monthly(Person(65, "M", nps_years=40))
    assert (NPSRules().base_asset(), NPSRules(basic_region="중소도시").base_asset(),
            NPSRules(basic_region="농어촌").base_asset()) == (13500.0, 8500.0, 7250.0)


def test_v28_directions():
    from retire_sim.config import NPSRules
    global CFG
    old = CFG
    try:
        res = {}
        for sv, bs in [(False, False), (True, False), (False, True)]:
            CFG = SimConfig(care=CareMarkov(), nps=NPSRules(survivor_on=sv, basic_on=bs))
            res[(sv, bs)] = engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]
        assert res[(True, False)] > res[(False, False)]
        assert res[(False, True)] > res[(False, False)]
    finally:
        CFG = old


# ── v29 최적화: 글라이드 경로 · 인출 순서 · 탐색 ──
def test_glide_and_order():
    from dataclasses import replace
    hh = couple_60()
    flat = engine_tax.summarize(_run(hh))
    same = engine_tax.summarize(_run(replace(hh, glide_end=hh.stock_weight)))      # 시작=끝이면 고정 비중과 동일
    assert abs(flat["기본생활 유지 확률"] - same["기본생활 유지 확률"]) < 1e-9
    low = engine_tax.summarize(_run(replace(hh, glide_end=0.1)))
    assert low["기본생활 유지 확률"] != flat["기본생활 유지 확률"]
    isa = engine_tax.summarize(_run(replace(hh, draw_order="isa_first")))
    assert 0 <= isa["고갈확률"] <= 1


def test_optimize_search_structure():
    from retire_sim import optimize
    out = optimize.search(couple_60(), CFG, "nps", e=ECO, tc=TaxConfig(overseas_share=0.2))
    assert len(out["rows"]) == 5 and out["rows"][2]["고갈확률 차이(기준 대비)"] == 0.0
    assert set(out["check"]) == {"기준", "추천 후보"}
    # 연금 개시를 미루면 (같은 시나리오에서) 이 가구는 기본생활 유지 확률이 높아진다
    assert out["rows"][4]["기본생활 유지 확률"] > out["rows"][0]["기본생활 유지 확률"]


# ── v30 장수위험: 확률적 사망률 · 사별 효과 ──
def test_longevity_multiplier():
    m = mortality.longevity_multiplier(20000, 30, 0.05, 0.005, 42)
    assert m.shape == (20000, 30) and abs(m.mean() - 1) < 0.02          # 평균 1 근처(분산 보정)
    assert m[:, 25].std() > m[:, 1].std()                                # 시간이 갈수록 불확실성 증가
    assert mortality.longevity_multiplier(10, 5, 0, 0, 42) is None
    assert np.allclose(mortality.longevity_multiplier(100, 5, 0.05, 0.005, 7), mortality.longevity_multiplier(100, 5, 0.05, 0.005, 7))


def test_widowhood_effect():
    q = mortality.default_qx(110)
    n, T = 40000, 30
    rng = np.random.default_rng(0)
    al = np.stack([mortality.simulate_alive(q["M"], 70, T, n, rng), mortality.simulate_alive(q["F"], 68, T, n, rng)])
    a2, _, _ = mortality.apply_widowhood(al, None, None, [q["M"], q["F"]], [70, 68], ["M", "F"], (0.3, 0.15), 3.0, False, 1)
    assert (a2 <= al).all()                                               # 추가 사망만 생길 수 있음
    both = al[0].sum(1) + al[1].sum(1); both2 = a2[0].sum(1) + a2[1].sum(1)
    assert both2.mean() < both.mean()
    off, _, _ = mortality.apply_widowhood(al, None, None, [q["M"], q["F"]], [70, 68], ["M", "F"], (0, 0), 3.0, False, 1)
    assert (off == al).all()


def test_v30_tail_risk_up():
    global CFG
    old = CFG
    try:
        CFG = SimConfig(care=CareMarkov(), longevity_sigma_level=0, longevity_sigma_trend=0, widow_excess=(0, 0))
        off = engine_tax.summarize(_run(couple_30()))["CTE90 부족액(실질)"]
        CFG = SimConfig(care=CareMarkov(), widow_excess=(0, 0))
        on = engine_tax.summarize(_run(couple_30()))["CTE90 부족액(실질)"]
        assert on > off                                                   # 장수 불확실성은 꼬리 위험을 키운다
    finally:
        CFG = old


# ── v31 해석해 대조: 결정론적 극한에서 닫힌식과 정확히 일치해야 함 ──
def _deterministic_setup(R=0.03, spend=3000.0, infl=0.0):
    """변동성·세금·사망·간병·연금을 모두 끈 세계. 유일한 움직임은 수익률과 지출."""
    from retire_sim.config import NPSRules
    eco = EconomyV2(r0=R, r_theta=R, r_sigma=0.0, q0=R - infl, q_theta=R - infl, q_sigma=0.0,
                    pi0=infl, pi_theta=infl, pi_sigma=0.0, erp=0.0, s_sigma=0.0, fee=0.0,
                    jump_lambda=0.0, erp_sd=0.0, pi_theta_sd=0.0, q_theta_sd=0.0, antithetic=False,
term_premium=0.0, house_real_sd=0.0, house_sigma=0.0)
    qx = {"M": np.zeros(111), "F": np.zeros(111)}
    qx["M"][100] = qx["F"][100] = 1.0                       # 100세까지 아무도 죽지 않음
    hh = Household(members=[Person(60, "M")], liquid_assets=50_000, stock_weight=1.0,
                   annual_spending=spend, essential=spend, lifestyle=0.0, legacy_target=0.0,
                   survivor_spending_ratio=1.0, floor_method="fixed95")
    cfg = SimConfig(n_paths=64, seed=1, care=CareMarkov(enabled=False), mortality_improvement=False,
                    longevity_sigma_level=0.0, longevity_sigma_trend=0.0, widow_excess=(0.0, 0.0),
                    nps=NPSRules(basic_on=False, survivor_on=False))
    return hh, cfg, eco, qx


def test_analytic_benchmark_no_inflation():
    """W(t+1) = (W(t) − 지출) × e^r 와 상대오차 1e-7 안에서 같아야 한다 (지출은 연초, 수익은 연말)."""
    R, spend = 0.03, 3000.0
    hh, cfg, eco, qx = _deterministic_setup(R, spend)
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), qx_table=qx)
    W = res["W_real"][0]
    w = float(hh.liquid_assets)
    for t in range(15):
        w = (w - spend) * np.exp(R)
        assert abs(W[t + 1] - w) <= 1e-7 * abs(w), f"{t + 1}년차: 모형 {W[t + 1]:.4f} vs 닫힌식 {w:.4f}"


def test_analytic_benchmark_with_inflation():
    """물가가 2%여도 실질자산은 실질수익률 기준 닫힌식과 같아야 한다."""
    R, infl, spend = 0.05, 0.02, 3000.0
    hh, cfg, eco, qx = _deterministic_setup(R, spend, infl)
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), qx_table=qx)
    W = res["W_real"][0]
    w = float(hh.liquid_assets)
    for t in range(15):
        w = (w - spend) * np.exp(R) / (1 + infl)            # 실질 = 명목 ÷ 물가지수
        assert abs(W[t + 1] - w) <= 1e-7 * abs(w), f"{t + 1}년차: 모형 {W[t + 1]:.4f} vs 닫힌식 {w:.4f}"


def test_analytic_pension_offsets_spending():
    """연금이 들어오면 꺼내 쓰는 돈이 그만큼 줄어야 한다 (연 1,200만 원 = 월 100만 원)."""
    R, spend = 0.03, 3000.0
    hh, cfg, eco, qx = _deterministic_setup(R, spend)
    hh.members[0].nps_monthly = 100.0                      # 조기·연기 조정이 없도록 정상개시=개시=60세
    hh.members[0].nps_normal_age = hh.members[0].nps_start_age = 60
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), qx_table=qx)
    W = res["W_real"][0]
    w = float(hh.liquid_assets)
    for t in range(15):
        w = (w - (spend - 1200.0)) * np.exp(R)
        assert abs(W[t + 1] - w) <= 1e-7 * abs(w), f"{t + 1}년차: 모형 {W[t + 1]:.4f} vs 닫힌식 {w:.4f}"


# ── v33 상담 세션: 저장·복원과 재현 ID ──
def test_session_roundtrip_and_calc_id():
    import json as _json
    from retire_sim import session, __version__

    inp = {"age": 60, "essential": 250, "spouse": True, "events": [{"age": 70, "amount": 1000}]}
    blob = session.save_blob(inp, "김철수", "홍길동 FP")
    back, client, advisor, warn = session.load_blob(blob)
    assert back == inp and client == "김철수" and advisor == "홍길동 FP" and warn == ""
    assert session.load_blob(blob.encode())[0] == inp                      # bytes 로 넣어도 같다

    # 계산 ID: 키 순서와 무관, 입력이 바뀌면 달라짐, 모델 버전이 바뀌어도 달라짐
    cid = session.calc_id(inp)
    assert len(cid) == 10 and cid == session.calc_id(dict(reversed(list(inp.items()))))
    assert cid != session.calc_id({**inp, "essential": 251})
    assert cid != session.calc_id(inp, version="999")
    assert _json.loads(blob)["calc_id"] == cid

    # 다른 모델 버전에서 저장된 파일은 경고와 함께 열린다
    d = _json.loads(blob); d["model_version"] = "1"
    back2, _, _, warn2 = session.load_blob(_json.dumps(d))
    assert back2 == inp and str(__version__) in warn2

    for bad in ('{"hello": 1}', '{"format": "retire-compass/1"}', '{"format": "other", "inputs": {}}'):
        with pytest.raises(ValueError):
            session.load_blob(bad)


def test_report_html_is_self_contained():
    """리포트는 외부 자료 없이 열려야 한다 — 그림은 SVG 문자열로 직접 그린다."""
    from retire_sim import report

    rows = [(60, 5.0, 4.0, 3.0), (70, 6.0, 4.0, 2.0), (80, 7.0, 3.0, 0.0)]
    svg = report.line_chart_svg(rows)
    assert svg.startswith("<svg") and svg.count("<polyline") == 3
    assert report.line_chart_svg([]) == ""
    # 폭포: 0인 항목은 빼고, 남는 부족분은 '모아둔 돈에서' 막대로 이어 붙인다
    wf = report.waterfall_svg([("연금", 2400), ("생활비", -3600), ("세금", 0.0)])
    assert wf.count("<rect") == 3 and "모아둔 돈에서" in wf
    assert report.waterfall_svg([("연금", 4000), ("생활비", -3600)]).count("<rect") == 2   # 남으면 안 붙는다
    assert report.waterfall_svg([]) == ""

    h = report.build(dict(client="김<철>수", advisor_line="", date="2026년 10월 13일", calc_id="ABC1234567",
                          version="33", hero="테스트", goals=[("A", "B", "C")], gap_eq="x", gap_big="y",
                          band="z", actions=[("무엇", "어떻게", "10% → 20%")], fan_svg=svg, flow_svg=wf,
                          assumptions="가정"))
    assert h.startswith("<!doctype html>") and "ABC1234567" in h and "무엇" in h
    assert "김&lt;철&gt;수" in h and "<철>" not in h                        # 고객명은 HTML 이스케이프
    assert "@media print" in h and "page-break-before" in h                # 인쇄 레이아웃
    assert "<img" not in h and "<script" not in h                          # 외부 이미지·스크립트 없음


def test_app_and_library_versions_match():
    """앱이 전제하는 모델 버전과 라이브러리 버전이 어긋나면 배포 때 조용히 깨진다 — 여기서 막는다."""
    import re
    from retire_sim import __version__
    src = open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py"), encoding="utf-8").read()
    expects = int(re.search(r"^APP_EXPECTS_MODEL = (\d+)", src, re.M).group(1))
    assert int(__version__) >= expects, (
        f"retire_sim v{__version__} < 앱이 요구하는 v{expects} — retire_sim/__init__.py 의 __version__ 을 올리세요")


# ── v34 연금보험 상품: 공시이율·최저보증·사업비·해지공제 ──
def _prod_eco(T=45, n=2000, seed=11):
    from retire_sim.economy_v2 import generate
    return generate(ECO, T, n, np.random.default_rng(seed))


def test_product_minimum_guarantee_is_a_floor():
    """최저보증이율은 공시이율의 바닥이다. 금리가 아무리 낮아도 그 아래로 안 내려간다."""
    from retire_sim.product import AnnuityProduct, credited_rate

    pr = AnnuityProduct()
    zero = np.zeros((50, 30))                                   # 시장금리 0% 세계
    on = credited_rate(pr, zero, guarantee=True)
    off = credited_rate(pr, zero, guarantee=False)
    assert np.allclose(off, 0.0)                                # 보증이 없으면 0까지 내려간다
    assert np.allclose(on[:, 0], 0.010) and np.allclose(on[:, 7], 0.0075) and np.allclose(on[:, 12], 0.005)
    assert (on >= off - 1e-12).all()
    high = np.full((50, 30), 0.08)                              # 금리가 높으면 보증은 작동하지 않는다
    assert np.allclose(credited_rate(pr, high, True), credited_rate(pr, high, False))


def test_product_costs_and_surrender():
    """사업비는 적립금을 줄이고, 해지공제 때문에 초기 해지는 낸 돈보다 적게 돌려받는다."""
    from retire_sim.product import AnnuityProduct, accumulate

    eco = _prod_eco(); rate, cpi = eco["rate"][:, :30], eco["cpi"]
    pr = AnnuityProduct(monthly_premium=50, pay_years=20, start_age=65)
    full = accumulate(pr, rate, cpi, until=20)
    free = accumulate(replace(pr, acq_rate_early=0, acq_rate_late=0, admin_rate_pay=0, admin_rate_post=0), rate, cpi, until=20)
    assert (free["balance"][:, 20] > full["balance"][:, 20]).all()          # 사업비가 없으면 더 쌓인다
    assert full["cost"].sum(1).mean() > 0
    paid = full["paid"].cumsum(1)
    assert (full["surrender"][:, 1] < paid[:, 0]).all()                     # 1년차 해지 → 원금 손실
    assert (full["surrender"][:, 1:] <= full["balance"][:, 1:] + 1e-9).all()
    assert (full["paid"][:, 20:] == 0).all()                                # 납입기간 뒤엔 보험료 없음
    assert np.allclose(full["balance"][:, 20], full["balance"][:, 29])      # 개시 뒤 적립금은 그대로 둔다


def test_product_annuity_factor_direction():
    """연금현가계수: 보증기간이 길수록·예정이율이 낮을수록·부부형일수록 커진다(= 연금액은 작아진다)."""
    from retire_sim.product import AnnuityProduct, annuity_factor

    qx = mortality.default_qx(); one = [Person(65, "M")]; two = [Person(65, "M"), Person(63, "F")]
    base = AnnuityProduct(guarantee_years=0)
    a = annuity_factor(base, one, qx, 0)
    assert a < annuity_factor(replace(base, guarantee_years=20), one, qx, 0)
    assert a < annuity_factor(replace(base, annuity_rate=0.01), one, qx, 0)   # 할인율이 낮으면 현가가 크다
    assert a < annuity_factor(replace(base, joint=True), two, qx, 0)
    assert a < annuity_factor(replace(base, annuity_mort_mult=0.5), one, qx, 0)   # 더 오래 살면 계수가 커진다


def test_product_guarantee_value_is_a_put():
    """최저보증은 금리가 낮게 흘러간 경로에서만 값을 한다 — 나쁜 경로의 이득이 평균보다 크다."""
    from retire_sim.product import AnnuityProduct, guarantee_value

    eco = _prod_eco(T=45, n=4000)
    g = guarantee_value(AnnuityProduct(monthly_premium=50, pay_years=20, start_age=65),
                        [Person(45, "M")], mortality.default_qx(), eco["rate"], eco["cpi"])
    assert g["평균 연금액 증가"] >= 0 and g["보증 있음 연금액(중앙값)"] >= g["보증 없음 연금액(중앙값)"]
    assert g["하위 10% 경로에서 증가"] > g["평균 연금액 증가"]            # 풋옵션의 성격
    assert 0 <= g["보증이 작동한 경로 비율"] <= 1


def test_product_vehicle_comparison():
    """연금보험은 모으는 데는 불리하지만 '살아 있는 동안 바닥날 확률'은 0이다."""
    from retire_sim.product import AnnuityProduct, compare_vehicles

    eco = _prod_eco(T=55, n=3000)
    c = compare_vehicles(AnnuityProduct(monthly_premium=50, pay_years=20, start_age=65),
                         [Person(45, "M")], mortality.default_qx(), eco)
    assert c["개시 나이"] == 65
    ins, fund = c["연금개시 적립금"]["연금보험"], c["연금개시 적립금"]["펀드"]
    assert ins[1] < fund[1]                                              # 사업비·낮은 적립이율 때문
    out = c["살아 있는 동안 바닥날 확률"]
    assert out["연금보험"] == 0.0 and out["펀드"] > 0 and out["예금"] > out["펀드"]
    assert c["사업비(명목 중앙값)"] > 0 and c["낸 돈(명목 중앙값)"] > 0


# ── v36 간병비 인플레이션: 사람이 하는 일이라 임금을 따라간다 ──
def test_care_cost_inflation_raises_tail_risk():
    """간병비가 일반 물가보다 빨리 오르면 꼬리 위험이 커진다. 기간이 길수록 영향이 크다."""
    global CFG
    old = CFG
    try:
        out = {}
        for g in (0.0, 0.01):
            CFG = SimConfig(care=CareMarkov(), care_real_growth=g)
            out[g] = {name: engine_tax.summarize(_run(fn()))
                      for name, fn in (("60대", couple_60), ("30대", couple_30))}
        for name in ("60대", "30대"):
            a, b = out[0.0][name], out[0.01][name]
            assert b["기본생활 유지 확률"] < a["기본생활 유지 확률"]
            assert b["CTE90 부족액(실질)"] > a["CTE90 부족액(실질)"]
        # 남은 기간이 긴 30대가 복리로 더 크게 받는다
        up = lambda n: out[0.01][n]["CTE90 부족액(실질)"] / out[0.0][n]["CTE90 부족액(실질)"]
        assert up("30대") > up("60대")
    finally:
        CFG = old


def test_care_inflation_zero_matches_old_behaviour():
    """0으로 두면 일반 물가와 같이 오른다 — v35까지의 가정과 같아야 한다."""
    global CFG
    old = CFG
    try:
        CFG = SimConfig(care=CareMarkov(), care_real_growth=0.0)
        a = engine_tax.summarize(_run(couple_60()))
        CFG = SimConfig(care=CareMarkov())          # 기본값은 0.010
        b = engine_tax.summarize(_run(couple_60()))
        assert a["기본생활 유지 확률"] != b["기본생활 유지 확률"]
        assert abs(SimConfig().care_real_growth - 0.010) < 1e-12
    finally:
        CFG = old


# ── v36 더블케어: 부양하는 부모의 간병비 ──
def test_dependent_parents_cost_more_as_share_rises():
    """부모를 많이 부양할수록 노후가 빠듯해진다. 분담 비율과 인원 수에 비례해 나빠져야 한다."""
    from dataclasses import replace as _replace

    hh = couple_30()
    base = engine_tax.summarize(_run(hh))["기본생활 유지 확률"]
    half = engine_tax.summarize(_run(_replace(hh, dependents=[
        dict(age=75, sex="M", share=0.5), dict(age=72, sex="F", share=0.5)])))["기본생활 유지 확률"]
    full = engine_tax.summarize(_run(_replace(hh, dependents=[
        dict(age=75, sex="M", share=1.0), dict(age=72, sex="F", share=1.0)])))["기본생활 유지 확률"]
    one = engine_tax.summarize(_run(_replace(hh, dependents=[
        dict(age=75, sex="M", share=0.5)])))["기본생활 유지 확률"]
    assert full < half < one < base
    # share=0 이나 빈 목록은 아무 영향이 없어야 한다
    zero = engine_tax.summarize(_run(_replace(hh, dependents=[dict(age=75, sex="M", share=0.0)])))
    assert zero["기본생활 유지 확률"] == base
    assert engine_tax.summarize(_run(_replace(hh, dependents=[])))["기본생활 유지 확률"] == base


# ── v36 행동 가정: 패닉 매도와 지출 경직성 ──
def test_panic_selling_hurts():
    """폭락 뒤 주식을 줄이면 회복을 놓친다. 더 크게 줄이거나 늦게 돌아올수록 나빠진다."""
    global CFG
    old = CFG
    try:
        def P(**kw):
            global CFG
            CFG = SimConfig(care=CareMarkov(), **kw)
            return engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]

        none = P()
        assert P(panic_drop=0.25) < none
        assert P(panic_drop=0.25, panic_cut=0.2) < P(panic_drop=0.25, panic_cut=0.8)      # 많이 팔수록 나쁘다
        assert P(panic_drop=0.25, panic_recover=0.05) < P(panic_drop=0.25, panic_recover=0.5)  # 늦게 돌아올수록 나쁘다
        assert P(panic_drop=0.0) == none                                                   # 끄면 그대로
    finally:
        CFG = old


def test_spending_rigidity_protects_lifestyle_but_hurts_essentials():
    """지출을 못 줄이면 여행·취미 충족률은 올라가지만 기본생활이 위태로워진다."""
    global CFG
    old = CFG
    try:
        CFG = SimConfig(care=CareMarkov())
        a = engine_tax.summarize(_run(couple_60()))
        CFG = SimConfig(care=CareMarkov(), lifestyle_floor=0.30)
        b = engine_tax.summarize(_run(couple_60()))
        assert b["여행·취미 평균 충족률"] > a["여행·취미 평균 충족률"]
        assert b["기본생활 유지 확률"] < a["기본생활 유지 확률"]
        assert b["CTE90 부족액(실질)"] > a["CTE90 부족액(실질)"]
    finally:
        CFG = old


def test_behaviour_defaults_are_off():
    """기본값은 '합리적 가구' — 행동 가정은 켜야 적용된다(기존 결과와의 연속성)."""
    c = SimConfig()
    assert c.panic_drop == 0.0 and c.lifestyle_floor == 0.0


# ── v37 루프 밖으로 뺀 계산들: 경로에 의존하지 않으므로 단독으로 검증할 수 있다 ──

def test_glide_path_is_linear_between_ages_and_flat_outside():
    """주식 비중 경로: 시작 나이 전엔 처음 비중, 끝 나이 후엔 목표 비중, 사이는 선형."""
    hh = replace(couple_60(), stock_weight=0.6, glide_end=0.2, glide_ages=(60, 80))
    w = engine_tax.glide_path(hh, 40)                      # 60세부터 40년
    assert w[0] == pytest.approx(0.6)                      # 60세: 아직 안 줄였다
    assert w[10] == pytest.approx(0.4)                     # 70세: 절반
    assert w[20] == pytest.approx(0.2) and w[30] == pytest.approx(0.2)   # 80세 이후 유지
    assert np.all(np.diff(w) <= 1e-12)                     # 단조 감소
    flat = engine_tax.glide_path(replace(couple_60(), stock_weight=0.45, glide_end=None), 20)
    assert np.allclose(flat, 0.45)


def test_joint_pv_falls_with_age_and_exceeds_single_life():
    """부부 연금현가: 나이가 들면 짧아지고, 둘 중 한 명이라도 살아있는 기간이므로 1인보다 길다."""
    hh = couple_60(); qx = mortality.default_qx()
    a, b = engine_tax.joint_pv(hh, qx, 0), engine_tax.joint_pv(hh, qx, 15)
    assert a > b > 0
    single = engine_tax.joint_pv(replace(hh, members=[hh.members[0]]), qx, 0)
    assert a > single                                      # 연생(last survivor) > 단생


def test_annuity_pv_ordering():
    """연금 1원의 현가는 할인율↑에 줄고, 보증기간·체증형·부부형에 늘어난다."""
    hh = couple_60(); qx = mortality.default_qx()
    f = lambda **kw: engine_tax.annuity_pv(hh, SimConfig(), AddOns(**kw), qx,
                                           [0, 1] if kw.get("annuity_joint") else [0], 0)
    base = f()
    assert f(annuity_rate=0.05) < base < f(annuity_rate=0.01)
    assert f(annuity_guarantee_years=20) > base            # 보증기간은 지급을 늘린다
    assert f(annuity_escalation=0.02) > base               # 체증형
    assert f(annuity_joint=True) > base                    # 부부형(생존자 지급)
    assert f(annuity_mort_mult=1.5) < base                 # 더 빨리 죽는다고 보면 짧다


def test_floor_table_shrinks_with_age_and_with_more_pension():
    """보호선: 남은 기간이 짧아지면 줄고, 보장소득이 많으면 줄고, 음수가 되지 않는다."""
    hh = couple_60(); e = EconomyV2.enhanced()
    tab = engine_tax.floor_table(hh, SimConfig(), e, HouseConfig(), mortality.default_qx(), 40, 3000)
    assert tab.shape == (41, 3)
    assert np.all(tab >= 0)
    assert tab[0, 0] > tab[20, 0] > tab[39, 0]             # 나이가 들수록 남은 부족분이 작다
    rich = replace(hh, members=[replace(hh.members[0], nps_monthly=400), replace(hh.members[1], nps_monthly=300)])
    tab2 = engine_tax.floor_table(rich, SimConfig(), e, HouseConfig(), mortality.default_qx(), 40, 3000)
    assert tab2[0, 0] < tab[0, 0]                          # 연금이 많으면 보호선이 낮다


def test_dependent_care_scales_with_share_and_is_zero_without_parents():
    """더블케어: 부양 부모가 없으면 0, 분담 비율에 비례, 두 명이면 한 명보다 크다."""
    hh = couple_60(); cfg = SimConfig(care=CareMarkov()); qx = mortality.default_qx()
    g = lambda deps, seed=1: engine_tax.dependent_care(
        replace(hh, dependents=deps), cfg, qx, 30, 3000, np.random.default_rng(seed), True).sum()
    assert g(None) == 0 and g([]) == 0
    assert g([{"age": 85, "sex": "F", "share": 0.0}]) == 0
    full, half = g([{"age": 85, "sex": "F"}]), g([{"age": 85, "sex": "F", "share": 0.5}])
    assert half == pytest.approx(full * 0.5)               # 같은 난수 → 정확히 절반
    assert g([{"age": 85, "sex": "F"}, {"age": 88, "sex": "M"}]) > full


def test_care_insurance_premium_follows_equivalence_principle():
    """간병보험료 = (1 + 사업비율) × 급부현가 / 납입현가. 사업비율에 선형, 급부액에 선형."""
    hh = couple_60(); cfg = SimConfig(care=CareMarkov()); qx = mortality.default_qx()
    alive, incare, severe, _, _ = engine_tax.life_paths(hh, cfg, qx, 40, 2000,
                                                        np.random.default_rng(0), True)
    p = lambda **kw: engine_tax.care_insurance_premium(hh, AddOns(care_benefit=1200, **kw),
                                                       alive, incare, severe, 40)
    base = p(); l0 = AddOns().care_loading
    assert base > 0
    assert p(care_loading=0.5) == pytest.approx(base / (1 + l0) * 1.5, rel=1e-9)   # 사업비율에 선형
    assert p(care_loading=0.0) == pytest.approx(base / (1 + l0), rel=1e-9)
    p2 = engine_tax.care_insurance_premium(hh, AddOns(care_benefit=2400), alive, incare, severe, 40)
    assert p2 == pytest.approx(base * 2, rel=1e-9)         # 급부 2배 → 보험료 2배


def test_life_paths_shapes_and_widowhood_applied():
    """생존·간병 경로의 모양과 단조성. 사별 효과는 부부에게만 적용된다."""
    hh = couple_60(); cfg = SimConfig(care=CareMarkov()); qx = mortality.default_qx()
    alive, incare, severe, lvl, slv = engine_tax.life_paths(hh, cfg, qx, 40, 2000,
                                                            np.random.default_rng(0), True)
    assert alive.shape == (2, 2000, 41) and incare.shape == alive.shape
    assert np.all(alive[:, :, 1:] <= alive[:, :, :-1])     # 한 번 죽으면 되살아나지 않는다
    assert np.all(~severe | incare)                        # 중증은 간병 상태의 부분집합
    assert np.all(~incare | alive)                         # 간병은 생존자만
    assert lvl.shape == (2, 2000) and np.all(slv > 0)
    a2, i2, s2, l2, _ = engine_tax.life_paths(hh, cfg, qx, 40, 2000,
                                              np.random.default_rng(0), False)
    assert i2 is None and s2 is None and l2 is None        # 마르코프를 끄면 간병 상태가 없다


def test_every_brand_field_reaches_the_report():
    """화이트라벨: 기관이 설정한 브랜드 값이 하나라도 조용히 무시되면 안 된다.
    (v37에서 tagline이 설정만 되고 리포트에 안 나오던 버그를 잡고 고정)"""
    from retire_sim import branding, inputs, reporting
    b = branding.Brand(name="가나다은행", tagline="타그라인표식", primary="#123456", accent="#654321",
                       ink="#111222", muted="#333444", line="#555666",
                       logo_url="https://example.com/logo.png",
                       footer="푸터표식", disclaimer="고지표식")
    html = reporting.build_report(inputs.defaults(), client_name="홍길동", advisor="김설계", brand=b)
    for f in ("name", "tagline", "primary", "accent", "ink", "muted", "line",
              "logo_url", "footer", "disclaimer"):
        assert getattr(b, f) in html, f"브랜드 '{f}' 값이 리포트에 나타나지 않는다"


def test_notebook_matches_modules():
    """노트북의 %%writefile 셀이 저장소 모듈과 같아야 한다.

    다르면 Colab에서 노트북을 돌릴 때 모듈이 옛 코드로 되돌아간다(실제로 겪은 사고).
    실패하면: python scripts/sync_notebook.py
    """
    import json
    import pathlib
    nb_path = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "retire_sim_colab.ipynb"
    if not nb_path.exists():
        pytest.skip("노트북이 없는 배포")
    root = nb_path.parent.parent
    stale = []
    for c in json.loads(nb_path.read_text())["cells"]:
        if c.get("cell_type") != "code":
            continue
        head, _, body = "".join(c["source"]).partition("\n")
        if not head.strip().startswith("%%writefile"):
            continue
        f = root / head.split(None, 1)[1].strip()
        if f.exists() and body != f.read_text():
            stale.append(str(f.relative_to(root)))
    assert not stale, "노트북이 모듈과 다릅니다 (python scripts/sync_notebook.py): " + ", ".join(stale)


def test_disabling_taxes_only_removes_taxes_not_the_pension_account():
    """세금을 끄는 것은 '세금을 안 매긴다'는 뜻이어야 한다.

    v37 전에는 tc.enabled=False가 연금계좌 인출까지 막아, 퇴직연금을 쌓는 가구는
    세금을 끄면 결과가 '나빠지는' 거꾸로 된 값이 나왔다(−19%p). 측정값을 믿으면 사고가 나는 자리.
    """
    hh = replace(couple_30(), pension_balance=1000, retirement_contrib=True)
    on = engine_tax.summarize(_run(hh))["기본생활 유지 확률"]
    off = engine_tax.summarize(_run(hh, tc=TaxConfig(overseas_share=0.2, enabled=False)))["기본생활 유지 확률"]
    assert off > on, "세금을 끄면 좋아져야 한다 (연금계좌가 함께 잠기면 안 된다)"


# ── v38 앱 연기 테스트: 앱을 실제로 돌려 본다 ──
# 지금까지 앱은 '버전이 맞는가'만 검사했다. 그래서 배포한 뒤에야 깨진 걸 알았다(NPSRules 사고).

def _app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import pathlib
    return AppTest.from_file(str(pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py"),
                             default_timeout=600)


def test_app_runs_and_example_household_reaches_results():
    """앱이 끝까지 돌고, '예시 가구로 보기'가 결과 화면까지 간다."""
    at = _app(); at.run()
    assert not at.exception, [e.value for e in at.exception]
    btn = [b for b in at.button if "60대" in b.label]
    assert btn, "예시 가구 버튼이 없습니다"
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join(m.value for m in at.markdown)
    assert "지금 계획대로면" in md


def test_app_whatif_draws_the_ab_comparison_chart():
    """'바꿔보기'에서 조건을 바꾸면 지금 계획 vs 바꾼 뒤 그래프가 그려진다."""
    at = _app(); at.run()
    [b for b in at.button if "60대" in b.label][0].click().run()
    p = at.session_state.get("p", {})
    at.session_state["tweak"] = dict(rd=0, sd=0, sm=-10, ns=p.get("nps_start", 65),
                                     pf=p.get("profile", "균형형"))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join(m.value for m in at.markdown)
    assert "<svg" in md and "지금 계획" in md, "A/B 비교 그래프가 그려지지 않았습니다"


def test_changelog_covers_current_version():
    """CHANGELOG 맨 위 항목이 현재 모델 버전이어야 한다.

    v37·v38에서 README만 쓰고 CHANGELOG를 빠뜨려, 저장소의 변경 내역이 v36에 멈춰 있었다.
    버전을 올릴 때 기록도 같이 올리도록 여기서 막는다.
    """
    import pathlib
    import re
    from retire_sim import __version__
    path = pathlib.Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    if not path.exists():
        pytest.skip("CHANGELOG 없는 배포")
    heads = re.findall(r"^## v(\d+(?:\.\d+)?)", path.read_text(), re.M)
    assert heads, "CHANGELOG에 '## vNN' 항목이 없습니다"
    assert int(float(heads[0])) == int(__version__), (
        f"CHANGELOG 맨 위가 v{heads[0]} 인데 모델은 v{__version__} — CHANGELOG.md 에 이번 버전 항목을 추가하세요")


# ── v39 상품 선택 vs 개시 나이: 실명 상품 없이 공시 범위로 재는 비교 ──

def _spread_setup(age=45, pay_years=20):
    from retire_sim.economy_v2 import generate
    from retire_sim.product import AnnuityProduct
    ec = ECO
    eco = generate(ec, 60, 3000, np.random.default_rng(42))
    prod = AnnuityProduct(monthly_premium=50, pay_years=pay_years, start_age=65,
                          credit_spread=0.023 - (ec.q0 + ec.pi0), guarantee_years=10)
    return prod, [Person(age=age, sex="M")], mortality.default_qx(), eco


def test_choice_spread_compares_only_ages_after_premiums_end():
    """개시 나이 비교는 납입이 끝난 뒤만. 납입 중에 개시하면 '덜 내는 효과'가 섞인다."""
    from retire_sim import product as prod_mod
    prod, me, qx, eco = _spread_setup(age=45, pay_years=20)       # 납입 종료 65세
    out = prod_mod.choice_spread(prod, me, qx, eco, 0.4, start_ages=(60, 65, 70, 75))
    assert set(out["개시 나이"]) == {65, 70, 75}, "납입 중(60세) 개시가 섞이면 폭이 부풀려진다"
    assert all(out["개시 나이"][a] < out["개시 나이"][b]
               for a, b in ((65, 70), (70, 75))), "늦게 받을수록 연금액이 커야 한다"


def test_cheaper_product_pays_more_but_start_age_matters_far_more():
    """사업비가 싼 상품이 더 주지만, 그 폭보다 개시 나이의 폭이 훨씬 크다 — 이 모델의 핵심 메시지."""
    from retire_sim import product as prod_mod
    prod, me, qx, eco = _spread_setup()
    out = prod_mod.choice_spread(prod, me, qx, eco, 0.4, start_ages=(65, 70, 75))
    assert out["상품 선택"]["싼 쪽"] > out["상품 선택"]["비싼 쪽"]
    cost, age = out["폭"]["상품 선택"], out["폭"]["개시 나이"]
    assert 0 < cost < 0.15, f"공시 사업비 범위의 폭이 비현실적입니다: {cost:.1%}"
    assert age > 3 * cost, f"개시 나이 폭 {age:.1%} vs 상품 선택 폭 {cost:.1%}"


def test_app_product_tab_shows_choice_spread():
    """'상품 추가' 탭에서 상품 비교를 돌리면 '상품 선택 vs 개시 나이' 블록이 그려진다."""
    at = _app(); at.run()
    [b for b in at.button if "30대" in b.label][0].click().run()
    btn = [b for b in at.button if b.key == "pr_go"]
    assert btn, "상품 비교 버튼을 찾지 못했습니다"
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join(m.value for m in at.markdown)
    assert "상품을 잘 고르는 것" in md and "보험연구원" in md


def test_sources_doc_matches_registry():
    """docs/SOURCES.md 는 retire_sim/sources.py 에서 만든다 — 손으로 고쳐 어긋나면 안 된다.

    실패하면: python scripts/make_sources_doc.py
    """
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_mk", root / "scripts" / "make_sources_doc.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = root / "docs" / "SOURCES.md"
    assert out.exists(), "docs/SOURCES.md 가 없습니다 — python scripts/make_sources_doc.py"
    assert out.read_text() == mod.build(), "docs/SOURCES.md 가 오래됐습니다 — python scripts/make_sources_doc.py"


def test_every_source_entry_is_complete():
    """출처 항목에 빈칸이 있으면 갱신할 때 어디서 뭘 받아야 할지 알 수 없다."""
    from retire_sim.sources import SOURCES
    assert len(SOURCES) >= 10
    for s in SOURCES:
        for f in ("name", "org", "used_for", "as_of", "cycle", "where", "how", "stale"):
            assert getattr(s, f).strip(), f"{s.key} 의 {f} 가 비었습니다"
        # 갱신 절차는 '어디를 고치면 되는지'가 들어 있어야 쓸모가 있다
        assert any(k in s.how for k in ("retire_sim/", "scripts/", "config.", "노트북", "앱의")), \
            f"{s.key} 의 갱신 방법에 고칠 대상이 없습니다: {s.how!r}"
    keys = [s.key for s in SOURCES]
    assert len(keys) == len(set(keys)), "출처 key 가 중복됩니다"


def test_docs_do_not_contradict_mortality_default():
    """README가 '미래 수명 연장 미반영'이라고 적혀 있었는데 모델은 기본으로 반영한다.

    출처·가정을 여러 문서에 베껴 쓰면 생기는 어긋남이라, 이 한 건만이라도 고정한다.
    (자료 출처 자체는 retire_sim/sources.py 한 곳에서만 관리한다.)
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    if not SimConfig().mortality_improvement:
        pytest.skip("기본값이 바뀌었으면 이 검사도 바뀌어야 한다")
    stale = ["미래 수명 연장 미반영", "미래 사망률 개선 미반영",
             "no future improvement", "period life table (no"]
    for f in ("README.md", "README.en.md", "docs/MODEL.md", "docs/MODEL.en.md"):
        path = root / f
        if not path.exists():
            continue
        text = path.read_text()
        hit = [x for x in stale if x in text]
        assert not hit, f"{f} 가 모델 기본값(코호트 개선 반영)과 어긋납니다: {hit}"


def test_app_explanation_tabs_are_customer_and_advisor_facing():
    """'이 계산에 대하여'(고객용)와 '설명하기'(설계사용)가 모두 그려지고,
    자료 출처 표가 레지스트리에서 나온다."""
    at = _app(); at.run()
    [b for b in at.button if "60대" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join(m.value for m in at.markdown)
    for k in ("이 계산은 무엇을 하나요", "자료는 어디서 가져왔나요",
              "고객에게 설명하실 때", "하지 마셔야 할 말"):
        assert k in md, f"'{k}' 가 화면에 없습니다"
    from retire_sim.sources import SOURCES
    joined = " ".join(str(d.value.to_dict()) for d in at.dataframe)
    assert SOURCES[0].org in joined, "자료 출처 표가 레지스트리에서 나오지 않았습니다"


def test_explanation_tabs_avoid_jargon():
    """설명 화면에서 어려운 전문용어를 쓰지 않는다 (50~60대 독자 기준)."""
    import pathlib
    import re
    src = (pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py").read_text()
    m = re.search(r"with tab5:(.*?)\nwith tab7:", src, re.S)
    assert m, "'이 계산에 대하여' 탭을 찾지 못했습니다"
    banned = ["확실성등가", "위험회피계수", "CRRA", "소득인정액", "연계감액", "코호트",
              "수지상등", "역선택", "기간프리미엄", "몬테카를로", "백분위"]
    found = [b for b in banned if b in m.group(1)]
    assert not found, f"고객용 설명에 어려운 말이 있습니다: {found}"


def test_no_nan_leaks_into_results():
    """결과 배열에 NaN·무한대가 섞이면 요약값이 조용히 망가진다.

    '해당 경로에 그 사건이 없었다'는 뜻으로 일부러 NaN 을 두는 배열만 예외로 둔다.
    (예: 상속세는 아무도 돌아가시지 않은 경로에서 NaN)
    """
    res = _run(couple_60())
    by_design = {"estate_tax_real", "transfer_real", "estate_real",
                 "life_ratio", "stock_ret", "port_ret", "debt_retire"}
    bad = {k: int((~np.isfinite(v)).sum()) for k, v in res.items()
           if isinstance(v, np.ndarray) and v.dtype.kind == "f"
           and k not in by_design and not np.isfinite(v).all()}
    assert not bad, f"NaN/inf 가 섞인 결과 배열: {bad}"
    # 일부러 둔 NaN 도 '정말 그 사건이 없는 경로'에서만 나와야 한다
    nan_at = ~np.isfinite(res["estate_real"])
    if nan_at.any():
        assert res["hh_alive"][nan_at, -1].all(), \
            "상속세가 NaN 인데 가구가 이미 사망한 경로가 있습니다 — 계산이 빠진 자리입니다"
    for k, v in engine_tax.summarize(res).items():
        assert not (isinstance(v, float) and not np.isfinite(v)), f"요약값 {k} 가 {v}"
