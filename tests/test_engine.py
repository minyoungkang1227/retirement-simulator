"""회귀 테스트: 코드를 고쳐도 대표 가구의 결과가 크게 달라지지 않는지 자동 확인.
실행: pytest -q    (커밋할 때 GitHub Actions가 자동 실행)
"""
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
    (couple_60, "기본생활 유지 확률", 0.46, 0.56),
    (couple_30, "기본생활 유지 확률", 0.77, 0.87),
    (couple_30, "여행·취미 평균 충족률", 0.85, 0.97),
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
