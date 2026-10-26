"""검증 장부 — 이 모델의 제도 계산을 **법령·공식 고시값에서 손으로 뽑은 값**과 나란히 둔 목록.

왜 따로 두나: 세금·건보료·연금 계산은 테스트로 손계산과 대조해 왔지만(v44·v45), 그 근거는 테스트 파일 안에만
있어 고객·설계사가 보는 화면에는 "세법을 단순하게 만들었다"는 말뿐이었다. 같은 목록을 화면과 테스트가
**함께 읽게** 하면, 화면에 적힌 "N개 중 N개 일치"는 테스트가 통과하는 한 거짓일 수 없다.

항목을 더하는 법: 아래 CHECKS 에 한 줄을 넣는다. `hand` 는 모델 코드를 부르지 않고 **법령 구간·고시값만으로**
적은 식이어야 한다(모델을 다시 돌려 맞춘 값은 검증이 아니다). 테스트(test_verify_*)가 자동으로 집어 간다.

`kind` 로 근거의 종류를 구분한다 — 법령 손계산 / 공식 고시값 / 닫힌식. 공단 '모의계산기' 화면값과의
대조는 아직 없다(국내 접속이 필요해 남겨 둠). 화면에도 그렇게 적는다.
(v50.1) v50에서 생긴 근로소득공제·근로소득세액공제·전월세 보증금 반영 비율을 더했다.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Check:
    area: str                       # 영역
    item: str                       # 무엇을 맞춰 봤나 (사람 말)
    basis: str                      # 근거 (법령 조항·고시)
    kind: str                       # 법령 손계산 | 공식 고시값 | 닫힌식
    source: str                     # retire_sim.sources 의 key
    hand_how: str                   # 손계산 식 (사람이 읽는 꼴)
    hand: float                     # 손으로 산출한 기대값
    model: Callable[[], float]      # 모델이 같은 입력에서 낸 값
    unit: str = "만 원"
    tol: float = 1e-6               # 상대 허용오차 (절대 1e-9 바닥)


def _a(x):
    return np.array([float(x)])


def _f(v):
    return float(np.asarray(v).ravel()[0])


# ── 모델 쪽 호출 (지연 import — 앱 첫 화면을 무겁게 하지 않으려고) ──
def _tax():
    from retire_sim import tax
    return tax


def _pension():
    from retire_sim import pension
    return pension


def analytic_rows(years: int = 12):
    """해석해 대조: 변동성·세금·사망·간병을 모두 끈 결정론적 극한에서
    W_{t+1} = (W_t − 지출) × e^R 와 엔진이 같은지. (연차, 모형, 닫힌식, 차이) 목록."""
    from retire_sim import engine_tax
    from retire_sim.config import CareMarkov, Household, NPSRules, Person, SimConfig
    from retire_sim.economy_v2 import EconomyV2
    from retire_sim.tax import HouseConfig, TaxConfig

    R = 0.03
    eco = EconomyV2(r0=R, r_theta=R, r_sigma=0.0, q0=R, q_theta=R, q_sigma=0.0,
                    pi0=0.0, pi_theta=0.0, pi_sigma=0.0, erp=0.0, s_sigma=0.0,
                    fee=0.0, jump_lambda=0.0, erp_sd=0.0, pi_theta_sd=0.0, q_theta_sd=0.0,
                    antithetic=False, term_premium=0.0,
                    fisher=True, house_real_sd=0.0, house_sigma=0.0)
    qx = {"M": np.zeros(111), "F": np.zeros(111)}            # 아무도 죽지 않음
    qx["M"][100] = 1.0; qx["F"][100] = 1.0
    spend = 3000.0
    hh = Household(members=[Person(60, "M")], liquid_assets=50_000, stock_weight=1.0,
                   annual_spending=spend, essential=spend, lifestyle=0.0, legacy_target=0.0,
                   survivor_spending_ratio=1.0, floor_method="fixed95")
    cfg = SimConfig(n_paths=64, seed=1, care=CareMarkov(enabled=False), mortality_improvement=False,
                    longevity_sigma_level=0.0, longevity_sigma_trend=0.0, widow_excess=(0.0, 0.0),
                    nps=NPSRules(basic_on=False, survivor_on=False))
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), house=HouseConfig(), qx_table=qx)
    W = res["W_real"][0]                                     # 물가 0 → 실질 = 명목
    w, rows = float(hh.liquid_assets), []
    for t in range(years):
        w = (w - spend) * np.exp(R)
        rows.append((t + 1, float(W[t + 1]), w, float(W[t + 1]) - w))
    return rows


def _compound(w, spend, r, years):
    """닫힌식 반복: 해마다 (w − 지출) × e^r. 엔진을 쓰지 않는 손계산 쪽."""
    for _ in range(years):
        w = (w - spend) * np.exp(r)
    return w


_B = 34.97        # 기초연금 기준연금액 2026 (월, 만원) — 복지부 고시
_HI_PT = 211.5    # 재산보험료 점수당 금액(원) — 시행령 제44조제2항

CHECKS = (
    # ── 소득세 (소득세법 제55조 세율표, 누진공제) ──
    Check("소득세", "과세표준 3,000만 원", "소득세법 제55조", "법령 손계산", "tax",
          "3,000 × 15% − 누진공제 126", 3000 * 0.15 - 126,
          lambda: _f(_tax().progressive(_a(3000), _tax().BRACKETS))),
    Check("소득세", "과세표준 1억 원", "소득세법 제55조", "법령 손계산", "tax",
          "10,000 × 35% − 누진공제 1,544", 10000 * 0.35 - 1544,
          lambda: _f(_tax().progressive(_a(10000), _tax().BRACKETS))),
    Check("소득세", "과세표준 20억 원 (최고구간)", "소득세법 제55조", "법령 손계산", "tax",
          "200,000 × 45% − 누진공제 6,594", 200000 * 0.45 - 6594,
          lambda: _f(_tax().progressive(_a(200000), _tax().BRACKETS))),
    # ── 연금소득 ──
    Check("연금소득", "연금소득공제 — 연금 1,400만 원", "소득세법 제47조의2", "법령 손계산", "tax",
          "350 + 350 × 40% + 700 × 20%", 350 + 0.4 * 350 + 0.2 * 700,
          lambda: _f(_tax().pension_deduction(_a(1400)))),
    Check("연금소득", "연금소득공제 한도 — 연금 1억 원", "소득세법 제47조의2", "법령 손계산", "tax",
          "공제 한도 900", 900.0,
          lambda: _f(_tax().pension_deduction(_a(10000)))),
    Check("연금소득", "사적연금 1,500만 원 (65세) 분리과세", "소득세법 제129조·제64조의4", "법령 손계산", "tax",
          "1,500 × 5.5%", 1500 * 0.055,
          lambda: _f(_tax().private_pension_tax(_a(1500), np.array([65])))),
    Check("연금소득", "사적연금 1,501만 원 — 한도를 넘으면 전액 16.5%", "소득세법 제64조의4", "법령 손계산", "tax",
          "1,501 × 16.5%", 1501 * 0.165,
          lambda: _f(_tax().private_pension_tax(_a(1501), np.array([65])))),
    Check("연금소득", "사적연금 1,000만 원 (75세)", "소득세법 제129조", "법령 손계산", "tax",
          "1,000 × 4.4%", 1000 * 0.044,
          lambda: _f(_tax().private_pension_tax(_a(1000), np.array([75])))),
    Check("퇴직소득", "퇴직급여 1억 원 · 근속 10년", "소득세법 제48조·제55조", "법령 손계산", "retirement_tax",
          "근속공제 1,500 → 환산 (10,000−1,500)×12/10 = 10,200 → 환산공제 6,170 + 200×45% = 6,260 "
          "→ 과표 3,940 → (3,940×15% − 126) × 10/12 × 1.1",
          (3940 * 0.15 - 126) * 10 / 12 * 1.1,
          lambda: _f(_tax().retirement_income_tax(_a(10000), _a(10)))),
    Check("퇴직소득", "퇴직급여 3,000만 원 · 근속 30년 — 세금 없음", "소득세법 제48조", "법령 손계산", "retirement_tax",
          "근속공제 4,000 + 300×10 = 7,000 > 3,000 → 0", 0.0,
          lambda: _f(_tax().retirement_income_tax(_a(3000), _a(30)))),
    Check("근로소득", "근로소득공제 — 총급여 2,400만 원", "소득세법 제47조", "법령 손계산", "earned_income",
          "750 + (2,400 − 1,500) × 15%", 750 + 900 * 0.15,
          lambda: _f(_tax().earned_income_deduction(_a(2400)))),
    Check("근로소득", "근로소득공제 — 총급여 6,000만 원", "소득세법 제47조", "법령 손계산", "earned_income",
          "1,200 + (6,000 − 4,500) × 5%", 1200 + 1500 * 0.05,
          lambda: _f(_tax().earned_income_deduction(_a(6000)))),
    Check("근로소득", "근로소득세액공제 한도 — 총급여 5,000만 원·산출세액 300만 원", "소득세법 제59조", "법령 손계산",
          "earned_income", "min(130×55% + 170×30%, max(74 − 1,700×0.008, 66)) = 66", 66.0,
          lambda: _f(_tax().earned_income_credit(_a(300), _a(5000)))),
    Check("건보료", "전월세 보증금 반영 비율", "국민건강보험법 시행령 별표 4 · 공단 안내", "법령 손계산", "hi_jeonse",
          "보증금의 30%를 재산과표에 더함", 0.30,
          lambda: float(_tax().HI_JEONSE_RATIO), unit="비율"),
    Check("금융소득", "이자·배당 1,000만 원 (분리과세)", "소득세법 제14조·제129조", "법령 손계산", "tax",
          "1,000 × 14% × 1.1(지방세)", 1000 * 0.14 * 1.1,
          lambda: _f(_tax().income_tax_person(_a(0), _a(1000), np.array([65]), _tax().TaxConfig()))),
    # ── 건강보험료 재산분 (시행령 별표 4) ──
    Check("건보료", "재산과표 1억 원 — 기본공제 이하", "국민건강보험법 시행령 별표 4", "법령 손계산", "hi_property",
          "1억 − 공제 1억 = 0 → 0점", 0.0,
          lambda: _f(_tax().hi_property_points(_a(10000))), unit="점"),
    Check("건보료", "공제 후 450만 원 (1등급)", "국민건강보험법 시행령 별표 4", "법령 손계산", "hi_property",
          "1등급", 22.0,
          lambda: _f(_tax().hi_property_points(_a(10450))), unit="점"),
    Check("건보료", "재산과표 2억 원 (18등급)", "국민건강보험법 시행령 별표 4", "법령 손계산", "hi_property",
          "공제 후 1억 → 18등급", 439.0,
          lambda: _f(_tax().hi_property_points(_a(20000))), unit="점"),
    Check("건보료", "재산과표 3억 원 (24등급)", "국민건강보험법 시행령 별표 4", "법령 손계산", "hi_property",
          "공제 후 2억 → 24등급", 586.0,
          lambda: _f(_tax().hi_property_points(_a(30000))), unit="점"),
    Check("건보료", "재산과표 2억 원 — 건강보험 재산분 월액", "시행령 제44조제2항 (점수당 211.5원)", "법령 손계산",
          "hi_property", "439점 × 211.5원", 439 * _HI_PT / 10000,
          lambda: _f(_tax().hi_property_points(_a(20000))) * _tax().HI_POINT_VALUE / 10000, unit="만 원/월"),
    # ── 부동산 ──
    Check("부동산", "1주택 12억 원 양도 — 비과세", "소득세법 제89조", "법령 손계산", "tax",
          "12억 이하 1세대 1주택 → 0", 0.0,
          lambda: _f(_tax().one_house_cgt(_a(120000), 50000, 10))),
    Check("부동산", "1주택 20억 원 양도 (취득 5억, 15년 보유)", "소득세법 제89조·제95조", "법령 손계산", "tax",
          "차익 15억 × (8억/20억) × (1 − 장기보유 80%) × 38.5%",
          (200000 - 50000) * (200000 - 120000) / 200000 * (1 - 0.8) * 0.385,
          lambda: _f(_tax().one_house_cgt(_a(200000), 50000, 15))),
    Check("부동산", "취득세율 — 6억 원 주택", "지방세법 제11조", "법령 손계산", "tax",
          "1% + 지방교육세 0.1%", 0.011,
          lambda: _f(_tax().acquisition_tax_rate(_a(60000))), unit="비율"),
    Check("부동산", "종부세 — 1주택 공시 11억 원", "종합부동산세법 제8조", "법령 손계산", "tax",
          "1주택 공제 12억 이하 → 0", 0.0,
          lambda: _f(_tax().comprehensive_property_tax(_a(110000), n_houses=1, age=60, years_held=10))),
    Check("상속·증여", "증여세 — 공제 후 2억 원", "상속세 및 증여세법 제26조·제56조", "법령 손계산", "tax",
          "20,000 × 20% − 누진공제 1,000", 20000 * 0.2 - 1000,
          lambda: _f(_tax().gift_tax(_a(20000)))),
    # ── 연금 ──
    Check("국민연금", "A값 (전체 가입자 평균소득월액)", "국민연금공단 예상연금 계산기 2026", "공식 고시값", "nps",
          "3,193,511원", 319.3511,
          lambda: float(_pension().NPS_A), unit="만 원/월"),
    Check("국민연금", "유족연금 지급률 — 가입 20년 이상", "국민연금법 제74조", "법령 손계산", "nps",
          "기본연금액의 60%", 0.6,
          lambda: float(_pension().survivor_pct(25)), unit="비율"),
    Check("기초연금", "국민연금 월 50만 원 — 감액 없음", "기초연금법 제5조", "법령 손계산", "basic_pension",
          "50 ≤ 기준연금액 × 150% → 34.97 전액", _B,
          lambda: _f(_pension().basic_pension_one(50.0, 0.0, _B)), unit="만 원/월"),
    Check("기초연금", "국민연금 월 100만·A급여 30만 원 — 연계감액", "기초연금법 제5조 (시행령 별표 1)", "법령 손계산",
          "basic_pension", "min(34.97, max(34.97×2.5 − 100, 34.97 − ⅔×30 + 34.97÷2))",
          max(_B * 2.5 - 100, _B - 2 / 3 * 30 + _B / 2),
          lambda: _f(_pension().basic_pension_one(100.0, 30.0, _B)), unit="만 원/월"),
    # ── 엔진 자체 ──
    Check("계산 엔진", "변동성·세금·사망을 모두 끈 극한 — 12년 뒤 잔고", "복리 닫힌식 (W − 지출) × e^0.03", "닫힌식",
          "rate_cpi", "5억에서 해마다 3,000만 원 쓰고 3% 연속복리, 12번 반복",
          _compound(50000.0, 3000.0, 0.03, 12),
          lambda: analytic_rows(12)[-1][1]),
)


def run_checks():
    """장부 전체를 돌려 행 목록을 돌려준다. 화면·테스트·점검기가 모두 이것을 쓴다."""
    out = []
    for c in CHECKS:
        try:
            got, err = float(c.model()), ""
        except Exception as e:                               # 한 항목이 깨져도 장부는 끝까지 보인다
            got, err = float("nan"), f"{type(e).__name__}: {e}"
        ok = bool(np.isfinite(got) and abs(got - c.hand) <= max(c.tol * abs(c.hand), 1e-9))
        out.append({"area": c.area, "item": c.item, "basis": c.basis, "kind": c.kind, "source": c.source,
                    "hand_how": c.hand_how, "hand": c.hand, "model": got, "unit": c.unit, "ok": ok, "error": err})
    return out


def summary(rows=None):
    """(일치 수, 전체 수, 영역 목록)."""
    rows = run_checks() if rows is None else rows
    areas = list(dict.fromkeys(r["area"] for r in rows))
    return sum(r["ok"] for r in rows), len(rows), areas
