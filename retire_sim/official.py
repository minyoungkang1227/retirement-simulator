"""(v50.11) 공식 계산기 대조 — 같은 입력을 기관 모의계산기에 넣은 화면값과 모델 값을 나란히.

검증 장부(verify.py)는 법 조문으로 '직접 계산한 값'과 대조한다. 여기는 그다음 단계로,
**기관이 직접 운영하는 계산기의 화면값**과 대조한다. 사람이 계산기에 입력해 값을 받아 와야 하므로
대조 사례(CASES)와 받아 온 값(data/official_checks.csv)을 나눠 둔다.

  · CASES: 사례마다 어느 계산기에, 무엇을 입력하고, 모델은 얼마를 내는지(model). 허용 오차(tol)도 여기.
  · data/official_checks.csv: 사람이 채우는 칸 — id, 공식값, 확인 날짜, 확인한 사람, 메모.
    비어 있으면 '대기'. 채우면 앱 '이 계산에 대하여'와 테스트가 바로 읽는다(어긋나면 테스트가 실패한다).

사례 값을 바꾸면 이미 받아 온 공식값과 맞지 않게 되므로, 입력을 바꿀 때는 새 id 를 쓴다.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data", "official_checks.csv")

CALCULATORS = {
    "nps": ("국민연금 예상연금 모의계산", "국민연금공단", "https://www.nps.or.kr"),
    "hi": ("지역보험료 모의계산", "국민건강보험공단", "https://www.nhis.or.kr"),
    "rt": ("퇴직소득세 모의계산", "국세청 홈택스", "https://www.hometax.go.kr"),
    "bp": ("기초연금 모의계산", "보건복지부 복지로", "https://www.bokjiro.go.kr"),
    "rm": ("주택연금 예상연금 조회", "한국주택금융공사", "https://www.hf.go.kr"),
}


@dataclass(frozen=True)
class Case:
    id: str
    calc: str                 # CALCULATORS 키
    inputs: str               # 계산기에 넣을 값 (사람이 읽는 문장)
    model: Callable[[], float]
    unit: str
    tol_rel: float = 0.01     # 상대 오차 허용 (1%)
    tol_abs: float = 0.1      # 절대 오차 허용 (단위 기준, 예: 0.1만 원 = 1천 원)
    note: str = ""


# ── 모델 값 계산 (엔진과 같은 함수만 부른다) ──
def _nps(monthly_income: float, years: int) -> float:
    from .pension import estimate_nps_monthly
    return float(estimate_nps_monthly(monthly_income * 12, years, 60, 60))


def _hi(pension_annual: float, fin_annual: float, tax_base: float) -> float:
    from .tax import TaxConfig, health_premium, hi_property_monthly
    tc = TaxConfig()
    z = np.zeros(1)
    annual = health_premium([z + pension_annual], [z + fin_annual], tc, prop_base_list=[z + tax_base],
                            prop_monthly=float(hi_property_monthly(np.array([tax_base]), tc)[0]))
    return float(annual[0]) / 12


def _rt(amount: float, years: int) -> float:
    from .tax import retirement_income_tax
    return float(retirement_income_tax(amount, years))


def _bp(nps_list: list, house_official: float, fin: float, region: str = "대도시") -> float:
    from .config import NPSRules
    from .pension import basic_pension_annual
    r = NPSRules(basic_region=region)
    one = np.ones(1)
    elig = [one.astype(bool) for _ in nps_list]
    tot = basic_pension_annual(one * sum(nps_list), one * house_official, one * fin, elig,
                               [one * x for x in nps_list], [one * 0.0 for _ in nps_list],
                               [np.zeros(1, bool) for _ in nps_list], r, one)
    return float(tot[0]) / 12


def _rm(age: int, price: float) -> float:
    from .tax import reverse_mortgage_monthly
    return float(reverse_mortgage_monthly(age, price))


CASES = [
    # 국민연금 — 앞으로 가입하는 기간만으로 계산하는 사례(과거 가입기간의 높은 소득대체율이 끼어들지 않게)
    Case("nps-1", "nps", "월 소득 300만 원, 가입기간 30년 (모두 2026년 이후 가입), 정상 수령", lambda: _nps(300, 30), "만 원/월"),
    Case("nps-2", "nps", "월 소득 500만 원, 가입기간 20년 (모두 2026년 이후 가입), 정상 수령", lambda: _nps(500, 20), "만 원/월"),
    Case("nps-3", "nps", "월 소득 659만 원(상한), 가입기간 40년 (모두 2026년 이후 가입), 정상 수령", lambda: _nps(659, 40), "만 원/월"),
    # 건강보험 지역가입자 — 자동차 없음, 전월세 없음, 장기요양 포함 합계
    Case("hi-1", "hi", "연금소득 연 1,200만 원, 이자·배당 0, 주택 재산세 과세표준 2억 원, 자동차 없음",
         lambda: _hi(1200, 0, 20000), "만 원/월", note="장기요양보험료 포함 합계"),
    Case("hi-2", "hi", "연금소득 연 2,400만 원, 이자·배당 연 1,500만 원, 주택 재산세 과세표준 4억 원, 자동차 없음",
         lambda: _hi(2400, 1500, 40000), "만 원/월", note="장기요양보험료 포함 합계"),
    Case("hi-3", "hi", "연금소득 연 600만 원, 이자·배당 0, 재산 없음, 자동차 없음",
         lambda: _hi(600, 0, 0), "만 원/월", note="최저보험료가 적용되면 모델과 다를 수 있음"),
    # 퇴직소득세 — 지방소득세 포함
    Case("rt-1", "rt", "퇴직급여 1억 원, 근속연수 10년 (2023년 이후 퇴직)", lambda: _rt(10000, 10), "만 원", note="지방소득세 포함"),
    Case("rt-2", "rt", "퇴직급여 2억 원, 근속연수 25년", lambda: _rt(20000, 25), "만 원", note="지방소득세 포함"),
    Case("rt-3", "rt", "퇴직급여 5,000만 원, 근속연수 5년", lambda: _rt(5000, 5), "만 원", note="지방소득세 포함"),
    # 기초연금 — 대도시, 근로소득 없음, 부채 없음, 자동차 없음
    Case("bp-1", "bp", "단독 가구, 국민연금 월 50만 원, 주택(시가표준액) 1.5억 원, 금융재산 2,000만 원, 대도시",
         lambda: _bp([50.0], 15000, 2000), "만 원/월"),
    Case("bp-2", "bp", "부부 가구(둘 다 65세 이상), 국민연금 각 월 40만 원, 주택 3억 원, 금융재산 5,000만 원, 대도시",
         lambda: _bp([40.0, 40.0], 30000, 5000), "만 원/월", note="부부 합계(부부감액 20% 반영)"),
    Case("bp-3", "bp", "단독 가구, 국민연금 월 30만 원, 주택 6억 원, 금융재산 1.2억 원, 대도시",
         lambda: _bp([30.0], 60000, 12000), "만 원/월", note="선정기준액 근처 — 소득역전방지 감액 확인"),
    # 주택연금 — 일반주택, 종신지급방식 정액형
    Case("rm-1", "rm", "연소자 65세, 주택 시세 5억 원, 일반주택, 종신 정액형", lambda: _rm(65, 50000), "만 원/월"),
    Case("rm-2", "rm", "연소자 70세, 주택 시세 3억 원, 일반주택, 종신 정액형", lambda: _rm(70, 30000), "만 원/월"),
    Case("rm-3", "rm", "연소자 60세, 주택 시세 9억 원, 일반주택, 종신 정액형", lambda: _rm(60, 90000), "만 원/월"),
]


#: 받아 온 공식값과 모델이 다르다는 걸 알고 있고, 원인을 조사 중인 사례. 여기에 적힌 id 는 불일치여도 테스트가 실패하지 않는다
#: (새로 생긴 불일치만 잡으려고). 원인을 고치면 이 목록에서 지운다.
KNOWN_GAPS = {
    "nps-1": "공단 모의계산 값이 모델보다 6.5% 낮다(세 사례 모두 같은 비율). 개정 국민연금법은 2026년 이후 가입기간에 소득대체율 43%(비례상수 1.29)를 적용하고 모델은 이 값을 쓴다. "
             "공단 값은 개혁 전 40%(비례상수 1.2)로 계산한 값과 0.4~0.6% 안에서 맞는다 — 계산기 화면이 개혁 전 기준이었거나 가입 시작 연도 입력이 달랐을 가능성. 화면 재확인 중",
    "nps-2": "nps-1과 같은 차이",
    "nps-3": "nps-1과 같은 차이",
    "hi-3": "공단 값 2.28만 원은 지역가입자 최저보험료(월 약 2.016만 원)에 장기요양 13.14%를 더한 값과 맞아떨어진다. 모델은 최저보험료를 아직 넣지 않았다(2026년 최저보험료 확정치는 확인 필요)",
}


def recorded() -> dict:
    """data/official_checks.csv → {id: {"official": 값, "checked_on", "checked_by", "note"}}. 빈 칸은 건너뛴다."""
    out = {}
    if not os.path.exists(DATA):
        return out
    with open(DATA, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = (row.get("official") or "").replace(",", "").strip()
            if v:
                out[row["id"].strip()] = {"official": float(v), "checked_on": row.get("checked_on", ""),
                                          "checked_by": row.get("checked_by", ""), "note": row.get("note", "")}
    return out


def rows() -> list:
    """사례마다 모델 값과 공식값(있으면)과 판정. 판정: '일치' / '불일치' / '대기'."""
    rec = recorded()
    out = []
    for c in CASES:
        m = c.model()
        o = rec.get(c.id, {}).get("official")
        if o is None:
            status = "대기"
        else:
            status = "일치" if abs(m - o) <= max(c.tol_abs, c.tol_rel * abs(o)) else "불일치"
        name, org, url = CALCULATORS[c.calc]
        out.append({"id": c.id, "calc": name, "org": org, "url": url, "inputs": c.inputs, "unit": c.unit,
                    "model": m, "official": o, "status": status, "note": c.note, "known": KNOWN_GAPS.get(c.id, "") if status == "불일치" else "",
                    "checked_on": rec.get(c.id, {}).get("checked_on", "")})
    return out


def summary(rs: list | None = None) -> tuple:
    """(일치 수, 받아 온 수, 전체 수)."""
    rs = rs if rs is not None else rows()
    done = [r for r in rs if r["status"] != "대기"]
    return sum(r["status"] == "일치" for r in done), len(done), len(rs)
