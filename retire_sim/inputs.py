"""입력 변환: 사람이 채우는 입력(dict)을 엔진이 받는 설정 객체로 바꾼다.

왜 라이브러리에 있는가
  이 로직은 원래 Streamlit 앱 안에 있었다. 그러면 화면을 거치지 않고는 계산을 돌릴 수 없어서
  기업이 자기 앱에 붙이거나(API) 배치로 돌리는 길이 막힌다. 업무 규칙은 화면이 아니라
  라이브러리에 있어야 한다. 앱과 API가 **같은 함수**를 쓰므로 둘의 결과가 어긋날 수 없다.

입력(dict)의 단위
  금액은 억 원(자산·유산)과 만 원(생활비·연금·저축)이 섞여 있다. 사람이 채우기 쉬운 단위를
  그대로 받고, 여기서 엔진의 단위(만 원)로 환산한다.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import mortality
from .config import AddOns, CareMarkov, CareShock, Household, NPSRules, Person, SimConfig
from .economy_v2 import EconomyV2
from .pension import estimate_nps_monthly
from .tax import HouseConfig, TaxConfig

#: 투자 성향 → 주식 비중
PROFILES = {"보수형": 0.3, "균형형": 0.5, "성장형": 0.7}

#: 국민연금연구원 「2024년 국민노후보장패널조사」 50세 이상이 생각하는 노후 생활비(월, 만 원)
AVG_SPEND = {"부부_최소": 217, "부부_적정": 298, "개인_최소": 139, "개인_적정": 198}

#: 입력의 기본값. API는 받은 값을 이 위에 덮어써서 빠진 항목을 채운다.
DEFAULTS = dict(
    age=52, sex="M", retire_age=60, spouse=True, s_age=50, s_sex="F",
    deposit=1.5, invest=2.0, ret_acct=1.5, other=0.0, saving=150, salary=0, s_salary=0,
    save_mode="월 금액", save_rate=15, wage_g=1.0, ret_contrib=False,
    nps_auto=False, nps_years=10, s_nps_years=10,
    essential=217, lifestyle=1000, nps=110, s_nps=50, nps_start=65,
    priv=0, priv_start=60, priv_years=20, legacy=1.0, children=2, profile="균형형",
    region="대도시", h_official=0.0, h_market=0.0, h_n=1, h_joint=False, h_years=10, h_hi=0.0, rent=0,
    biz=0, biz_until=65, biz_work=False, rm_age=0, ds_age=0, ds_ratio=50, events=[],
    ev_ltv=60, ev_loan_years=30, ev_jeonse=0.0, ev_rent=0,
    overseas=20, own_equal=True, survivor=70, enhanced=True, paths=10000,
    # (v36) 간병비 초과상승률(%p/년), 부양하는 부모, 행동 가정
    care_growth=1.0, parents=[], behaviour=False,
    # (v48) 이미 갚고 있는 대출: 남은 원금(억), 금리(%, 0이면 시장금리+1.5%p), 남은 기간(년)
    debt=0.0, debt_rate=0.0, debt_years=10,
    # (v48) 입력 단계: 1 빠른 진단 / 2 내 재산 반영 / 3 전문가 상담. 화면에 보일 항목만 정한다(계산엔 영향 없음)
    tier=2,
    # (v49) 퇴직급여: 제도("DC" 확정기여형·"DB" 확정급여형·퇴직금), 지금 직장 근속연수(년, 본인·배우자)
    ret_plan="DC", tenure=0, s_tenure=0,
    # (v50) 은퇴 후 근로(월 만 원, 몇 살 전까지 — 본인·배우자), 전월세 보증금(억), 나이에 따른 지출 변화
    post_inc=0, post_until=0, s_post_inc=0, s_post_until=0, jeonse=0.0, spend_profile="lifestyle",
    # (v50.9) 자녀 직장건보 피부양자로 등록할 수 있나 (소득 2,000만·재산 기준은 엔진이 해마다 판정)
    hi_dependent=False,
)


def defaults() -> dict:
    """기본 입력 한 벌(복사본)."""
    return deepcopy(DEFAULTS)


def fill(p: dict | None) -> dict:
    """빠진 항목을 기본값으로 채운 입력. API가 부분 입력을 받을 때 쓴다."""
    out = defaults()
    out.update(p or {})
    return out


def total_assets(p: dict) -> float:
    """금융자산 합계(억 원)."""
    return p["deposit"] + p["invest"] + p["ret_acct"] + p["other"]


def nps_values(p: dict) -> tuple:
    """국민연금 월액(본인, 배우자). 자동 계산이면 소득·가입기간으로 추정."""
    if p.get("nps_auto"):
        n1 = estimate_nps_monthly(p.get("salary", 0), p.get("nps_years", 0), p["age"], p["retire_age"])
        n2 = estimate_nps_monthly(p.get("s_salary", 0), p.get("s_nps_years", 0), p["s_age"],
                                  p["retire_age"] - (p["age"] - p["s_age"])) if p["spouse"] else 0.0
        return n1, n2
    return float(p["nps"]), float(p["s_nps"]) if p["spouse"] else 0.0


def monthly_saving(p: dict) -> float:
    """은퇴 전 월 저축액(만 원)."""
    if p["age"] >= p["retire_age"]:
        return 0.0
    if p.get("save_mode") == "소득의 %":
        return (p.get("salary", 0) + (p.get("s_salary", 0) if p["spouse"] else 0)) * p.get("save_rate", 0) / 100 / 12
    return float(p["saving"])


def severance_estimate(p: dict) -> tuple:
    """(v49) 은퇴 때 받을 퇴직급여의 대략적인 크기(억 원, 오늘 가치)와 퇴직소득세(만 원, 오늘 가치).
    화면 안내용 결정론적 근사 — 계산은 엔진이 경로별로 한다. DB는 마지막 월급 × 근속연수,
    DC는 연봉의 1/12 적립을 실질 2%로 굴린 값."""
    from .tax import retirement_income_tax
    if not p.get("ret_contrib") or p["age"] >= p["retire_age"]:
        return 0.0, 0.0
    ty = p["retire_age"] - p["age"]; g = p.get("wage_g", 1.0) / 100
    tot = tax = 0.0
    people = [(p.get("salary", 0), p.get("tenure", 0))]
    if p["spouse"]:
        people.append((p.get("s_salary", 0), p.get("s_tenure", 0)))
    for sal, ten in people:
        if sal <= 0:
            continue
        last = sal * (1 + g) ** max(ty - 1, 0)
        if p.get("ret_plan", "DC") == "DB":
            R = last / 12 * (ten + ty)
        else:
            R = sum(sal * (1 + g) ** j / 12 * 1.02 ** (ty - j) for j in range(ty))
        cpi = 1.0247 ** ty                                        # 공제표·세율표는 명목 고정 → 명목으로 세금을 매기고 오늘 가치로
        tot += R; tax += float(retirement_income_tax(R * cpi, ten + ty)) / cpi
    return tot / 10000, tax


def income_gap(p: dict) -> tuple:
    """(필요한 생활비, 받는 연금, 차액) — 모두 월 만 원."""
    need = p["essential"] + p["lifestyle"] / 12
    n1, n2 = nps_values(p)
    pension = n1 + n2 + p["priv"] / 12
    return need, pension, need - pension


#: 값이 정해져 있는 항목 — API로 들어온 엉뚱한 값이 엔진 안에서 터지지 않게 앞에서 막는다
CHOICES = {"sex": ("M", "F"), "s_sex": ("M", "F"), "profile": tuple(PROFILES),
           "region": ("대도시", "중소도시", "농어촌"), "save_mode": ("월 금액", "소득의 %"),
           "floor": ("계리적", "보수적"), "ret_plan": ("DC", "DB"),
           "spend_profile": ("flat", "lifestyle", "all")}
RANGES = {"age": (18, 100), "s_age": (18, 100), "retire_age": (30, 100), "nps_start": (55, 75),
          "paths": (100, 200_000), "survivor": (0, 100), "overseas": (0, 100), "children": (0, 10),
          "care_growth": (-2.0, 5.0), "debt_rate": (0.0, 20.0), "debt_years": (1, 40), "tier": (1, 3),
          "tenure": (0, 50), "s_tenure": (0, 50), "post_until": (0, 100), "s_post_until": (0, 100)}
NON_NEGATIVE = ("deposit", "invest", "ret_acct", "other", "saving", "salary", "s_salary",
                "essential", "lifestyle", "nps", "s_nps", "priv", "legacy",
                "h_official", "h_market", "h_n", "rent", "biz", "debt",
                "post_inc", "s_post_inc", "jeonse")


def validate(p: dict) -> dict:
    """입력을 검사하고 그대로 돌려준다. 어긋나면 어디가 틀렸는지 적어 ValueError."""
    bad = []
    for k, allowed in CHOICES.items():
        if k in p and p[k] is not None and p[k] not in allowed:
            bad.append(f"{k}={p[k]!r} — 가능한 값: {', '.join(map(str, allowed))}")
    for k, (lo, hi) in RANGES.items():
        if k in p:
            try:
                x = float(p[k])
            except (TypeError, ValueError):
                bad.append(f"{k}={p[k]!r} — 숫자여야 합니다"); continue
            if not (lo <= x <= hi):
                bad.append(f"{k}={p[k]} — {lo}~{hi} 범위여야 합니다")
    for k in NON_NEGATIVE:
        if k in p:
            try:
                if float(p[k]) < 0:
                    bad.append(f"{k}={p[k]} — 0 이상이어야 합니다")
            except (TypeError, ValueError):
                bad.append(f"{k}={p[k]!r} — 숫자여야 합니다")
    for key in ("events", "parents"):
        if not isinstance(p.get(key, []), list):
            bad.append(f"{key} — 목록이어야 합니다")
    for d in (p.get("parents") or []):
        if not isinstance(d, dict) or not (18 <= float(d.get("age", -1)) <= 110):
            bad.append(f"parents — 나이가 18~110이어야 합니다: {d!r}")
        elif d.get("sex", "F") not in ("M", "F"):
            bad.append(f"parents — sex 는 M 또는 F 여야 합니다: {d!r}")
    if bad:
        raise ValueError("; ".join(bad))
    return p


def build_case(p: dict, v: dict | None = None) -> tuple:
    """입력 p와 변형 v로 엔진 인자 한 벌을 만든다.

    v(variant)는 '이것만 바꿔 보기'용 덮어쓰기다 — 은퇴 시기, 저축액, 주식 비중, 충격 시나리오 등.
    같은 난수(seed=42)를 쓰므로 v를 바꾼 결과와 원래 결과의 차이가 곧 그 선택의 효과다.

    반환: (Household, SimConfig, EconomyV2, TaxConfig, HouseConfig, biz, qx_table, AddOns)
    """
    v = v or {}
    validate(p)
    nps_start = v.get("nps_start", p["nps_start"])
    retire = p["retire_age"] + v.get("retire_delta", 0)

    def mk(age, sex, nps, priv=0, ps=0, py=0):
        return Person(age=age, sex=sex, nps_monthly=nps, nps_start_age=nps_start,
                      private_pension_annual=priv, private_pension_start=ps, private_pension_years=py)

    n1, n2 = nps_values(p)
    members = [mk(p["age"], p["sex"], n1, p["priv"], p["priv_start"], p["priv_years"] if p["priv"] > 0 else 0)]
    members[0].salary = float(p.get("salary", 0)); members[0].tenure = float(p.get("tenure", 0))
    members[0].post_income = float(p.get("post_inc", 0)) * 12; members[0].post_until = int(p.get("post_until", 0))
    if p["spouse"]:
        members.append(mk(p["s_age"], p["s_sex"], n2))
        members[1].salary = float(p.get("s_salary", 0)); members[1].tenure = float(p.get("s_tenure", 0))
        members[1].post_income = float(p.get("s_post_inc", 0)) * 12; members[1].post_until = int(p.get("s_post_until", 0))

    m = v.get("spend_mult", 1.0)
    liquid = max(v.get("assets", total_assets(p)) - p["ret_acct"], 0)
    use_rate = p.get("save_mode") == "소득의 %" and "saving_monthly" not in v
    sal_tot = max(p.get("salary", 0) + (p.get("s_salary", 0) if p["spouse"] else 0), 1)

    events = []
    for ev in p.get("events", []):
        if ev["kind"] == "주택 구입":
            events.append(dict(label=ev["kind"], age=int(ev["age"]), amount=float(ev["amount"]), kind="house",
                               ltv=p.get("ev_ltv", 60) / 100, loan_years=int(p.get("ev_loan_years", 30)),
                               deposit_back=float(p.get("ev_jeonse", 0)) * 10000,
                               rent_saving=float(p.get("ev_rent", 0)) * 12))
        else:
            events.append(dict(label=ev["kind"], age=int(ev["age"]), amount=float(ev["amount"]),
                               years=int(ev["years"]), kind="cost"))

    hh = Household(
        members=members, liquid_assets=liquid * 10000, pension_balance=p["ret_acct"] * 10000, events=events,
        saving_rate=(v.get("save_rate_override", p.get("save_rate", 0) / 100)
                     + v.get("saving_delta", 0) * 12 / sal_tot) if use_rate else 0.0,
        wage_growth=p.get("wage_g", 1.0) / 100,
        retirement_contrib=bool(p.get("ret_contrib")) and p["age"] < p["retire_age"],
        ret_plan=p.get("ret_plan", "DC"), spend_profile=v.get("spend_profile", p.get("spend_profile", "lifestyle")),
        stock_weight=v.get("stock_weight", PROFILES[p["profile"]]),
        annual_spending=p["essential"] * 12 * m, essential=p["essential"] * 12 * m,
        lifestyle=p["lifestyle"] * m, legacy_target=p["legacy"] * 10000,
        retire_age=retire if p["age"] < retire else None,
        annual_saving=(v.get("saving_monthly", p["saving"]) + v.get("saving_delta", 0)) * 12,
        survivor_spending_ratio=p["survivor"] / 100,
        dependents=[dict(age=int(d["age"]), sex=d.get("sex", "F"), share=float(d.get("share", 1.0)) )
                    for d in (p.get("parents") or [])],
        floor_method="fixed95" if p.get("floor") == "보수적" else "actuarial",
        debt_balance=float(p.get("debt", 0)) * 10000,
        debt_rate=(float(p["debt_rate"]) / 100) if p.get("debt_rate") else None,
        debt_years=int(p.get("debt_years", 10)))

    care = CareMarkov() if p["enhanced"] else CareShock()
    if v.get("care_mult"):
        care = (CareMarkov(incidence=tuple((a, x * v["care_mult"]) for a, x in CareMarkov().incidence))
                if p["enhanced"] else CareShock(lam=0.03 * v["care_mult"]))
    if v.get("care_off"):                                   # (v50.7) '다른 계산기라면' 비교용: 간병 위험을 뺀다
        from dataclasses import replace as _rp
        care = _rp(care, enabled=False)
    # (v47) 지출 경직성(lifestyle_floor=0.30)은 SimConfig 기본값이 됐다. '행동' 토글은 이제 패닉 매도만 더한다.
    beh = dict(panic_drop=0.25) if (v.get("behaviour", p.get("behaviour"))) else {}
    cfg = SimConfig(n_paths=v.get("paths", p["paths"]), seed=42, care=care,
                    nps=NPSRules(basic_region=p.get("region", "대도시"), basic_on=not v.get("basic_off", False),
                                 basic_couple_cut=v.get("basic_couple_cut", NPSRules.basic_couple_cut)),   # (v50.9) 개정 시나리오
                    care_real_growth=v.get("care_growth", p.get("care_growth", 1.0)) / 100, **beh)

    ekw = dict(v.get("eco", {}))
    for key in ("stock_shocks", "infl_shocks"):
        if key in ekw:
            ekw[key] = {int(t): x for t, x in ekw[key].items()}
    eco = EconomyV2.enhanced(**ekw) if p["enhanced"] else EconomyV2(**ekw)

    tkw = dict(overseas_share=p["overseas"] / 100, n_children=p["children"],
               dependent_possible=bool(p.get("hi_dependent", False)),
               ownership=None if (p["own_equal"] or not p["spouse"]) else (1, 0))
    tkw.update(v.get("tax", {}))

    k = len(members)
    hv = v.get("house", {})
    rm_age = hv.get("rm_age", p.get("rm_age", 0))
    ds_age = hv.get("ds_age", p.get("ds_age", 0))
    house = HouseConfig(official=p["h_official"] * 10000, market=p["h_market"] * 10000, n_houses=p["h_n"],
                        owner_share=(tuple([1 / k] * k) if p["h_joint"] else None), years_held=p["h_years"],
                        hi_property_monthly=p["h_hi"] or None, rent_annual=p["rent"],
                        reverse_mortgage_age=rm_age or None, downsize_age=ds_age or None,
                        downsize_ratio=hv.get("ds_ratio", p.get("ds_ratio", 50)) / 100,
                        jeonse=float(p.get("jeonse", 0)) * 10000)

    biz = [dict(income=p["biz"], until_age=p["biz_until"], workplace=p["biz_work"])] if p["biz"] > 0 else [None]
    biz += [None] * (k - 1)

    q = None
    if v.get("q_mult"):
        q = {s: np.minimum(a * v["q_mult"], 1) for s, a in mortality.default_qx().items()}
        for a in q.values():
            a[-1] = 1
    return hh, cfg, eco, TaxConfig(**tkw), house, biz, q, AddOns(**v.get("addons", {}))
