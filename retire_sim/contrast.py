"""(v50.7) '다른 계산기였다면' — 같은 가구를 시중 계산기 방식으로도 계산해 나란히 보여준다.

시중 계산기(앱·웹)는 대부분
  · 수익률을 하나로 정해 매년 똑같이 받는다고 보고,
  · 정해진 나이(보통 90~100세)까지 살며,
  · 세금·건강보험료·간병비·기초연금을 넣지 않는다.
그래서 답이 '된다 / 안 된다' 하나로 나온다. 이 모듈은 그 방식(simple_projection)과,
노후나침반이 넣은 한국 제도를 하나씩 뺐을 때의 차이(breakdown)를 같은 입력으로 계산한다.

공정하게 비교하려고, 고정 수익률은 아무 숫자가 아니라 **이 모델의 평균 수익률**(투자 성향별 주식·채권 평균)을 쓴다.
같은 평균인데도 결과가 다르다면, 그 차이는 수익률을 '정하느냐 / 흔들리게 두느냐'와 제도에서 나온 것이다.
"""
from __future__ import annotations

from . import advice
from .inputs import PROFILES, income_gap, monthly_saving, total_assets

STOCK_MEAN = 0.083      # 모델의 주식 명목 평균 수익률(보수 차감 후) — streamlit_app.STOCK_MEAN_NOMINAL 과 같은 값
BOND_MEAN = 0.032       # 채권 명목 평균
INFLATION = 0.025       # 물가 장기 평균
END_AGE = 95            # 시중 계산기에서 흔한 계획 끝 나이 — 노후나침반의 기본생활 보호 기준과도 같다


def fixed_return(p: dict) -> float:
    """투자 성향별 명목 평균 수익률 (주식 비중 × 주식 + 나머지 × 채권)."""
    w = PROFILES[p["profile"]]
    return w * STOCK_MEAN + (1 - w) * BOND_MEAN


def simple_projection(p: dict, r_nominal: float | None = None, end_age: int = END_AGE) -> dict:
    """시중 계산기 방식: 고정 수익률, end_age까지 생존, 세금·건보료·간병·기초연금 없음. 모두 오늘 돈 가치(만 원).

    은퇴 전: 해마다 자산이 실질수익률만큼 늘고 월 저축 × 12가 더해진다.
    은퇴 후: 해마다 생활비(기본 + 여행)가 나가고, 연금 개시 나이부터 연금(국민연금 + 사적연금)이 들어온다.
    반환: {"r": 명목 수익률, "depleted_age": 바닥나는 나이 또는 None, "left": end_age 때 남는 돈(억)}"""
    r = fixed_return(p) if r_nominal is None else float(r_nominal)
    real = (1 + r) / (1 + INFLATION) - 1
    w = total_assets(p) * 10000
    save = monthly_saving(p) * 12
    spend = p["essential"] * 12 + p["lifestyle"]
    _, pension, _ = income_gap(p)
    nps_part = pension - p.get("priv", 0) / 12
    for age in range(int(p["age"]), end_age):
        w *= 1 + real
        if age < p["retire_age"]:
            w += save
        else:
            inflow = (nps_part * 12 if age >= p["nps_start"] else 0.0)
            if p.get("priv", 0) and p.get("priv_start", 60) <= age < p.get("priv_start", 60) + p.get("priv_years", 20):
                inflow += p["priv"]
            w += inflow - spend
        if w < 0:
            return {"r": r, "depleted_age": age + 1, "left": 0.0}
    return {"r": r, "depleted_age": None, "left": w / 10000}


def breakdown(p: dict, paths: int = 3000) -> dict:
    """노후나침반이 넣은 한국 제도를 하나씩 뺐을 때의 기본생활 유지 확률 (같은 난수 위에서).

    반환: {"full": 확률, "rows": [(이름, 그것을 뺀 확률, 차이), ...], "none": 셋 다 뺀 확률, "tax20": 20년 세금, "hi20": 20년 건보료,
          "prop20": 20년 보유세} — 금액은 오늘 가치 만 원."""
    from . import engine_tax
    v0 = {"paths": min(p.get("paths", paths), paths)}
    base = advice.run_case(p, v0)
    full = advice.essential_prob(base)
    s = engine_tax.summarize(base)
    rows = []
    items = [("세금·건강보험료", {"tax": {"enabled": False}}),
             ("간병비 (장기요양)", {"care_off": True}),
             ("기초연금", {"basic_off": True})]
    if p["age"] < p["retire_age"]:                                  # (v51) 은퇴 전 가구만 — 이미 은퇴했으면 없는 위험
        items.insert(2, ("은퇴 시점 위험", {"early_exit_prob": 0.0}))
    for name, v in items:
        pr = advice.essential_prob(advice.run_case(p, {**v0, **v}))
        rows.append((name, pr, pr - full))
    allv = {**v0, "tax": {"enabled": False}, "care_off": True, "basic_off": True, "early_exit_prob": 0.0}
    none = advice.essential_prob(advice.run_case(p, allv))          # 셋 다 빼고 수익률·수명만 흔들리게
    return {"full": full, "rows": rows, "none": none, "tax20": float(s["20년 세금"]), "hi20": float(s["20년 건보료"]),
            "prop20": float(s["20년 보유세"])}


#: (v50.9) 논의 중인 제도 개정 — 확정 전이라 기본 결과에는 넣지 않고 '바뀌면?'으로만 비교한다.
#: (이름, 근거·상태, 엔진 변형). 상태가 바뀌면 여기만 고친다(확정되면 기본값으로 옮기고 이 목록에서 뺀다).
POLICY_SCENARIOS = [
    ("기초연금 부부감액 20% → 15%",
     "정부안 — 2026-03 보건복지부 국회 보고. 소득 하위 40% 부부부터, 2027년까지 15%·2030년 10%",
     {"basic_couple_cut": 0.15}),
    ("기초연금 부부감액 폐지",
     "국회 법안 — 2026년 10% → 2027년 5% → 2028년 폐지. 재정 2030년까지 약 16.7조 원 (국회예산정책처)",
     {"basic_couple_cut": 0.0}),
    ("상속세 일괄공제 5억 → 8억",
     "국회 의원안 — 2025-11 기획재정위 조세소위 안건. 배우자공제 확대(5억 → 10억)는 부부가 모두 떠난 뒤 계산하는 이 모델에는 해당 없음",
     {"tax": {"estate_lump": 80000}}),
]


def policy_scenarios(p: dict, paths: int | None = None) -> list:
    """논의 중인 개정이 이 가구에 생기면 — 같은 난수 위에서 기본 결과와 비교.

    반환: [{"name", "basis", "prob", "diff", "ci", "estate_tax_diff", "transfer_diff"}] — 금액은 중앙값 차이(오늘 가치, 만 원)."""
    import numpy as np
    from . import engine_tax, metrics
    v0 = {"paths": paths} if paths else {}
    base = advice.run_case(p, v0)
    med = lambda r, k: float(np.nanmedian(r[k]))
    out = []
    for name, basis, v in POLICY_SCENARIOS:
        r = advice.run_case(p, {**v0, **v})
        c = metrics.compare(r, base)
        out.append({"name": name, "basis": basis, "prob": advice.essential_prob(r), "diff": -c["차이"], "ci": c["±"],
                    "estate_tax_diff": med(r, "estate_tax_real") - med(base, "estate_tax_real"),
                    "transfer_diff": med(r, "transfer_real") - med(base, "transfer_real")})
    return out
