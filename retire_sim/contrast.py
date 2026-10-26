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
    for name, v in (("세금·건강보험료", {"tax": {"enabled": False}}),
                    ("간병비 (장기요양)", {"care_off": True}),
                    ("기초연금", {"basic_off": True})):
        pr = advice.essential_prob(advice.run_case(p, {**v0, **v}))
        rows.append((name, pr, pr - full))
    allv = {**v0, "tax": {"enabled": False}, "care_off": True, "basic_off": True}
    none = advice.essential_prob(advice.run_case(p, allv))          # 셋 다 빼고 수익률·수명만 흔들리게
    return {"full": full, "rows": rows, "none": none, "tax20": float(s["20년 세금"]), "hi20": float(s["20년 건보료"]),
            "prop20": float(s["20년 보유세"])}
