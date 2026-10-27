"""(v52) 생애 단계별 첫 문장 — 같은 엔진, 단계마다 '먼저 보여줄 숫자'만 다르다.

20대에게 "95세까지 기본생활 유지 확률"은 너무 멀어서 아무 행동도 일으키지 않는다. 단계마다 지금 결정에 닿는 숫자를 앞에 둔다.
  1 사회초년생    : 지금 저축이면 은퇴 때 얼마가 모이나(범위), 매달 50만 원 더 모으면 확률이 얼마나 움직이나
  2 결혼·내 집    : 집을 사는 계획을 넣었을 때와 뺐을 때(전세 유지) 확률·은퇴 때 자산 차이
  3 자녀·부모     : 자녀 양육·교육비와 부모 간병 분담을 넣기 전/후 확률, 앞으로 나갈 자녀 비용 총액
  4 은퇴 준비·후  : 기본 화면 그대로(은퇴 가능 나이·부족분)
계산은 모두 advice.run_case 의 변형(v)으로 한다 — 같은 난수 위에서 비교하므로 차이가 곧 그 결정의 효과다.
"""
from __future__ import annotations

import numpy as np

from . import advice
from .inputs import STAGES, kid_schedule, kid_total

EXTRA_SAVING = 50          # 1단계: '매달 이만큼 더 모으면' (만 원)


def wealth_at(p: dict, res, age: int) -> tuple:
    """age 세 때 금융자산(오늘 가치, 억 원)의 (중앙값, 10%, 90%). 그 나이가 범위 밖이면 (None, None, None)."""
    t = int(age) - int(p["age"])
    W = res["W_real"]
    if t < 0 or t >= W.shape[1]:
        return None, None, None
    w = W[:, t] / 10000
    return float(np.median(w)), float(np.percentile(w, 10)), float(np.percentile(w, 90))


def headline(p: dict, paths: int = 3000) -> dict:
    """단계별 첫 문장 재료. 반환 dict 의 "kind" 로 화면이 문장을 고른다. 단계가 0이나 4면 {"kind": "none"}."""
    st = int(p.get("stage", 0) or 0)
    v0 = {"paths": min(int(p.get("paths", paths)), paths)}
    if st == 1 and p["age"] < p["retire_age"]:
        base = advice.run_case(p, v0)
        med, lo, hi = wealth_at(p, base, p["retire_age"])
        p0 = advice.essential_prob(base)
        p1 = advice.essential_prob(advice.run_case(p, {**v0, "saving_delta": EXTRA_SAVING}))
        return {"kind": "start", "retire_age": p["retire_age"], "median": med, "lo": lo, "hi": hi,
                "prob": p0, "prob_extra": p1, "extra": EXTRA_SAVING}
    if st == 2:
        has_house_plan = any(e.get("kind") == "주택 구입" for e in p.get("events", []))
        if not has_house_plan:
            return {"kind": "house_none", "own": p.get("h_market", 0) > 0}
        base = advice.run_case(p, v0)
        alt = advice.run_case(p, {**v0, "house_ev_off": True})
        ev = next(e for e in p["events"] if e.get("kind") == "주택 구입")
        return {"kind": "house", "age": int(ev["age"]), "price": float(ev["amount"]) / 10000,
                "prob": advice.essential_prob(base), "prob_rent": advice.essential_prob(alt),
                "w_buy": wealth_at(p, base, p["retire_age"])[0], "w_rent": wealth_at(p, alt, p["retire_age"])[0]}
    if st == 3:
        kids = kid_schedule(p); pars = p.get("parents") or []
        if not kids and not pars:
            return {"kind": "family_none"}
        base = advice.run_case(p, v0)
        alt = advice.run_case(p, {**v0, "kids_off": True, "parents_off": True})
        return {"kind": "family", "prob": advice.essential_prob(base), "prob_without": advice.essential_prob(alt),
                "n_kids": len(kids), "kid_total": kid_total(p), "n_parents": len(pars)}
    return {"kind": "none"}


def stage_label(p: dict) -> str:
    st = int(p.get("stage", 0) or 0)
    return STAGES[st][0] if st in STAGES else ""
