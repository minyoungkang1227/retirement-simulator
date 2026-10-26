"""(v51) 태블릿 앱 '가벼운 모드' — 1단계 빠른 진단만 담은 얇은 층.

앱(app/)은 이 모듈을 브라우저 안의 파이썬(Pyodide)에서 그대로 불러 쓴다. 서버가 없으므로
고객 입력은 기기 밖으로 나가지 않는다. 계산 로직은 하나도 여기 없다 — 웹 앱과 같은
`inputs.build_case` · `engine_tax.run` · `advice` 를 부르므로 같은 입력이면 같은 숫자가 나온다.

화면 입력(form) → 엔진 입력(p) 변환은 웹 앱 1단계(streamlit_app.py STEPS1)와 같은 규칙이다.
그 규칙이 바뀌면 여기도 바꾸고, tests/test_engine.py 의 lite 테스트가 웹과 같은지 감시한다.

결과는 무거운 순서로 나눠 부른다(base → retire_age → band → actions → assets).
앱은 하나 끝날 때마다 화면에 채워, 첫 숫자를 몇 초 안에 보여준다.
"""
from __future__ import annotations

import json

import numpy as np

from . import __version__, advice, engine_tax
from .inputs import AVG_SPEND, fill, income_gap, monthly_saving, severance_estimate, total_assets, validate
from .session import calc_id

TIER1_SPLIT = (0.3, 0.4, 0.3)     # 금융자산 합계 → 예금 · 주식·펀드 · 연금계좌 (웹 앱 1단계와 같은 비율)
TARGET = 0.9                      # '10번 중 9번'
STOCK_MEAN_NOMINAL = 0.083        # 주식 명목 기대수익률(웹 앱과 같은 값) — 수익률 직접 정해 보기의 기준
SIDE_PATHS = 3000                 # 보조 계산(은퇴 가능 나이·순위·가정 범위)의 경로 수 — 웹 앱의 보조 계산과 같은 크기


def tier1_nps_years(age: int) -> int:
    """국민연금을 소득으로 추정할 때 '지금까지 가입 기간' 근사: 30세부터 그 기간의 80%를 납부."""
    return int(round(max(0, min(age - 30, 35)) * 0.8))


def averages(spouse: bool) -> dict:
    """모를 때 쓰는 평균값(국민연금연구원 2024 국민노후보장패널). 기본생활비 = 최소, 여행·취미 = (적정−최소)×12."""
    ess = AVG_SPEND["부부_최소"] if spouse else AVG_SPEND["개인_최소"]
    d = (AVG_SPEND["부부_적정"] - AVG_SPEND["부부_최소"]) if spouse else (AVG_SPEND["개인_적정"] - AVG_SPEND["개인_최소"])
    return {"essential": ess, "lifestyle": int(round(d * 12 / 100) * 100)}


def to_inputs(form: dict) -> dict:
    """앱 화면의 값(1단계 질문들) → 엔진 입력 한 벌. 묻지 않은 값은 기본값."""
    f = dict(form)
    p = fill({})
    age, ra = int(f["age"]), int(f["retire_age"])
    spouse = bool(f.get("spouse", False))
    pre = age < ra
    tot = float(f.get("total", 0.0))
    h = float(f.get("h_market", 0.0))
    nps, s_nps = int(f.get("nps", 0)), int(f.get("s_nps", 0))
    sal, ssal = float(f.get("salary", 0)), float(f.get("s_salary", 0))
    post = int(f.get("post_inc", 0))
    p.update(
        tier=1,
        age=age, sex=f.get("sex", "M"), retire_age=ra, spouse=spouse,
        s_age=int(f.get("s_age", p["s_age"])), s_sex=f.get("s_sex", "F"),
        deposit=round(tot * TIER1_SPLIT[0], 2), invest=round(tot * TIER1_SPLIT[1], 2),
        ret_acct=round(tot * TIER1_SPLIT[2], 2), other=0.0,
        saving=int(f.get("saving", 0)) if pre else 0, save_mode="월 금액",
        h_market=h, h_official=round(h * 0.69, 1), h_n=1, h_years=10,
        debt=float(f.get("debt", 0.0)), debt_years=int(f.get("debt_years", 10)), debt_rate=0.0,
        jeonse=float(f.get("jeonse", 0.0)),
        essential=int(f["essential"]), lifestyle=int(f["lifestyle"]),
        ret_contrib=bool(f.get("ret_contrib", False)) and pre, ret_plan="DB",
        tenure=int(f.get("tenure", 0)), s_tenure=int(f.get("s_tenure", 0)),
        post_inc=post, post_until=int(f.get("post_until", 0)) if post > 0 else 0,
        nps_auto=(nps == 0 and sal > 0), nps=nps, s_nps=s_nps, salary=sal, s_salary=ssal,
        nps_years=tier1_nps_years(age), s_nps_years=tier1_nps_years(int(f.get("s_age", p["s_age"]))),
        nps_start=65, priv=0, legacy=0.0, profile=f.get("profile", "균형형"),
    )
    if "paths" in f:
        p["paths"] = int(f["paths"])
    return validate(p)


def assumptions(p: dict) -> list:
    """1단계에서 묻지 않고 채운 값 — 결과 화면에 그대로 보여준다 (웹 앱 tier_assumptions 의 1단계 부분과 같은 문장)."""
    out = [f"금융자산을 예금 {TIER1_SPLIT[0]:.0%} · 주식·펀드 {TIER1_SPLIT[1]:.0%} · 연금계좌 {TIER1_SPLIT[2]:.0%}로 나눠 계산 (세금이 계좌마다 다릅니다)",
           f"국민연금은 {p['nps_start']}세부터 받고, 사적연금은 연금계좌 적립금 외에는 없음",
           "남길 자산 목표는 없음"]
    if p.get("nps_auto"):
        out.append(f"국민연금은 소득으로 추정 — 30세부터 가입해 그 기간의 80%를 냈다고 가정 (지금까지 {p.get('nps_years', 0)}년)")
    if p.get("h_market", 0) > 0:
        out.append("집은 1채, 공시가격은 시세의 69%, 보유 10년으로 가정")
    if p.get("debt", 0) > 0:
        out.append("대출 금리는 지금 시장금리 + 1.5%p 고정")
    if p.get("ret_contrib"):
        out.append("퇴직금은 '마지막 월급 × 근속연수'(퇴직금·DB 제도) 기준, 은퇴 때 IRP로 받아 연금으로 수령")
    out += ["배우자는 은퇴 후 일하지 않음",
            "여행·취미 지출은 75세부터 줄어 85세에 절반, 기본생활비는 평생 같음 (나이에 따른 지출 변화)",
            "부모님 간병비 분담 없음",
            f"한 명만 남았을 때 생활비는 부부일 때의 {p.get('survivor', 70)}%",
            f"주식 중 해외주식 {p.get('overseas', 20)}%, 금융자산은 부부 균등 명의",
            "간병비는 물가보다 연 1.0%p 더 오름 (2022~2026 장기요양 수가 실적)"]
    return out


def _prob(p: dict, v: dict | None = None) -> float:
    return advice.essential_prob(advice.run_case(p, v or {}))


def base(p: dict) -> dict:
    """맨 위 결론: 기본생활 유지 확률, 모자라기 시작하는 나이, 여행 충족률, 월 부족분."""
    res = advice.run_case(p, {})
    g = engine_tax.summarize(res)
    d = res["depleted_at"]; y = res["youngest_age"]
    need, pension, gap = income_gap(p)
    sev, sev_tax = severance_estimate(p)
    return {"ok": float(g["기본생활 유지 확률"]),
            "dep_age": float(np.median(d[d >= 0] + y)) if (d >= 0).any() else None,
            "life": float(g.get("여행·취미 평균 충족률", 0.0)) if p["lifestyle"] > 0 else None,
            "need": float(need), "pension": float(pension), "gap": float(gap),
            "assets": float(total_assets(p)), "saving": float(monthly_saving(p)),
            "severance": float(sev), "pre": p["age"] < p["retire_age"],
            "calc_id": calc_id(p), "model_version": __version__, "paths": int(p["paths"]),
            "assumptions": assumptions(p)}


def retire_age(p: dict, target: float = TARGET) -> dict:
    """10번 중 9번 기본생활을 지키는 가장 이른 은퇴 나이 (웹 앱 retire_age_for 와 같은 탐색)."""
    if p["age"] >= p["retire_age"]:
        return {"age": None, "prob": None, "applies": False}
    paths = min(p["paths"], SIDE_PATHS)
    f = lambda a: _prob(p, {"retire_delta": a - p["retire_age"], "paths": paths})
    lo, hi = max(p["age"] + 1, p["retire_age"] - 10), p["retire_age"] + 15
    p_hi = f(hi)
    if p_hi < target:
        return {"age": None, "prob": p_hi, "applies": True, "max_age": hi}
    p_lo = f(lo)
    if p_lo >= target:
        return {"age": lo, "prob": p_lo, "applies": True}
    best = (hi, p_hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2; pm = f(mid)
        if pm >= target:
            hi, best = mid, (mid, pm)
        else:
            lo = mid
    return {"age": best[0], "prob": best[1], "applies": True}


def band(p: dict) -> dict:
    """주식 위험프리미엄 · 수명 · 간병 발생률을 하나씩 추정 폭만큼 움직인 범위 (advice 와 같은 계산)."""
    lo, hi = advice.assumption_band(p, paths=min(p["paths"], SIDE_PATHS))
    return {"lo": float(lo), "hi": float(hi)}


def actions(p: dict, k: int = 3) -> list:
    """효과가 큰 선택지 k개 (신뢰구간을 넘는 것만)."""
    return [{"label": lab, "how": how, "gain": float(gain), "ci": float(ci)}
            for lab, how, gain, ci in advice.action_ranking(p, paths=SIDE_PATHS)[:k]]


def stock_return(p: dict, r_stock: float) -> float:
    """주식 기대수익률을 직접 정했을 때의 확률 — '다른 계산기처럼 정하면' 시연용 (웹 앱과 같은 방식)."""
    from .economy_v2 import EconomyV2
    erp = EconomyV2.enhanced().erp + (float(r_stock) - STOCK_MEAN_NOMINAL)
    return _prob(p, {"eco": {"erp": erp}, "paths": min(p["paths"], SIDE_PATHS)})


def call(name: str, form_json: str, extra_json: str = "{}") -> str:
    """앱(자바스크립트)이 부르는 단일 입구. 입력·출력 모두 JSON 문자열."""
    form, extra = json.loads(form_json), json.loads(extra_json or "{}")
    if name == "averages":
        return json.dumps(averages(bool(form.get("spouse"))), ensure_ascii=False)
    p = to_inputs(form)
    if name == "base":
        out = base(p)
    elif name == "retire_age":
        out = retire_age(p)
    elif name == "band":
        out = band(p)
    elif name == "actions":
        out = actions(p)
    elif name == "stock_return":
        out = {"prob": stock_return(p, extra["r"])}
    elif name == "inputs":
        out = p
    else:
        raise ValueError(f"알 수 없는 계산: {name}")
    return json.dumps(out, ensure_ascii=False)
