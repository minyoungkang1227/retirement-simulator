"""무엇을 바꾸면 얼마나 좋아지는가 — 선택지 순위와 가정의 폭.

이 로직은 화면이 아니라 업무 규칙이다. 앱과 API가 같은 함수를 쓰도록 라이브러리에 둔다.
모든 비교는 **같은 난수** 위에서 하므로, 달라진 만큼이 곧 그 선택의 효과다.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import engine_tax, metrics
from .economy_v2 import EconomyV2
from .inputs import PROFILES, build_case, monthly_saving

ERP_SD = 0.014          # 주식 위험프리미엄 추정 표준오차 (KOSPI 2000~2026)


def run_case(p: dict, v: dict | None = None):
    hh, cfg, eco, tc, house, biz, q, ad = build_case(p, v or {})
    return engine_tax.run(hh, cfg, e=eco, tc=tc, house=house, qx_table=q, biz=biz, addons=ad)


def essential_prob(res) -> float:
    return float(engine_tax.summarize(res)["기본생활 유지 확률"])


def candidates(p: dict) -> list:
    """이 가구가 실제로 바꿀 수 있는 선택지들. (이름, 어떻게, variant)"""
    out = []
    if p["age"] < p["retire_age"]:
        out += [("은퇴를 2년 늦추기", f"{p['retire_age']}세 → {p['retire_age'] + 2}세", {"retire_delta": 2}),
                ("매달 저축 30만 원 늘리기",
                 f"월 {monthly_saving(p):.0f}만 원 → {monthly_saving(p) + 30:.0f}만 원", {"saving_delta": 30})]
    out.append(("생활비를 10% 줄이기", f"기본 월 {p['essential']}만 원 → {p['essential'] * 0.9:.0f}만 원",
                {"spend_mult": 0.9}))
    for k, w in PROFILES.items():
        if k != p["profile"]:
            out.append((f"투자 성향을 {k}으로", f"{p['profile']} → {k} (주식 비중 {w * 100:.0f}%)", {"stock_weight": w}))
    for a in (60, 65, 68, 70):
        if a != p["nps_start"] and a >= min(p["age"], 70):
            out.append((f"국민연금을 {a}세부터 받기", f"{p['nps_start']}세 → {a}세", {"nps_start": a}))
    if p["h_market"] > 0 and p["h_n"] == 1 and p["h_official"] <= 12 and not p.get("rm_age"):
        young = min(p["age"], p["s_age"]) if p["spouse"] else p["age"]
        for ra in sorted({max(young, 55, p["retire_age"] - (p["age"] - young)), max(young, 70)}):
            out.append((f"{ra}세에 주택연금 가입", "집에 살면서 매달 연금 받기", {"house": {"rm_age": int(ra)}}))
    if p["h_market"] > 0 and not p.get("ds_age"):
        da = max(p["age"], 70)
        out.append((f"{da}세에 집을 줄여 이사", f"시세 {p['h_market']:.1f}억 → 절반 가격 집",
                    {"house": {"ds_age": int(da), "ds_ratio": 50}}))
    return out


def action_ranking(p: dict, paths: int = 3000) -> list:
    """효과가 큰 순서. 신뢰구간을 넘는 선택지만 남긴다 — 난수 때문에 좋아 보이는 것을 거른다.

    반환: [(이름, 어떻게, 개선폭, 신뢰구간)]"""
    paths = min(p.get("paths", paths), paths)          # 후보가 많아 가볍게 — 순위만 보면 충분
    base = run_case(p, {"paths": paths})
    out = []
    for lab, how, v in candidates(p):
        r = run_case(p, {**v, "paths": paths})
        c = metrics.compare(r, base)
        gain = -c["차이"]
        if gain > c["±"]:
            out.append((lab, how, gain, c["±"]))
    return sorted(out, key=lambda x: -x[2])


def assumption_band(p: dict) -> tuple:
    """주식 위험프리미엄을 ±1 표준오차만큼 움직였을 때 '기본생활 지킬 확률'의 범위.

    경로를 늘려도 줄지 않는 불확실성이다. 결과를 숫자 하나로 읽지 않게 하려고 함께 보여준다."""
    base_erp = EconomyV2.enhanced().erp
    vals = [essential_prob(run_case(p, {"eco": {"erp": base_erp + d}})) for d in (-ERP_SD, ERP_SD)]
    return min(vals), max(vals)
