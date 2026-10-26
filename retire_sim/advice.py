"""무엇을 바꾸면 얼마나 좋아지는가 — 선택지 순위와 가정의 폭.

이 로직은 화면이 아니라 업무 규칙이다. 앱과 API가 같은 함수를 쓰도록 라이브러리에 둔다.
모든 비교는 **같은 난수** 위에서 하므로, 달라진 만큼이 곧 그 선택의 효과다.
"""
from __future__ import annotations



from . import engine_tax, metrics
from .economy_v2 import EconomyV2
from .inputs import PROFILES, build_case, monthly_saving

ERP_SD = 0.014          # 주식 위험프리미엄 추정 표준오차 (KOSPI 2000~2026, economy_v2.CALIBRATED_EQUITY_2026_08.erp_sd)
LONGEVITY_SD = 0.05     # 사망률 수준의 로그 표준편차 (SimConfig.longevity_sigma_level과 같은 값)
CARE_INCIDENCE_SD = 0.20  # 간병 발생률의 판단 폭 ±20%. 공단 통계 보정(calibrate_care.py)에 표준오차가 없어 판단값이다.

# 가정 범위를 만들 때 하나씩 움직이는 가정들. (이름, [낮은 쪽 variant, 높은 쪽 variant])
# q_mult는 사망률 배수라 >1 이 '수명 짧음', care_mult는 발생률 배수라 >1 이 '간병 많음'.
def _band_variants(base_erp: float) -> list:
    import math
    return [
        ("주식 위험프리미엄", [{"eco": {"erp": base_erp - ERP_SD}}, {"eco": {"erp": base_erp + ERP_SD}}]),
        ("수명", [{"q_mult": math.exp(LONGEVITY_SD)}, {"q_mult": math.exp(-LONGEVITY_SD)}]),
        ("간병 발생률", [{"care_mult": 1 + CARE_INCIDENCE_SD}, {"care_mult": 1 - CARE_INCIDENCE_SD}]),
    ]


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


def assumption_band_detail(p: dict, paths: int | None = None) -> dict:
    """세 가정(주식 위험프리미엄 · 수명 · 간병 발생률)을 **하나씩** ±1 표준편차(간병은 판단 폭)만큼
    움직였을 때 '기본생활 유지 확률'이 어디까지 가는지. 가정별 (낮은 쪽, 높은 쪽)과 전체 (min, max)를 돌려준다.

    v46까지는 위험프리미엄 하나만 흔들었다. 수명·간병이 측정된 영향 1·3위인데 범위에서 빠져 있어서
    폭이 과소였다(v47). 세 가정을 동시에 흔들지는 않는다 — 동시에 나쁜 쪽으로 가는 결합확률은 작고,
    그 경우의 크기는 CTE90 부족액이 따로 보여준다.

    이 폭은 경로를 늘려도 줄지 않는 불확실성이다. 표본오차(1만 경로에서 약 ±1%p)와는 다른 종류다."""
    base_erp = EconomyV2.enhanced().erp
    extra = {"paths": paths} if paths else {}
    per = {}
    for name, (lo_v, hi_v) in _band_variants(base_erp):
        a = essential_prob(run_case(p, {**lo_v, **extra}))
        b = essential_prob(run_case(p, {**hi_v, **extra}))
        per[name] = (min(a, b), max(a, b))
    lo = min(v[0] for v in per.values()); hi = max(v[1] for v in per.values())
    return {"overall": (lo, hi), "by_factor": per}


def assumption_band(p: dict, paths: int | None = None) -> tuple:
    """(min, max) — 호환용. 자세한 분해는 assumption_band_detail."""
    return assumption_band_detail(p, paths)["overall"]
