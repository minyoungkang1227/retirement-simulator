"""HTTP API: 기관이 자기 앱에 이 계산을 붙일 수 있게 하는 얇은 껍데기.

실행
    pip install fastapi uvicorn
    RETIRE_API_KEYS="기관A:키1,기관B:키2" uvicorn retire_sim.api:app --port 8000

엔드포인트
    GET  /health              살아 있는지
    GET  /version             모델 버전 (결과 재현에 필요)
    GET  /defaults            입력 기본값 한 벌 — 여기에 아는 값만 덮어쓰면 된다
    POST /simulate            가구 입력 → 목표 달성 확률·꼬리 위험·세금
    POST /compare             여러 선택지를 **같은 난수**로 비교 (차이의 신뢰구간 포함)
    POST /report              상담 리포트 HTML
    POST /product/compare     연금보험 vs 펀드 vs 예금
    GET  /usage               우리 기관의 호출 수·엔진 실행 수 (?month=2026-10). 종량 과금과 상담 로그의 근거

설계 원칙
  · 계산 로직은 하나도 여기 없다. 앱과 **같은** `inputs.build_case` 를 쓰므로 둘이 어긋날 수 없다.
  · 응답에 calc_id 와 model_version 을 넣는다. 나중에 "그때 그 숫자"를 그대로 재현할 수 있어야 한다.
  · 호출마다 (시각·기관·기능·계산 ID·모델 버전·실행 수)를 남긴다. 고객 이름·입력값은 남기지 않는다 (usage.py).
  · 인증은 API 키 한 줄. 키를 설정하지 않으면 **로컬 개발용으로만** 열린다(기동 시 경고).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__, engine_tax, metrics, usage
from .branding import Brand
from .inputs import build_case, defaults, fill
from .session import calc_id

MAX_PATHS = 20_000          # 한 요청이 서버를 독차지하지 않도록
MAX_VARIANTS = 8


# ── 인증 ───────────────────────────────────────────────────────────────────
def _keys() -> Dict[str, str]:
    """RETIRE_API_KEYS="기관A:키1,기관B:키2" → {키: 기관}"""
    raw = os.environ.get("RETIRE_API_KEYS", "").strip()
    out = {}
    for part in raw.split(","):
        if ":" in part:
            who, key = part.split(":", 1)
            if key.strip():
                out[key.strip()] = who.strip()
    return out


def client(x_api_key: Optional[str] = Header(default=None)) -> str:
    """키가 설정돼 있으면 반드시 맞아야 한다. 설정이 없으면 개발 모드로 통과시킨다."""
    keys = _keys()
    if not keys:
        return "dev"
    who = keys.get((x_api_key or "").strip())
    if not who:
        raise HTTPException(status_code=401, detail="X-API-Key 헤더가 없거나 올바르지 않습니다.")
    return who


# ── 요청·응답 모양 ──────────────────────────────────────────────────────────
class Case(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict, description="가구 입력. 빠진 항목은 기본값으로 채운다.")
    variant: Dict[str, Any] = Field(default_factory=dict, description="이것만 바꿔 보기 (은퇴 시기·저축·충격 등)")


class CompareRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    variants: List[Dict[str, Any]] = Field(default_factory=list, description="기준(빈 dict)과 비교할 선택지들")
    labels: Optional[List[str]] = None


class ReportRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    client_name: str = ""
    advisor: str = ""
    brand: Dict[str, Any] = Field(default_factory=dict)


class ProductRequest(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)
    product: Dict[str, Any] = Field(default_factory=dict, description="상품 스펙 (AnnuityProduct 필드)")
    paths: int = 5000


app = FastAPI(title="노후나침반 API", version=__version__,
              description="확률 기반 은퇴 진단 엔진. 응답의 calc_id 로 같은 결과를 다시 재현할 수 있습니다.")


# ── 도우미 ─────────────────────────────────────────────────────────────────
def _run(p: dict, v: dict | None = None):
    if p.get("paths", 0) > MAX_PATHS:
        raise HTTPException(status_code=422, detail=f"paths 는 {MAX_PATHS:,} 이하여야 합니다.")
    try:
        hh, cfg, eco, tc, house, biz, q, ad = build_case(p, v or {})
    except (KeyError, TypeError, ValueError) as ex:
        raise HTTPException(status_code=422, detail=f"입력을 해석할 수 없습니다: {ex}")
    return engine_tax.run(hh, cfg, e=eco, tc=tc, house=house, qx_table=q, biz=biz, addons=ad)


def _summary(res) -> dict:
    s = engine_tax.summarize(res)
    return {k: (None if v is None else float(v)) for k, v in s.items()}


# ── 엔드포인트 ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": __version__}


@app.get("/version")
def version():
    return {"model_version": __version__,
            "note": "같은 입력 + 같은 모델 버전 = 같은 결과. 모델이 바뀌면 이 번호가 올라갑니다."}


@app.get("/defaults")
def get_defaults(_: str = Depends(client)):
    return {"inputs": defaults(),
            "note": "아는 값만 덮어써서 /simulate 에 보내면 나머지는 이 기본값이 쓰입니다."}


@app.post("/simulate")
def simulate(req: Case, who: str = Depends(client)):
    p = fill(req.inputs)
    res = _run(p, req.variant)
    usage.record(who, "simulate", calc_id(p), __version__, units=1)
    return {"model_version": __version__, "calc_id": calc_id(p), "client": who,
            "summary": _summary(res),
            "depletion_age_median": (lambda d: float(d.mean()) if len(d) else None)(
                res["depleted_at"][res["depleted_at"] >= 0] + res["youngest_age"])}


@app.post("/compare")
def compare(req: CompareRequest, who: str = Depends(client)):
    """여러 선택지를 같은 난수로 돌려 기준 대비 차이와 95% 신뢰구간을 준다."""
    if len(req.variants) > MAX_VARIANTS:
        raise HTTPException(status_code=422, detail=f"variants 는 {MAX_VARIANTS}개 이하여야 합니다.")
    p = fill(req.inputs)
    base = _run(p, {})
    labels = req.labels or [f"선택지 {i + 1}" for i in range(len(req.variants))]
    rows = [{"label": "기준", "summary": _summary(base), "diff": 0.0, "ci": 0.0, "significant": False}]
    for lab, v in zip(labels, req.variants):
        r = _run(p, v)
        c = metrics.compare(r, base)
        rows.append({"label": lab, "summary": _summary(r),
                     "diff": -c["차이"], "ci": c["±"], "significant": c["유의"],
                     "note": "diff 는 '기본생활을 지킬 확률'의 변화(+가 좋음), ci 는 짝지은 95% 신뢰구간"})
    usage.record(who, "compare", calc_id(p), __version__, units=1 + len(req.variants))
    return {"model_version": __version__, "calc_id": calc_id(p), "client": who, "results": rows}


@app.post("/report")
def report_html(req: ReportRequest, who: str = Depends(client)):
    """상담 리포트(HTML). 브라우저에서 인쇄하면 PDF가 됩니다."""
    from .reporting import build_report
    p = fill(req.inputs)
    html = build_report(p, client_name=req.client_name, advisor=req.advisor,
                        brand=Brand.from_dict(req.brand))
    usage.record(who, "report", calc_id(p), __version__, units=1)
    return {"model_version": __version__, "calc_id": calc_id(p), "client": who,
            "content_type": "text/html", "html": html}


@app.post("/product/compare")
def product_compare(req: ProductRequest, who: str = Depends(client)):
    """같은 돈을 연금보험·펀드·예금에 넣었을 때를 같은 시나리오로 비교."""
    import numpy as np

    from . import mortality, product as prod_mod
    from .config import Person
    from .economy_v2 import EconomyV2, generate
    from .inputs import PROFILES

    p = fill(req.inputs)
    try:
        pr = prod_mod.AnnuityProduct(**req.product)
    except TypeError as ex:
        raise HTTPException(status_code=422, detail=f"상품 스펙을 해석할 수 없습니다: {ex}")
    n = max(500, min(int(req.paths), MAX_PATHS))
    me = [Person(age=p["age"], sex=p["sex"])]
    if pr.joint and p.get("spouse"):
        me.append(Person(age=p["s_age"], sex=p["s_sex"]))
    T = max(pr.start_age - me[0].age + 35, 40)
    eco = generate(EconomyV2.enhanced(), T, n, np.random.default_rng(42))
    qx = mortality.default_qx()
    out = prod_mod.compare_vehicles(pr, me, qx, eco, stock_weight=PROFILES[p["profile"]])
    out["guarantee"] = prod_mod.guarantee_value(pr, me, qx, eco["rate"], eco["cpi"])
    out["horizon_cap_years"] = T - (out["개시 나이"] - me[0].age)
    usage.record(who, "product_compare", calc_id(p), __version__, units=1)
    return {"model_version": __version__, "client": who, "result": out}


@app.get("/usage")
def get_usage(month: Optional[str] = None, who: str = Depends(client)):
    """우리 기관의 사용량. month 는 '2026-10' 형식. 다른 기관 것은 볼 수 없다."""
    if month is not None and not (len(month) == 7 and month[4] == "-" and month[:4].isdigit() and month[5:].isdigit()):
        raise HTTPException(status_code=422, detail="month 는 2026-10 같은 형식이어야 합니다.")
    return usage.summary(who, month)
