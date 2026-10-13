"""API·화이트라벨·입력 분리 테스트.

fastapi 가 없는 환경(예: Streamlit Cloud 배포본)에서는 API 테스트만 건너뛴다.
"""
import json
import os

import pytest

from retire_sim import __version__
from retire_sim.branding import Brand
from retire_sim.inputs import DEFAULTS, build_case, defaults, fill

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

KEY = "test-key-123"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("RETIRE_API_KEYS", f"테스트기관:{KEY}")
    import importlib

    from retire_sim import api
    importlib.reload(api)
    return TestClient(api.app)


H = {"X-API-Key": KEY}
SMALL = {"paths": 800}


# ── 인증 ──
def test_api_requires_key(client):
    assert client.get("/defaults").status_code == 401
    assert client.get("/defaults", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/defaults", headers=H).status_code == 200


def test_health_and_version_are_open(client):
    assert client.get("/health").json()["model_version"] == __version__
    assert client.get("/version").json()["model_version"] == __version__


# ── 계산 ──
def test_simulate_fills_defaults_and_is_reproducible(client):
    r1 = client.post("/simulate", headers=H, json={"inputs": {"age": 55, **SMALL}}).json()
    r2 = client.post("/simulate", headers=H, json={"inputs": {"age": 55, **SMALL}}).json()
    assert r1["calc_id"] == r2["calc_id"]                                  # 같은 입력 → 같은 ID
    assert r1["summary"] == r2["summary"]                                  # 같은 시드 → 같은 결과
    assert r1["model_version"] == __version__ and r1["client"] == "테스트기관"
    assert 0 <= r1["summary"]["기본생활 유지 확률"] <= 1
    other = client.post("/simulate", headers=H, json={"inputs": {"age": 56, **SMALL}}).json()
    assert other["calc_id"] != r1["calc_id"]


def test_compare_returns_paired_intervals(client):
    body = {"inputs": {"age": 50, **SMALL},
            "variants": [{"spend_mult": 0.9}, {"retire_delta": 3}],
            "labels": ["생활비 10% 줄이기", "은퇴 3년 늦추기"]}
    rows = client.post("/compare", headers=H, json=body).json()["results"]
    assert len(rows) == 3 and rows[0]["label"] == "기준" and rows[0]["diff"] == 0.0
    for row in rows[1:]:
        assert row["ci"] > 0 and row["diff"] > 0                           # 둘 다 도움이 되는 선택지
        assert isinstance(row["significant"], bool)


def test_report_endpoint_applies_brand(client):
    body = {"inputs": {"age": 60, **SMALL}, "client_name": "김<철>수",
            "brand": {"name": "○○생명", "primary": "#0A3D62", "footer": "○○생명 자산관리"}}
    out = client.post("/report", headers=H, json=body).json()
    h = out["html"]
    assert h.startswith("<!doctype html>") and out["content_type"] == "text/html"
    assert "○○생명 노후 점검 리포트" in h and "#0A3D62" in h and "○○생명 자산관리" in h
    assert "김&lt;철&gt;수" in h                                            # 이스케이프
    assert out["calc_id"] and out["model_version"] == __version__


def test_product_endpoint(client):
    body = {"inputs": {"age": 45, "sex": "M"},
            "product": {"monthly_premium": 50, "pay_years": 20, "start_age": 65}, "paths": 800}
    out = client.post("/product/compare", headers=H, json=body).json()["result"]
    assert out["개시 나이"] == 65
    assert out["살아 있는 동안 바닥날 확률"]["연금보험"] == 0.0
    assert out["살아 있는 동안 바닥날 확률"]["펀드"] > 0


# ── 입력 검증 ──
@pytest.mark.parametrize("bad", [{"profile": "없는성향"}, {"paths": 10 ** 9}, {"sex": 123}])
def test_bad_input_gives_422_not_500(client, bad):
    assert client.post("/simulate", headers=H, json={"inputs": bad}).status_code == 422


def test_too_many_variants(client):
    body = {"inputs": SMALL, "variants": [{"spend_mult": 0.9}] * 20}
    assert client.post("/compare", headers=H, json=body).status_code == 422


# ── 화이트라벨 (fastapi 없이도 돈다) ──
def test_brand_rejects_unsafe_values():
    b = Brand(primary="red; } body { display:none", logo_url="javascript:alert(1)")
    assert b.primary == Brand().primary and b.logo_url == ""
    assert Brand(primary="#0A3D62").primary == "#0A3D62" and Brand(primary="#ABC").primary == "#ABC"
    assert Brand(logo_url="https://x/l.png").logo_url == "https://x/l.png"


def test_brand_from_env_survives_bad_config(monkeypatch):
    monkeypatch.setenv("RETIRE_BRAND", json.dumps({"name": "환경변수생명", "accent": "#C00", "몰라": 1}))
    b = Brand.from_env()
    assert b.name == "환경변수생명" and b.accent == "#C00"                  # 모르는 키는 무시
    monkeypatch.setenv("RETIRE_BRAND", "{깨진 json")
    assert Brand.from_env().name == Brand().name                           # 설정이 깨져도 기본값으로 뜬다


# ── 입력 분리 ──
def test_inputs_fill_and_build():
    p = fill({"age": 40})
    assert p["age"] == 40 and p["profile"] == DEFAULTS["profile"] and len(p) == len(DEFAULTS)
    assert defaults() is not DEFAULTS and defaults()["events"] is not DEFAULTS["events"]   # 깊은 복사
    hh, cfg, eco, tc, house, biz, q, ad = build_case(p, {})
    assert hh.members[0].age == 40 and cfg.seed == 42
    hh2, *_ = build_case(p, {"retire_delta": 2})
    assert hh2.retire_age == hh.retire_age + 2


def test_app_and_library_share_one_builder():
    """앱이 자체 build 를 다시 만들면 API 와 결과가 어긋난다 — 그걸 막는다."""
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py"), encoding="utf-8").read()
    assert "from retire_sim.inputs import" in src
    assert not re.search(r"^def build\(p", src, re.M)
    assert not re.search(r"^def total_assets\(", src, re.M)
