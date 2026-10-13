"""화이트라벨: 기관마다 다른 이름·색·로고·고지문을 한곳에서 관리한다.

왜 필요한가
  이 도구는 설계사가 고객 앞에서 쓰는 물건이다. 리포트에 우리 이름이 박혀 있으면
  기관은 쓰지 못한다. 기관 계약의 전제 조건이라 코드 곳곳의 색·문구를 설정 하나로 뺐다.

쓰는 법
  · 코드에서  Brand(name="○○생명", primary="#0A3D62", ...)
  · 환경변수  RETIRE_BRAND='{"name": "○○생명", "primary": "#0A3D62"}'
  · 파일      Brand.load("brand.json")
  색은 #RGB·#RRGGBB 만 받는다 — 리포트 CSS에 그대로 들어가므로 임의 문자열을 넣지 않는다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_ENV = "RETIRE_BRAND"


def _color(value: str, fallback: str) -> str:
    """CSS에 넣어도 안전한 색만 통과시킨다."""
    return value if isinstance(value, str) and _HEX.match(value) else fallback


@dataclass
class Brand:
    name: str = "노후나침반"                 # 서비스·기관 이름
    tagline: str = "확률로 보는 노후 점검"
    primary: str = "#1F6F78"                # 제목·강조
    accent: str = "#E2725B"                 # 경고·부족분
    ink: str = "#1B2B2E"
    muted: str = "#5A6B6E"
    line: str = "#CBD7DB"
    logo_url: str = ""                      # 비우면 로고 자리를 쓰지 않는다
    footer: str = "노후나침반 · retire-compass.streamlit.app"
    disclaimer: str = "이 리포트는 투자 권유나 세무 자문이 아닙니다. 실제 결정 전에 전문가와 상의하시기 바랍니다."

    def __post_init__(self):
        for f in ("primary", "accent", "ink", "muted", "line"):
            setattr(self, f, _color(getattr(self, f), Brand.__dataclass_fields__[f].default))
        if self.logo_url and not self.logo_url.startswith(("https://", "data:image/")):
            self.logo_url = ""              # http·javascript 등은 쓰지 않는다

    # ── 불러오기 ──
    @classmethod
    def from_dict(cls, d: dict | None) -> "Brand":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    @classmethod
    def from_env(cls, var: str = _ENV) -> "Brand":
        raw = os.environ.get(var)
        if not raw:
            return cls()
        try:
            return cls.from_dict(json.loads(raw))
        except (ValueError, TypeError):
            return cls()                    # 설정이 깨져도 서비스는 떠야 한다

    @classmethod
    def load(cls, path: str) -> "Brand":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Brand()
