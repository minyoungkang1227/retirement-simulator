"""태블릿 앱(docs/app/)에 넣을 계산 엔진 묶음을 만든다.

    python scripts/build_app.py

retire_sim 의 계산 파일과 생명표를 docs/app/engine.zip 하나로 묶는다. 앱은 이 파일을 내려받아
브라우저 안의 파이썬(Pyodide)에 풀고 retire_sim.lite 를 부른다 — 웹 앱과 같은 코드다.
엔진을 고치면 이 스크립트를 다시 돌려야 하며, 안 돌리면 tests/test_engine.py 가 실패해 알려준다.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "retire_sim"
OUT = ROOT / "docs" / "app" / "engine.zip"
META = ROOT / "docs" / "app" / "engine.json"
# 브라우저에서 쓰지 않거나(서버·외부 데이터) 브라우저에 없는 라이브러리가 필요한 파일
SKIP = {"api.py", "market_data.py", "calibrate.py", "usage.py"}


def engine_files() -> list:
    files = sorted(f for f in PKG.glob("*.py") if f.name not in SKIP)
    files += sorted((PKG / "data").glob("*.csv"))
    return files


def digest() -> str:
    """엔진 파일 내용의 지문 — 앱 묶음이 최신인지 확인하는 데 쓴다."""
    h = hashlib.sha256()
    for f in engine_files():
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build() -> dict:
    sys.path.insert(0, str(ROOT))
    from retire_sim import __version__
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in engine_files():
            info = zipfile.ZipInfo(f.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())       # 날짜를 고정해 같은 엔진이면 같은 파일이 나오게
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(buf.getvalue())
    meta = {"model_version": __version__, "digest": digest(), "files": len(engine_files()),
            "bytes": OUT.stat().st_size}
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(build())
