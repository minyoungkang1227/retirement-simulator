"""API 사용 기록: 종량 과금의 근거이자 상담 로그(감사 기록).

무엇을 남기나 — 언제 · 어느 기관이 · 어느 기능을 · 몇 번 계산했나 · 어떤 입력(계산 ID)과 어떤 모델 버전이었나.
무엇을 **안** 남기나 — 고객 이름·상담자·나이·자산 같은 입력값. 계산 ID 는 입력의 해시라서
원문은 복원되지 않지만, 기관이 자기 기록과 대조해 "그때 그 계산"을 가려낼 수는 있다
(같은 입력 + 같은 모델 버전 → 같은 결과이므로 그 시점의 숫자를 그대로 재현할 수 있다).

켜는 법: 환경변수 RETIRE_USAGE_LOG=/경로/usage.jsonl (한 줄에 한 건, 덧붙이기만 한다).
설정하지 않으면 서버가 떠 있는 동안의 메모리에만 쌓이고, 재시작하면 사라진다 — 과금에 쓰려면 반드시 켠다.

기록이 실패해도(디스크 가득 참 등) 계산 응답은 막지 않는다. 대신 실패 횟수를 세어 `/usage` 에 보인다.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_MEM: List[dict] = []
_FAILED = 0


def _path() -> Optional[str]:
    p = os.environ.get("RETIRE_USAGE_LOG", "").strip()
    return p or None


def record(client: str, endpoint: str, calc_id: str, model_version: str,
           units: int = 1, now: Optional[datetime] = None) -> dict:
    """한 건을 남긴다. units 는 엔진을 돌린 횟수(비교는 기준 + 선택지 수)."""
    global _FAILED
    row = {"ts": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "client": client, "endpoint": endpoint, "calc_id": calc_id,
           "model_version": model_version, "units": int(units)}
    with _LOCK:
        _MEM.append(row)
        path = _path()
        if path:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                _FAILED += 1
    return row


def _rows() -> List[dict]:
    """파일이 설정돼 있으면 파일(재시작 후에도 남는 기록)을, 아니면 메모리를 읽는다."""
    path = _path()
    if not path:
        with _LOCK:
            return list(_MEM)
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass                      # 반쯤 쓰인 줄은 건너뛴다
    except OSError:
        pass
    return out


def summary(client: str, month: Optional[str] = None) -> Dict[str, Any]:
    """한 기관의 사용량. month='2026-10' 처럼 주면 그 달만 센다. 다른 기관 것은 볼 수 없다."""
    rows = [r for r in _rows() if r.get("client") == client and (not month or str(r.get("ts", "")).startswith(month))]
    calls = Counter(r["endpoint"] for r in rows)
    units = Counter()
    for r in rows:
        units[r["endpoint"]] += int(r.get("units", 1))
    return {"client": client, "month": month or "전체",
            "calls": sum(calls.values()), "engine_runs": sum(units.values()),
            "by_endpoint": {k: {"calls": calls[k], "engine_runs": units[k]} for k in sorted(calls)},
            "model_versions": sorted({str(r.get("model_version")) for r in rows}),
            "persistent": _path() is not None, "log_write_failures": _FAILED,
            "note": "호출 수(calls)와 엔진 실행 수(engine_runs)를 따로 센다. 비교는 한 호출에 엔진을 여러 번 돈다."}


def reset_for_tests() -> None:
    global _FAILED
    with _LOCK:
        _MEM.clear()
        _FAILED = 0
