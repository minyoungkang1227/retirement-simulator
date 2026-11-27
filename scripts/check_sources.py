"""바깥 자료의 갱신 시한 점검: python scripts/check_sources.py

retire_sim/sources.py 의 REFRESH_MONTHS 로 시한을 계산한다. 시한이 지난 자료가 있으면 종료 코드 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retire_sim.sources import due_status  # noqa: E402

if __name__ == "__main__":
    rows = due_status()
    order = {"지남": 0, "이번 달": 1, "정상": 2, "수동 확인": 3}
    for s, st, due in sorted(rows, key=lambda r: (order[r[1]], r[0].key)):
        print(f"[{st}] {s.key:16s} 반영 {s.as_of:24s} 시한 {due or '-':8s} {s.name}")
    late = [s.key for s, st, _ in rows if st == "지남"]
    print(f"\n시한 지남 {len(late)}건" + (f": {', '.join(late)}" if late else ""))
    raise SystemExit(1 if late else 0)
