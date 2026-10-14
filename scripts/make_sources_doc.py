import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""retire_sim/sources.py 로부터 docs/SOURCES.md 를 만든다.

    python scripts/make_sources_doc.py          # 만든다
    python scripts/make_sources_doc.py --check  # 어긋났는지만 본다 (테스트·CI용)
"""
import pathlib

from retire_sim import __version__
from retire_sim.sources import SOURCES, refresh_order

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "SOURCES.md"


def build() -> str:
    L = ["# 자료 출처와 갱신", "",
         f"모델 v{__version__} 기준. **이 파일은 `retire_sim/sources.py` 에서 자동으로 만듭니다** — "
         "직접 고치지 마시고 그 파일을 고친 뒤 `python scripts/make_sources_doc.py` 를 돌리세요.", "",
         "이 계산기는 추측한 숫자를 쓰지 않습니다. 아래가 바깥에서 가져온 자료 전부입니다.", "",
         "## 한눈에 보기", "",
         "| 자료 | 만든 곳 | 반영 시점 | 갱신 주기 |", "|---|---|---|---|"]
    for s in SOURCES:
        L.append(f"| {s.name} | {s.org} | {s.as_of} | {s.cycle} |")
    L += ["", "## 갱신이 급한 순서", "",
          "'늦으면 결과가 많이 달라지는 것'을 위에 놓았습니다.", ""]
    for i, s in enumerate(refresh_order(), 1):
        L += [f"### {i}. {s.name}", "",
              f"- **쓰이는 곳**: {s.used_for}",
              f"- **반영 시점**: {s.as_of} · **갱신 주기**: {s.cycle}",
              f"- **어디서**: {s.where}",
              f"- **어떻게**: {s.how}",
              f"- **안 갱신하면**: {s.stale}", ""]
    L += ["## 갱신한 뒤 꼭 할 일", "",
          "1. `retire_sim/sources.py` 의 `as_of` 를 새 시점으로 고친다",
          "2. `python scripts/make_sources_doc.py` — 이 문서를 다시 만든다",
          "3. `python -m pytest -q` — 대표 가구 결과가 범위를 벗어나면 테스트가 알려 준다",
          "4. 결과가 눈에 띄게 달라졌으면 `python scripts/ablate.py` 로 어느 요소 때문인지 확인한다",
          "5. `retire_sim/__init__.py` 의 `__version__` 을 올리고 `CHANGELOG.md` 에 무엇이 바뀌었는지 적는다",
          "6. `python scripts/sync_notebook.py` — 노트북을 모듈과 맞춘다", ""]
    return "\n".join(L)


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print("docs/SOURCES.md 가 retire_sim/sources.py 와 다릅니다 — "
                  "python scripts/make_sources_doc.py 로 다시 만드세요.")
            raise SystemExit(1)
        print("docs/SOURCES.md 가 최신입니다.")
    else:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(text)
        print(f"{OUT.relative_to(OUT.parent.parent)} 를 만들었습니다 ({len(text.splitlines())}줄).")
