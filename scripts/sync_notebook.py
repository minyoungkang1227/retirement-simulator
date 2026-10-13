import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""노트북의 %%writefile 셀을 저장소의 실제 파일 내용과 맞춘다.

노트북은 Colab에서 모듈을 풀어쓰기 위해 파일 내용을 셀 안에 복사해 두는데,
모듈만 고치고 노트북을 안 고치면 노트북 실행이 모듈을 '옛 코드로 되돌려' 버린다(실제로 두 번 겪었다).
그래서 모듈을 고친 뒤에는 항상 이걸 돌린다:

    python scripts/sync_notebook.py          # 맞춘다
    python scripts/sync_notebook.py --check  # 다른 데가 있는지만 본다(CI·테스트용)
"""
import json
import pathlib

NB = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "retire_sim_colab.ipynb"
ROOT = NB.parent.parent


def cells(nb):
    """(셀, 대상 파일 경로)를 %%writefile 셀마다 돌려준다."""
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        head = src.split("\n", 1)[0].strip()
        if head.startswith("%%writefile"):
            yield c, ROOT / head.split(None, 1)[1].strip()


def expected(cell, path):
    """그 셀이 가져야 할 source: 첫 줄(%%writefile ...) + 파일 내용 그대로.
    ipynb의 source는 '줄 끝 개행을 포함한 줄들의 리스트'라서 splitlines(keepends)를 쓴다."""
    head = "".join(cell["source"]).split("\n", 1)[0]
    return [head + "\n"] + path.read_text().splitlines(keepends=True)


def main(check=False):
    nb = json.loads(NB.read_text())
    drift = []
    for cell, path in cells(nb):
        if not path.exists():
            drift.append(f"{path.relative_to(ROOT)} (파일이 없음)"); continue
        want = expected(cell, path)
        if "".join(cell["source"]) != "".join(want):
            drift.append(str(path.relative_to(ROOT)))
            cell["source"] = want
    if check:
        if drift:
            print("노트북이 모듈과 다릅니다 — python scripts/sync_notebook.py 로 맞추세요:")
            for d in drift:
                print("  -", d)
            return 1
        print("노트북과 모듈이 일치합니다.")
        return 0
    if drift:
        NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
        print(f"{len(drift)}개 셀을 맞췄습니다:")
        for d in drift:
            print("  -", d)
    else:
        print("이미 일치합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check="--check" in sys.argv))
