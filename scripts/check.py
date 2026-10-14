import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""작업을 마칠 때마다 돌리는 **오류 검증 + 최적화 점검** 한 번에.

    PYTHONPATH=. python scripts/check.py          # 전체 (테스트 포함, 약 3분)
    PYTHONPATH=. python scripts/check.py --fast   # 테스트 빼고 빠른 점검만 (약 20초)

맨 끝에 '고칠 것 / 지울 것'만 모아서 보여 준다. 아무것도 없으면 '깨끗합니다'.

주의: '언급 0곳'이라고 바로 지우면 안 된다. v41에서 run_example.py·sensitivity.py 가
0곳으로 나왔는데 실제로는 README 그림을 만들고 있었다. 무엇을 만드는지 먼저 확인할 것.
"""
import ast
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAST = "--fast" in sys.argv
todo, trash = [], []


def head(t):
    print(f"\n{'─' * 58}\n{t}\n{'─' * 58}")


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, env={**os.environ, "PYTHONPATH": "."})
    return r.returncode, (r.stdout + r.stderr).strip()


# ── ① 오류 검증 ────────────────────────────────────────────────────────
head("① 오류 검증")

if FAST:
    print("  테스트: 건너뜀 (--fast)")
else:
    t0 = time.time()
    code, out = sh("python -m pytest -q", timeout=1800)
    last = [l for l in out.splitlines() if "passed" in l or "failed" in l]
    print(f"  테스트: {last[-1] if last else out[-200:]}  ({time.time() - t0:.0f}초)")
    if code:
        todo.append("테스트 실패 — python -m pytest -q 로 확인")
        for l in out.splitlines():
            if l.startswith("FAILED"):
                print(f"    {l}")

code, out = sh("python scripts/benchmark_analytic.py")
err = [l for l in out.splitlines() if "최대 오차" in l]
print(f"  해석해 대조: {err[-1] if err else '실행 실패'}")
if code or not err or not err[-1].endswith("0.0"):
    todo.append("해석해 대조가 어긋남 — 모델 계산이 바뀌었는지 확인")

for cmd, name in (("python scripts/make_sources_doc.py --check", "자료 출처 문서"),
                  ("python scripts/sync_notebook.py --check", "노트북↔모듈")):
    code, out = sh(cmd)
    print(f"  {name}: {out.splitlines()[-1] if out else '?'}")
    if code:
        todo.append(f"{name} 어긋남 — {cmd.replace(' --check', '')} 실행")

# 버전 일관성
ver = re.search(r'__version__ = "(\d+)"', (ROOT / "retire_sim" / "__init__.py").read_text()).group(1)
app = re.search(r"^APP_EXPECTS_MODEL = (\d+)", (ROOT / "streamlit_app.py").read_text(), re.M).group(1)
chg = re.search(r"^## v(\d+)", (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
print(f"  버전: 모델 v{ver} · 앱 기대 v{app} · CHANGELOG v{chg}")
if not (ver == app == chg):
    todo.append(f"버전 불일치 (모델 {ver} / 앱 {app} / CHANGELOG {chg})")

# ── ② 최적화 점검 ──────────────────────────────────────────────────────
head("② 최적화 점검")

pyfiles = [p for p in ROOT.rglob("*.py") if "__pycache__" not in str(p) and ".git" not in str(p)]
blob = "\n".join(p.read_text() for p in pyfiles)
NB = (ROOT / "notebooks" / "retire_sim_colab.ipynb")
nb_text = NB.read_text() if NB.exists() else ""

dead = []
for p in pyfiles:
    body = p.read_text()
    tree = ast.parse(body)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                nm = (a.asname or a.name).split(".")[0]
                if nm in ("*", "annotations"):
                    continue
                if len(re.findall(rf"\b{re.escape(nm)}\b", body)) <= 1:
                    dead.append(f"안 쓰는 import  {p.relative_to(ROOT)}:{n.lineno}  {nm}")
    if p.name.startswith("test_") or p.parent.name == "scripts":
        continue
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("__"):
            uses = len(re.findall(rf"\b{re.escape(n.name)}\b", blob))
            in_nb = n.name in nb_text          # 노트북에서 쓰는 함수는 살아 있다
            decorated = bool(n.decorator_list)  # 라우트 등은 호출 흔적이 없다
            if uses <= 1 and not in_nb and not decorated:
                dead.append(f"죽은 함수      {p.relative_to(ROOT)}:{n.lineno}  {n.name}()")
print("  죽은 코드:", "없음" if not dead else "")
for d in dead:
    print(f"    {d}")
trash += dead

refs = "\n".join(p.read_text(errors="ignore") for p in ROOT.rglob("*")
                 if p.is_file() and p.suffix in (".md", ".py", ".yml", ".ipynb") and ".git" not in str(p))
orphan = [p.relative_to(ROOT) for p in (ROOT / "docs" / "images").glob("*") if p.name not in refs]
print("  고아 파일:", "없음" if not orphan else "")
for o in orphan:
    print(f"    {o}")
    trash.append(f"고아 파일 {o} (아무도 참조하지 않음)")

# 스크립트는 '언급 0곳'이어도 지우면 안 된다 — 무엇을 만드는지 먼저 본다
quiet = []
for p in sorted((ROOT / "scripts").glob("*.py")):
    if len(re.findall(re.escape(p.name), refs)) <= 1:
        makes = re.findall(r'savefig\([^)]*"([^"]+)"|write_text\(|OUT\.write_text', p.read_text())
        quiet.append(f"{p.name} (문서 언급 없음{' · 파일을 만드는 스크립트라 확인 필요' if makes else ''})")
if quiet:
    print("  문서에 언급 없는 스크립트 (지우기 전 무엇을 만드는지 확인):")
    for q in quiet:
        print(f"    {q}")

code, out = sh("""python -c "
import time, numpy as np, ast, pathlib
from retire_sim import engine_tax, mortality
from retire_sim.config import CareMarkov, Household, Person, SimConfig
from retire_sim.economy_v2 import EconomyV2
from retire_sim.tax import TaxConfig
hh=Household(members=[Person(60,'M',nps_monthly=110,private_pension_annual=600,private_pension_start=60),Person(58,'F',nps_monthly=50)],liquid_assets=50_000,stock_weight=0.4,annual_spending=3600,essential=3000,lifestyle=600,legacy_target=10_000)
E,T,Q=EconomyV2.enhanced(),TaxConfig(overseas_share=0.2),mortality.default_qx()
c=SimConfig(n_paths=10000,care=CareMarkov())
r=engine_tax.run(hh,c,e=E,tc=T,qx_table=Q)
ts=[]
for _ in range(3):
    t=time.time(); engine_tax.run(hh,c,e=E,tc=T,qx_table=Q); ts.append(time.time()-t)
src=open('retire_sim/engine_tax.py').read()
f=max((x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef)),key=lambda x:x.end_lineno-x.lineno)
ln=sum(1 for p in pathlib.Path('retire_sim').glob('*.py') for _ in open(p))
print(f\\"{sorted(ts)[1]:.2f}|{f.name}|{f.end_lineno-f.lineno+1}|{ln}|{engine_tax.summarize(r)['기본생활 유지 확률']*100:.1f}\\")
" """)
if code == 0 and "|" in out:
    sec, fn, flen, lines, prob = out.splitlines()[-1].split("|")
    print(f"  속도: 1만 경로 {sec}초 (3회 중앙값)")
    print(f"  가장 긴 함수: {fn}() {flen}줄 · retire_sim 전체 {lines}줄")
    print(f"  기준 가구 기본생활 유지 확률: {prob}%  (회귀 범위 65~75%)")
    if not 65 <= float(prob) <= 75:
        todo.append(f"기준 가구 결과가 회귀 범위를 벗어남 ({prob}%) — scripts/ablate.py 로 원인 확인")
    if float(sec) > 3:
        todo.append(f"1만 경로가 {sec}초로 느려짐")
else:
    todo.append("속도·회귀 점검 실행 실패")

# ── ③ 결론 ────────────────────────────────────────────────────────────
head("③ 결론")
if not todo and not trash:
    print("  깨끗합니다. 고칠 것도 지울 것도 없습니다.")
else:
    if todo:
        print("  ■ 고칠 것")
        for x in todo:
            print(f"    - {x}")
    if trash:
        print("  ■ 지울 것 (지우기 전 무엇을 만드는지 확인)")
        for x in trash:
            print(f"    - {x}")
raise SystemExit(1 if todo else 0)
