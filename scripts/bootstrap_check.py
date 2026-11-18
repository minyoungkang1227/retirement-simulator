import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""(v59) 검증: 주식 수익률이 '해마다 독립·정규'라는 가정이 결과를 얼마나 바꾸나 — 과거 KOSPI의 블록 부트스트랩과 비교.

실행: PYTHONPATH=. python scripts/bootstrap_check.py

모형의 주식 로그수익 = 금리 + 위험프리미엄 + 변동성 × Z. 여기서 Z(표준화 충격)만 바꿔 본다.
  ① 모형        Z를 모형이 뽑는다(해마다 독립, 정규)
  ② 과거·독립   과거 KOSPI 연 로그수익을 표준화한 값에서 한 해씩 무작위로 뽑는다 → 분포의 모양(왜도·꼬리)만 과거를 따른다
  ③ 과거·블록   같은 값에서 연속 5년(또는 10년)씩 뽑는다(원형 블록) → 모양에 더해 오르내림의 순서까지 과거를 따른다
평균·변동성·금리·물가·수명·간병은 세 경우 모두 같다. 그래서 ②−①은 모양의 효과, ③−②는 순서(평균회귀·추세)의 효과다.

자료: scripts/data/kospi_annual.csv — KOSPI 연말 종가 1995~2025(한국거래소, FinanceDataReader 'KS11'로 2026-11-19 받음).
한계: 30개 연도뿐이고 가격지수(배당 제외)다. 표준화하므로 수준은 영향이 없지만 표본이 작아 블록 결과는 참고용이다.
모형의 점프(연 5% 확률)는 세 경우 모두 그대로 더해지므로 ②·③은 급락을 조금 겹쳐 센다(보수적인 쪽).
"""
import contextlib
import csv
import io
from dataclasses import replace

import numpy as np

from retire_sim import __version__

DATA = os.path.join(os.path.dirname(__file__), "data", "kospi_annual.csv")
T_MAX = 90


def hist_z() -> np.ndarray:
    """과거 KOSPI 연 로그수익의 표준화 값(평균 0, 표준편차 1)."""
    with open(DATA, encoding="utf-8") as f:
        close = np.array([float(r["close"]) for r in csv.DictReader(f)])
    lr = np.diff(np.log(close))
    return (lr - lr.mean()) / lr.std()


def block_paths(z: np.ndarray, n: int, T: int, block: int, seed: int = 20261119) -> np.ndarray:
    """원형 블록 부트스트랩: 시작 해를 무작위로 잡아 block년씩 이어 붙인다. block=1이면 한 해씩 독립 추출."""
    rng = np.random.default_rng(seed)
    k = -(-T // block)
    start = rng.integers(0, len(z), (n, k))
    idx = (start[:, :, None] + np.arange(block)[None, None, :]) % len(z)
    return z[idx.reshape(n, -1)[:, :T]]


def shape(z: np.ndarray) -> dict:
    return {"왜도": float(np.mean(z ** 3)), "초과첨도": float(np.mean(z ** 4) - 3),
            "1년 자기상관": float(np.corrcoef(z[:-1], z[1:])[0, 1])}


def run_all(n: int = 10_000) -> list:
    sys.path.insert(0, os.path.dirname(__file__))
    with contextlib.redirect_stdout(io.StringIO()):
        import readme_numbers as R                     # 기준 가구 셋과 웹 앱과 같은 설정(가져올 때 README 표를 한 번 계산한다)
    z = hist_z()
    arms = [("① 모형(독립·정규)", None), ("② 과거·독립", block_paths(z, n, T_MAX, 1)),
            ("③ 과거·5년 블록", block_paths(z, n, T_MAX, 5)), ("③' 과거·10년 블록", block_paths(z, n, T_MAX, 10))]
    rows = []
    for name, hh in (("52·50세 부부", R.couple_52()), ("30대 맞벌이", R.couple_30()), ("65·63세 은퇴 부부", R.couple_65())):
        probs = [R.show(R.run(hh, eco=replace(R.ECO, stock_z=sz), cfg=replace(R.CFG, n_paths=n)))[0] for _, sz in arms]
        rows.append((name, probs))
    return [a for a, _ in arms], rows


if __name__ == "__main__":
    z = hist_z()
    print(f"모델 v{__version__} · 1만 경로 · 과거 KOSPI {len(z)}개 연도")
    print("과거 표준화 충격:", {k: round(v, 2) for k, v in shape(z).items()})
    arms, rows = run_all()
    print(f"\n{'기본생활 유지 확률':<14}" + "".join(f"{a:>16}" for a in arms))
    for name, probs in rows:
        print(f"{name:<14}" + "".join(f"{p * 100:>15.1f}%" for p in probs))
        print(f"{'  모형 대비':<14}" + "".join(f"{(p - probs[0]) * 100:>+14.1f}%p" for p in probs))
