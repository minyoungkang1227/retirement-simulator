"""사망률 모듈.

v1은 Gompertz 근사(임시값). 실제 사용 시 통계청(KOSIS) 완전생명표 CSV를
load_life_table()로 불러와 교체한다. CSV 형식: age,qx_M,qx_F
"""
import numpy as np
import csv

# 임시 Gompertz 파라미터 mu(x) = a*exp(b*x) — 한국 대략 수준에 맞춘 값, 실데이터로 교체 필요
_GOMPERTZ = {"M": (1.65e-5, 0.100), "F": (4.28e-6, 0.108)}  # 60세 기대여명 남 23.5 / 여 29.5년에 보정


def gompertz_qx(max_age: int = 110) -> dict:
    ages = np.arange(max_age + 1)
    out = {}
    for sex, (a, b) in _GOMPERTZ.items():
        H = a / b * (np.exp(b * (ages + 1)) - np.exp(b * ages))  # 1년 누적위험
        q = 1 - np.exp(-H)
        q[-1] = 1.0
        out[sex] = np.clip(q, 0, 1)
    return out


def load_life_table(path: str, max_age: int = 110) -> dict:
    qM, qF = np.ones(max_age + 1), np.ones(max_age + 1)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = int(row["age"])
            if a <= max_age:
                qM[a], qF[a] = float(row["qx_M"]), float(row["qx_F"])
    return {"M": qM, "F": qF}


def life_expectancy(qx: np.ndarray, age: int) -> float:
    surv = np.cumprod(1 - qx[age:])
    return float(surv.sum() + 0.5)


def simulate_alive(qx: np.ndarray, start_age: int, T: int, n: int, rng) -> np.ndarray:
    """alive[p, t] : t년 시작 시점 생존 여부 (t=0은 현재)."""
    ages = np.minimum(start_age + np.arange(T), len(qx) - 1)
    die = rng.random((n, T)) < qx[ages]
    alive = np.ones((n, T + 1), dtype=bool)
    alive[:, 1:] = np.cumprod(~die, axis=1).astype(bool)
    return alive
