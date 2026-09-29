"""사망률 모듈.

기본값(v13): 통계청 「생명표」 2024 완전생명표(1세별, KOSIS DT_1B42)의 성별 사망확률.
100세 이상은 90~99세 사망력에 Gompertz(로그-선형)를 맞춰 109세까지 외삽하고 110세에서 종료.
CSV 형식: age,qx_M,qx_F (retire_sim/data/life_table_2024.csv). gompertz_qx()는 이전 임시값(비교용).
"""
import os
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


_DEFAULT_TABLE = os.path.join(os.path.dirname(__file__), "data", "life_table_2024.csv")


def default_qx(max_age: int = 110) -> dict:
    """기본 사망률: 통계청 2024 완전생명표(0~99세) + 100~109세 Gompertz 외삽. 파일이 없으면 임시 Gompertz."""
    if os.path.exists(_DEFAULT_TABLE):
        return load_life_table(_DEFAULT_TABLE, max_age)
    return gompertz_qx(max_age)


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


def simulate_life(qx: np.ndarray, start_age: int, T: int, n: int, rng, care) -> tuple:
    """다중상태(건강/간병/사망) 경로. 반환: alive (n, T+1), in_care (n, T+1).

    건강 상태 사망률 q_H는 전체 사망률 q_x가 보존되도록 보정:
    q_x ≈ (1-p)·q_H + p·m·q_H  →  q_H = q_x / (1 + (m-1)·p),
    p(x) = 간병 유병률 근사 = i(x)·D(x),  D(x) = 1 / (m·q_x + rec) (평균 간병 기간, 최대 10년)
    """
    alive = np.ones((n, T + 1), bool); incare = np.zeros((n, T + 1), bool)
    m, rec = care.mort_mult, care.recovery
    for t in range(T):
        x = min(start_age + t, len(qx) - 1)
        q = qx[x]; i = care.inc(x)
        D = min(1.0 / max(m * q + rec, 1e-9), 10.0)
        p = min(i * D, 0.5)
        qH = min(q / (1 + (m - 1) * p), 1.0); qC = min(m * qH, 1.0)
        a, c = alive[:, t], incare[:, t]
        u = rng.random(n); v = rng.random(n)
        die = np.where(c, u < qC, u < qH)
        to_care = ~c & ~die & (v < i)
        recover = c & ~die & (v < rec)
        alive[:, t + 1] = a & ~die
        incare[:, t + 1] = alive[:, t + 1] & ((c & ~recover) | to_care)
    return alive, incare
