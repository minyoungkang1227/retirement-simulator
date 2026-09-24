"""결과 지표."""
import numpy as np


def summarize(res: dict) -> dict:
    d, y = res["depleted_at"], res["youngest_age"]
    dep = d >= 0
    ages = d[dep] + y
    W_real = res["W_real"]
    p = float(dep.mean()); se = float(np.sqrt(p * (1 - p) / len(d)))
    return {
        "고갈확률(생존 중)": p,
        "95% 신뢰구간 ±": 1.96 * se,
        "고갈 나이 중앙값(최연소 기준)": float(np.median(ages)) if dep.any() else None,
        "고갈 나이 10%분위": float(np.percentile(ages, 10)) if dep.any() else None,
        "가구 존속기간 중앙값(년)": float(np.median(res["last_alive_t"] + 1)),
        "10년 후 실질자산 중앙값(만원)": float(np.median(W_real[:, 10])),
    }


def depletion_curve(res: dict) -> tuple:
    """나이별 누적 고갈확률."""
    T, y, d = res["T"], res["youngest_age"], res["depleted_at"]
    ts = np.arange(T + 1)
    cum = np.array([(d >= 0) & (d <= t) for t in ts]).mean(axis=1)
    return ts + y, cum


def fan(res: dict, pct=(5, 25, 50, 75, 95)) -> tuple:
    """생존 가구 기준 실질자산 분위수."""
    W, alive = res["W_real"], res["hh_alive"]
    out = []
    for t in range(W.shape[1]):
        v = W[alive[:, t], t]
        out.append(np.percentile(v, pct) if v.size > 100 else [np.nan] * len(pct))
    return np.arange(W.shape[1]) + res["youngest_age"], np.array(out)
