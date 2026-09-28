"""결과 지표."""
import numpy as np


def prob_se(flags, antithetic=False):
    """고갈 여부(0/1) 평균의 표준오차. 반대 난수면 짝(i, i+h)의 평균으로 계산."""
    x = np.asarray(flags, float); n = len(x)
    if antithetic and n >= 4:
        h = n // 2
        pair = (x[:h] + x[h:2 * h]) / 2
        return float(pair.std(ddof=1) / np.sqrt(h))
    p = x.mean()
    return float(np.sqrt(p * (1 - p) / n))


def summarize(res: dict) -> dict:
    d, y = res["depleted_at"], res["youngest_age"]
    dep = d >= 0
    ages = d[dep] + y
    W_real = res["W_real"]
    p = float(dep.mean()); se = prob_se(dep, res.get("antithetic", False))
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


def compare(res_a: dict, res_b: dict) -> dict:
    """두 선택지의 고갈확률 차이와 '짝지은' 95% 신뢰구간 (공통 난수 전제).

    경로별 d_i = 1[A 고갈] − 1[B 고갈] 의 평균과 표준오차를 쓴다.
    같은 시나리오를 공유하므로 Var(A−B) = VarA + VarB − 2Cov(A,B) 가 각각의 분산 합보다 훨씬 작다.
    """
    a = (np.asarray(res_a["depleted_at"]) >= 0).astype(float)
    b = (np.asarray(res_b["depleted_at"]) >= 0).astype(float)
    d = a - b; n = len(d)
    if res_a.get("antithetic") and res_b.get("antithetic") and n >= 4:
        h = n // 2; x = (d[:h] + d[h:2 * h]) / 2; se = x.std(ddof=1) / np.sqrt(h)
    else:
        se = d.std(ddof=1) / np.sqrt(n)
    naive = np.sqrt(a.mean() * (1 - a.mean()) / n + b.mean() * (1 - b.mean()) / n)
    diff = d.mean()
    return {"차이": float(diff), "±": float(1.96 * se), "독립 가정 시 ±": float(1.96 * naive),
            "유의": bool(abs(diff) > 1.96 * se)}
