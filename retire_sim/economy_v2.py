"""경제 시나리오 v2: Vasicek 단기금리 + OU 물가 + 초과수익 주식, 정확 이산화.

- 단기금리 r (Vasicek):   dr = k_r(th_r - r)dt + s_r dW_r
- 물가상승률 pi (OU):      dpi = k_p(th_p - pi)dt + s_p dW_p
- 주식 로그수익 = r_t + ERP - s_s^2/2 + s_s * Z_s
- 채권: 만기 D년 무이표채를 1년 보유 후 (D-1)년물로 매도하는 롤링 펀드.
  가격은 Vasicek 해석해 P(tau, r) = exp(A(tau) - B(tau) r)
  (단순화: 위험프리미엄 0, 실측=위험중립 가정)
- 세 충격(W_r, W_p, Z_s)은 상관행렬로 결합.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class EconomyV2:
    r0: float = 0.025
    r_theta: float = 0.03
    r_kappa: float = 0.15
    r_sigma: float = 0.010
    pi0: float = 0.02
    pi_theta: float = 0.02
    pi_kappa: float = 0.40
    pi_sigma: float = 0.010
    erp: float = 0.04          # 주식 위험프리미엄
    s_sigma: float = 0.18
    bond_duration: float = 5.0
    rho_rp: float = 0.5        # 금리-물가 충격 상관
    rho_rs: float = -0.2       # 금리-주식 충격 상관
    rho_ps: float = -0.1       # 물가-주식 충격 상관
    stock_shocks: dict | None = None  # 쇼크 테스트: {연차: 주식수익률}, 예 {0: -0.40}
    infl_shocks: dict | None = None   # 쇼크 테스트: {연차: 물가상승률}


def _ou_step(x, theta, kappa, sigma, z):
    """OU 정확 이산화 (dt=1)."""
    e = np.exp(-kappa)
    sd = sigma * np.sqrt((1 - e**2) / (2 * kappa))
    return theta + (x - theta) * e + sd * z


def vasicek_price(tau, r, e: EconomyV2):
    k, th, s = e.r_kappa, e.r_theta, e.r_sigma
    B = (1 - np.exp(-k * tau)) / k
    A = (th - s**2 / (2 * k**2)) * (B - tau) - s**2 * B**2 / (4 * k)
    return np.exp(A - B * r)


def generate(e: EconomyV2, T: int, n: int, rng) -> dict:
    C = np.array([[1, e.rho_rp, e.rho_rs],
                  [e.rho_rp, 1, e.rho_ps],
                  [e.rho_rs, e.rho_ps, 1]])
    L = np.linalg.cholesky(C)
    Z = rng.standard_normal((n, T, 3)) @ L.T

    r = np.empty((n, T + 1)); r[:, 0] = e.r0
    pi = np.empty((n, T + 1)); pi[:, 0] = e.pi0
    stock = np.empty((n, T)); bond = np.empty((n, T))
    D = e.bond_duration
    for t in range(T):
        r[:, t + 1] = _ou_step(r[:, t], e.r_theta, e.r_kappa, e.r_sigma, Z[:, t, 0])
        pi[:, t + 1] = _ou_step(pi[:, t], e.pi_theta, e.pi_kappa, e.pi_sigma, Z[:, t, 1])
        stock[:, t] = np.exp(r[:, t] + e.erp - 0.5 * e.s_sigma**2 + e.s_sigma * Z[:, t, 2]) - 1
        bond[:, t] = vasicek_price(D - 1, r[:, t + 1], e) / vasicek_price(D, r[:, t], e) - 1

    for t, v in (e.stock_shocks or {}).items():
        if t < T: stock[:, t] = v
    infl = pi[:, 1:].copy()
    for t, v in (e.infl_shocks or {}).items():
        if t < T: infl[:, t] = v
    cpi = np.concatenate([np.ones((n, 1)), np.cumprod(1 + infl, axis=1)], axis=1)
    return {"stock": stock, "bond": bond, "infl": infl, "cpi": cpi, "rate": r}
