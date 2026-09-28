"""경제 시나리오 v2(+v10 강화): 금리·물가·주식·채권, 정확 이산화.

기본 모드(fisher=False, v2와 동일)
- 명목 단기금리 r (Vasicek):  dr = k_r(th_r - r)dt + s_r dW_r
- 물가상승률 pi (OU):           dpi = k_p(th_p - pi)dt + s_p dW_p

피셔 모드(fisher=True, v10) — 금리와 물가를 구조적으로 연결
- 실질금리 q (OU):  dq = k_q(th_q - q)dt + s_q dW_q
- 명목금리 r = q + pi   → 물가가 오르면 명목금리가 자동으로 따라 오름
- 채권 가격: 2요인 가우시안 해석해 P = exp(-E[∫r] + Var[∫r]/2)

공통
- 주식 로그수익 = r + ERP - s_s²/2 + s_s·Z_s  (+ 선택: Merton 점프, 보정항으로 평균 유지)
- 채권: 만기 D년 무이표채 롤링 펀드 (위험프리미엄 0 가정)
- 운용보수 fee: 주식·채권 수익률에서 매년 차감
- 파라미터 불확실성: 경로마다 ERP·장기 물가를 한 번 뽑아 그 경로 전체에 사용
- 반대 난수(antithetic): 경로를 절반씩 +Z / -Z 로 짝지어 표본오차 감소
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
    rho_rp: float = 0.5        # (기본) 금리-물가 / (피셔) 실질금리-물가 충격 상관
    rho_rs: float = -0.2       # 금리-주식 충격 상관
    rho_ps: float = -0.1       # 물가-주식 충격 상관
    stock_shocks: dict | None = None  # 쇼크 테스트: {연차: 주식수익률}, 예 {0: -0.40}
    infl_shocks: dict | None = None   # 쇼크 테스트: {연차: 물가상승률}
    # ── v10 강화 옵션 (기본값은 v2와 동일하게 꺼져 있음) ──
    fisher: bool = False       # 명목금리 = 실질금리 + 물가
    q0: float = 0.005          # 실질금리 초기값 (피셔 모드)
    q_theta: float = 0.01      # 실질금리 장기평균 → 명목 장기평균 = q_theta + pi_theta
    q_kappa: float = 0.20
    q_sigma: float = 0.008
    fee: float = 0.0           # 연 운용보수 (예: 0.005 = 0.5%)
    erp_sd: float = 0.0        # ERP 추정 불확실성 (경로별 추출 표준편차)
    pi_theta_sd: float = 0.0   # 장기 물가 추정 불확실성
    jump_lambda: float = 0.0   # 주가 점프 연평균 횟수 (Merton)
    jump_mu: float = -0.15     # 점프 크기 로그평균
    jump_sigma: float = 0.10   # 점프 크기 로그표준편차
    antithetic: bool = False   # 반대 난수 사용
    # ── v11 수리적 보완 ──
    integrated_rate: bool = False  # 주식 수익률에 연초 금리 대신 1년 금리 적분 ∫r 사용 (채권과 일관)
    term_premium: float = 0.0      # 위험의 시장가격: 가격용(Q) 장기평균 = θ + term_premium → 채권 기간 프리미엄

    @classmethod
    def enhanced(cls, **kw):
        """v10 권장 설정: 피셔 연결 + 운용보수 0.5% + 파라미터 불확실성 + 폭락 점프 + 반대 난수."""
        base = dict(fisher=True, rho_rp=0.0, fee=0.005, erp_sd=0.01, pi_theta_sd=0.005,
                    jump_lambda=0.1, antithetic=True, integrated_rate=True, term_premium=0.005)
        base.update(kw)
        return cls(**base)


def _ou_step(x, theta, kappa, sigma, z):
    """OU 정확 이산화 (dt=1)."""
    e = np.exp(-kappa)
    sd = sigma * np.sqrt((1 - e**2) / (2 * kappa))
    return theta + (x - theta) * e + sd * z


def vasicek_price(tau, r, e: EconomyV2, theta=None):
    k, s = e.r_kappa, e.r_sigma
    th = e.r_theta if theta is None else theta
    B = (1 - np.exp(-k * tau)) / k
    A = (th - s**2 / (2 * k**2)) * (B - tau) - s**2 * B**2 / (4 * k)
    return np.exp(A - B * r)


def _B(k, tau):
    return (1 - np.exp(-k * tau)) / k


def gaussian2_price(tau, x, y, kx, thx, sx, ky, thy, sy, rho):
    """r = x + y (두 상관 OU) 일 때 무이표채 가격 exp(-E[∫r] + Var[∫r]/2)."""
    if tau <= 0:
        return np.ones_like(x)
    Bx, By = _B(kx, tau), _B(ky, tau)
    mean = thx * tau + (x - thx) * Bx + thy * tau + (y - thy) * By
    vx = sx**2 / kx**2 * (tau - 2 * Bx + _B(2 * kx, tau))
    vy = sy**2 / ky**2 * (tau - 2 * By + _B(2 * ky, tau))
    cxy = rho * sx * sy / (kx * ky) * (tau - Bx - By + _B(kx + ky, tau))
    return np.exp(-mean + 0.5 * (vx + vy + 2 * cxy))


def _E(a):
    return (1 - np.exp(-a)) / a


def integrated_cov(k1, s1, k2, s2, c12, c1s, c2s):
    """1년 동안 (X1_{t+1} 잡음, ∫X1 잡음, X2_{t+1} 잡음, ∫X2 잡음, W_s(1))의 공분산 행렬.
    OU 해 X_{t+1} = θ + (X_t-θ)e^{-κ} + σ∫e^{-κ(1-s)}dW,  ∫X = θ + (X_t-θ)E(κ) + (σ/κ)∫(1-e^{-κ(1-s)})dW
    에 이토 등거리를 적용해 계산."""
    K, S = [k1, k2], [s1, s2]
    C = np.array([[1, c12], [c12, 1]]); Cs = [c1s, c2s]
    M = np.zeros((5, 5))
    for i in range(2):
        for j in range(2):
            ki, kj, si, sj, c = K[i], K[j], S[i], S[j], C[i, j]
            M[2*i, 2*j] = c * si * sj * _E(ki + kj)                                   # (끝값_i, 끝값_j)
            M[2*i, 2*j+1] = c * si * sj / kj * (_E(ki) - _E(ki + kj))                  # (끝값_i, 적분_j)
            M[2*i+1, 2*j] = c * si * sj / ki * (_E(kj) - _E(ki + kj))                  # (적분_i, 끝값_j)
            M[2*i+1, 2*j+1] = c * si * sj / (ki * kj) * (1 - _E(ki) - _E(kj) + _E(ki + kj))
        M[2*i, 4] = M[4, 2*i] = Cs[i] * S[i] * _E(K[i])
        M[2*i+1, 4] = M[4, 2*i+1] = Cs[i] * S[i] / K[i] * (1 - _E(K[i]))
    M[4, 4] = 1.0
    return (M + M.T) / 2


def generate(e: EconomyV2, T: int, n: int, rng) -> dict:
    C = np.array([[1, e.rho_rp, e.rho_rs],
                  [e.rho_rp, 1, e.rho_ps],
                  [e.rho_rs, e.rho_ps, 1]])
    L = np.linalg.cholesky(C)
    if e.integrated_rate:
        return _generate_integrated(e, T, n, rng)
    need_u = e.erp_sd > 0 or e.pi_theta_sd > 0          # 불확실성 끄면 v2와 난수 흐름 동일
    if e.antithetic:
        h = (n + 1) // 2
        Zh = rng.standard_normal((h, T, 3)); Z = np.concatenate([Zh, -Zh])[:n]
        u = rng.standard_normal((h, 2)) if need_u else np.zeros((h, 2)); U = np.concatenate([u, -u])[:n]
    else:
        Z = rng.standard_normal((n, T, 3)); U = rng.standard_normal((n, 2)) if need_u else np.zeros((n, 2))
    Z = Z @ L.T

    # 파라미터 불확실성: 경로별 ERP·장기 물가
    erp = e.erp + e.erp_sd * U[:, 0]
    pth = e.pi_theta + e.pi_theta_sd * U[:, 1]

    pi = np.empty((n, T + 1)); pi[:, 0] = e.pi0
    r = np.empty((n, T + 1))
    q = np.empty((n, T + 1))
    if e.fisher:
        q[:, 0] = e.q0; r[:, 0] = e.q0 + e.pi0
    else:
        r[:, 0] = e.r0
    stock = np.empty((n, T)); bond = np.empty((n, T))
    D = e.bond_duration
    k_j = np.exp(e.jump_mu + 0.5 * e.jump_sigma**2) - 1       # 점프 보정항
    for t in range(T):
        pi[:, t + 1] = _ou_step(pi[:, t], pth, e.pi_kappa, e.pi_sigma, Z[:, t, 1])
        if e.fisher:
            q[:, t + 1] = _ou_step(q[:, t], e.q_theta, e.q_kappa, e.q_sigma, Z[:, t, 0])
            r[:, t + 1] = q[:, t + 1] + pi[:, t + 1]
            g = lambda tau, qq, pp: gaussian2_price(tau, qq, pp, e.q_kappa, e.q_theta + e.term_premium, e.q_sigma,
                                                     e.pi_kappa, pth, e.pi_sigma, e.rho_rp)
            bond[:, t] = g(D - 1, q[:, t + 1], pi[:, t + 1]) / g(D, q[:, t], pi[:, t]) - 1
        else:
            r[:, t + 1] = _ou_step(r[:, t], e.r_theta, e.r_kappa, e.r_sigma, Z[:, t, 0])
            th_q = e.r_theta + e.term_premium
            bond[:, t] = vasicek_price(D - 1, r[:, t + 1], e, th_q) / vasicek_price(D, r[:, t], e, th_q) - 1
        logret = r[:, t] + erp - 0.5 * e.s_sigma**2 + e.s_sigma * Z[:, t, 2]
        if e.jump_lambda > 0:
            nj = rng.poisson(e.jump_lambda, n)
            logret += nj * e.jump_mu + np.sqrt(nj) * e.jump_sigma * rng.standard_normal(n) - e.jump_lambda * k_j
        stock[:, t] = np.exp(logret) - 1

    for t, v in (e.stock_shocks or {}).items():
        if t < T: stock[:, t] = v
    if e.fee > 0:
        stock = (1 + stock) * (1 - e.fee) - 1
        bond = (1 + bond) * (1 - e.fee) - 1
    infl = pi[:, 1:].copy()
    for t, v in (e.infl_shocks or {}).items():
        if t < T: infl[:, t] = v
    cpi = np.concatenate([np.ones((n, 1)), np.cumprod(1 + infl, axis=1)], axis=1)
    return {"stock": stock, "bond": bond, "infl": infl, "cpi": cpi, "rate": r,
            "antithetic": e.antithetic, "erp_path": erp}


def _generate_integrated(e: EconomyV2, T: int, n: int, rng) -> dict:
    """v11: (금리 요인의 연말값, 1년 적분)을 결합정규로 정확히 뽑아 주식은 ∫r, 채권은 가격식을 사용."""
    if e.fisher:
        k1, th1, s1, x0 = e.q_kappa, e.q_theta, e.q_sigma, e.q0
    else:
        k1, th1, s1, x0 = e.r_kappa, e.r_theta, e.r_sigma, e.r0
    k2, s2 = e.pi_kappa, e.pi_sigma
    M = integrated_cov(k1, s1, k2, s2, e.rho_rp, e.rho_rs, e.rho_ps)
    L = np.linalg.cholesky(M + 1e-14 * np.eye(5))
    need_u = e.erp_sd > 0 or e.pi_theta_sd > 0
    if e.antithetic:
        h = (n + 1) // 2
        Zh = rng.standard_normal((h, T, 5)); Z = np.concatenate([Zh, -Zh])[:n]
        u = rng.standard_normal((h, 2)) if need_u else np.zeros((h, 2)); U = np.concatenate([u, -u])[:n]
    else:
        Z = rng.standard_normal((n, T, 5)); U = rng.standard_normal((n, 2)) if need_u else np.zeros((n, 2))
    X = Z @ L.T
    erp = e.erp + e.erp_sd * U[:, 0]
    pth = e.pi_theta + e.pi_theta_sd * U[:, 1]
    f1 = np.empty((n, T + 1)); f1[:, 0] = x0
    pi = np.empty((n, T + 1)); pi[:, 0] = e.pi0
    r = np.empty((n, T + 1)); w2 = 1.0 if e.fisher else 0.0
    r[:, 0] = f1[:, 0] + w2 * pi[:, 0]
    stock = np.empty((n, T)); bond = np.empty((n, T)); D = e.bond_duration
    k_j = np.exp(e.jump_mu + 0.5 * e.jump_sigma**2) - 1
    ek1, ek2 = np.exp(-k1), np.exp(-k2)
    for t in range(T):
        I1 = th1 + (f1[:, t] - th1) * _E(k1) + X[:, t, 1]
        I2 = pth + (pi[:, t] - pth) * _E(k2) + X[:, t, 3]
        f1[:, t + 1] = th1 + (f1[:, t] - th1) * ek1 + X[:, t, 0]
        pi[:, t + 1] = pth + (pi[:, t] - pth) * ek2 + X[:, t, 2]
        r[:, t + 1] = f1[:, t + 1] + w2 * pi[:, t + 1]
        if e.fisher:
            g = lambda tau, qq, pp: gaussian2_price(tau, qq, pp, k1, th1 + e.term_premium, s1, k2, pth, s2, e.rho_rp)
            bond[:, t] = g(D - 1, f1[:, t + 1], pi[:, t + 1]) / g(D, f1[:, t], pi[:, t]) - 1
        else:
            th_q = e.r_theta + e.term_premium
            bond[:, t] = vasicek_price(D - 1, r[:, t + 1], e, th_q) / vasicek_price(D, r[:, t], e, th_q) - 1
        Iint = I1 + w2 * I2
        logret = Iint + erp - 0.5 * e.s_sigma**2 + e.s_sigma * X[:, t, 4]
        if e.jump_lambda > 0:
            nj = rng.poisson(e.jump_lambda, n)
            logret += nj * e.jump_mu + np.sqrt(nj) * e.jump_sigma * rng.standard_normal(n) - e.jump_lambda * k_j
        stock[:, t] = np.exp(logret) - 1
    for t, v in (e.stock_shocks or {}).items():
        if t < T: stock[:, t] = v
    if e.fee > 0:
        stock = (1 + stock) * (1 - e.fee) - 1
        bond = (1 + bond) * (1 - e.fee) - 1
    infl = pi[:, 1:].copy()
    for t, v in (e.infl_shocks or {}).items():
        if t < T: infl[:, t] = v
    cpi = np.concatenate([np.ones((n, 1)), np.cumprod(1 + infl, axis=1)], axis=1)
    return {"stock": stock, "bond": bond, "infl": infl, "cpi": cpi, "rate": r,
            "antithetic": e.antithetic, "erp_path": erp}
