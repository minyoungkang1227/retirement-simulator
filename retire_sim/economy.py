"""경제 시나리오 생성기: 상관 GBM(주식·채권) + OU 물가."""
import numpy as np
from .config import EconomyAssumptions


def generate(e: EconomyAssumptions, T: int, n: int, rng) -> dict:
    cov = np.array([[e.stock_sigma**2, e.rho_stock_bond * e.stock_sigma * e.bond_sigma],
                    [e.rho_stock_bond * e.stock_sigma * e.bond_sigma, e.bond_sigma**2]])
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n, T, 2)) @ L.T
    stock = np.exp(e.stock_mu - 0.5 * e.stock_sigma**2 + Z[..., 0]) - 1
    bond = np.exp(e.bond_mu - 0.5 * e.bond_sigma**2 + Z[..., 1]) - 1

    infl = np.empty((n, T))
    pi = np.full(n, e.infl_init)
    for t in range(T):
        pi = pi + e.infl_kappa * (e.infl_theta - pi) + e.infl_sigma * rng.standard_normal(n)
        infl[:, t] = pi
    cpi = np.cumprod(1 + infl, axis=1)          # t년 말 물가지수
    cpi = np.concatenate([np.ones((n, 1)), cpi], axis=1)  # cpi[:, t] = t년 시작 시점
    return {"stock": stock, "bond": bond, "infl": infl, "cpi": cpi}
