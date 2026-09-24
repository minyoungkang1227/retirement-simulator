"""연간 현금흐름 엔진. 모든 경로를 벡터화해 한 번에 계산."""
import numpy as np
from .config import Household, SimConfig
from . import economy, mortality, pension


def run(hh: Household, cfg: SimConfig, qx_table: dict | None = None, economy_v2=None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    qx_table = qx_table or mortality.gompertz_qx(cfg.max_age)
    youngest = min(m.age for m in hh.members)
    T = cfg.max_age - youngest
    n = cfg.n_paths

    if economy_v2 is not None:
        from . import economy_v2 as ev2
        eco = ev2.generate(economy_v2, T, n, rng)
    else:
        eco = economy.generate(cfg.economy, T, n, rng)
    cpi = eco["cpi"]

    alive = np.stack([mortality.simulate_alive(qx_table[m.sex], m.age, T, n, rng)
                      for m in hh.members])            # (k, n, T+1)
    n_alive = alive.sum(axis=0)                        # (n, T+1)
    hh_alive = n_alive > 0

    # 간병비 상태
    care_left = np.zeros((len(hh.members), n), dtype=int)
    care_cost = np.zeros((len(hh.members), n))

    W = np.zeros((n, T + 1))
    W[:, 0] = hh.liquid_assets
    depleted_at = np.full(n, -1)
    comp = {k: np.zeros((n, T)) for k in ["income", "spending", "care"]}

    for t in range(T):
        # 1) 수입
        income = np.zeros(n)
        for i, m in enumerate(hh.members):
            age = m.age + t
            a = alive[i, :, t]
            if age >= m.nps_start_age:
                income += a * pension.nps_annual_amount(m, cfg.nps) * cpi[:, t]
            if m.private_pension_start <= age < m.private_pension_start + m.private_pension_years:
                income += a * m.private_pension_annual      # 명목 고정
        # 2) 지출
        ratio = np.where(n_alive[:, t] >= 2, 1.0, hh.survivor_spending_ratio)
        spend = hh.annual_spending * ratio * cpi[:, t] * hh_alive[:, t]
        # 3) 간병비 점프
        care = np.zeros(n)
        if cfg.care.enabled:
            c = cfg.care
            for i, m in enumerate(hh.members):
                a = alive[i, :, t]
                new = a & (care_left[i] == 0) & (m.age + t >= c.start_age) & (rng.random(n) < c.lam)
                care_cost[i] = np.where(new, c.cost_median * np.exp(c.cost_log_sigma * rng.standard_normal(n)), care_cost[i])
                care_left[i] = np.where(new, c.duration_years, care_left[i])
                active = a & (care_left[i] > 0)
                care += active * care_cost[i] * cpi[:, t]
                care_left[i] = np.where(active, care_left[i] - 1, 0)
        # 4) 자산 갱신: 연초 순인출 후 수익률 적용
        net = income - spend - care
        w = np.maximum(W[:, t] + net, 0)
        newly = (W[:, t] + net <= 0) & (depleted_at < 0) & hh_alive[:, t]
        depleted_at[newly] = t
        r = hh.stock_weight * eco["stock"][:, t] + (1 - hh.stock_weight) * eco["bond"][:, t]
        W[:, t + 1] = w * (1 + r)
        comp["income"][:, t], comp["spending"][:, t], comp["care"][:, t] = income, spend, care

    last_alive_t = hh_alive.sum(axis=1) - 1
    return {"W_nominal": W, "W_real": W / cpi, "cpi": cpi, "alive": alive,
            "hh_alive": hh_alive, "depleted_at": depleted_at,
            "last_alive_t": last_alive_t, "youngest_age": youngest, "T": T, "rate": eco.get("rate"), "infl": eco["infl"], **comp}
