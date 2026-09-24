"""세금·계좌·부동산 반영 엔진 (v7).

계좌: 과세계좌(Wt, 취득가 Bt 추적) / ISA(Wi) / 연금계좌(Wp, 원금 Pp)
주택: 유동성 없음(생활비로 못 씀), 물가만큼 가치 상승, 보유세·건보 재산분·상속재산에 반영
순서(매년): 연금수입 → 증여 → 계좌 이전 → 지출 → 연금계좌 인출 → 세금·건보료 → 과세계좌·ISA 인출 → 수익률 → 상속
"""
import numpy as np
from .config import Household, SimConfig
from . import mortality, pension
from .economy_v2 import EconomyV2, generate
from .tax import (TaxConfig, HouseConfig, income_tax_person, private_pension_tax, health_premium, estate_tax,
                  property_tax, comprehensive_property_tax, rent_tax, gift_tax)


def run(hh: Household, cfg: SimConfig, e: EconomyV2 = None, tc: TaxConfig = None,
        house: HouseConfig = None, qx_table=None, buffer_years=5, biz=None):
    """biz: 구성원별 사업 정보 리스트 [dict(income=연 사업소득(만원), until_age=폐업 나이, workplace=직장가입자 여부)] 또는 None"""
    e, tc, house = e or EconomyV2(), tc or TaxConfig(), house or HouseConfig()
    biz = biz or [None] * len(hh.members)
    rng = np.random.default_rng(cfg.seed)
    qx_table = qx_table or mortality.gompertz_qx(cfg.max_age)
    k = len(hh.members); youngest = min(m.age for m in hh.members)
    T, n = cfg.max_age - youngest, cfg.n_paths
    eco = generate(e, T, n, rng); cpi = eco["cpi"]
    alive = np.stack([mortality.simulate_alive(qx_table[m.sex], m.age, T, n, rng) for m in hh.members])
    hh_alive = alive.sum(0) > 0
    own = np.array(tc.ownership if tc.ownership else [1 / k] * k, float)
    hown = np.array(house.owner_share if house.owner_share else [1.0] + [0.0] * (k - 1), float)

    Wt = np.full(n, float(hh.liquid_assets)); Bt = Wt.copy(); Wi = np.zeros(n); Wp = np.zeros(n); Pp = np.zeros(n)
    isa_in = np.zeros(n); cg_due = np.zeros(n)
    gifts_hist = []                                   # (t, 명목 증여액, 증여세)
    gift_val = np.zeros(n)                          # 증여한 돈의 현재 가치(자녀도 같은 수익률로 운용 가정, 명목)
    care_left = np.zeros((k, n), int); care_cost = np.zeros((k, n))
    dep_at = np.full(n, -1)
    tax_y = np.zeros((n, T)); hi_y = np.zeros((n, T)); prop_y = np.zeros((n, T))
    estate_real = np.full(n, np.nan); etax_real = np.full(n, np.nan); transfer_real = np.full(n, np.nan)
    w = hh.stock_weight; W_hist = np.zeros((n, T + 1)); W_hist[:, 0] = Wt

    def sell_from_taxable(amount):
        """과세계좌에서 매도: 실현이익 반환(해외주식분만 과세 대상)."""
        nonlocal Wt, Bt
        amt = np.minimum(np.maximum(amount, 0), Wt)
        ratio = np.where(Wt > 0, Bt / np.maximum(Wt, 1e-9), 1)
        gain = amt * np.maximum(1 - ratio, 0)
        Bt = np.maximum(Bt - amt * np.minimum(ratio, 1), 0); Wt = Wt - amt
        return amt, gain * w * tc.overseas_share

    for t in range(T):
        a = alive[:, :, t]; na = a.sum(0); live = na > 0
        # 명의 재분배(사망자 몫은 생존자에게)
        def redistribute(base):
            sh = base[:, None] * a; orphan = sh.sum(0) == 0
            sh = np.where(orphan[None, :], a, sh); return sh / np.maximum(sh.sum(0), 1e-9)
        fshare, hshare = redistribute(own), redistribute(hown)

        # 1) 연금 수입
        nps = np.zeros((k, n)); priv = np.zeros((k, n))
        for i, m in enumerate(hh.members):
            age = m.age + t
            if age >= m.nps_start_age: nps[i] = a[i] * pension.nps_annual_amount(m, cfg.nps) * cpi[:, t]
            if m.private_pension_start <= age < m.private_pension_start + m.private_pension_years:
                priv[i] = a[i] * m.private_pension_annual
        bz = np.zeros((k, n)); work = np.zeros((k, n), bool)
        for i, m in enumerate(hh.members):
            b = biz[i]
            if b and m.age + t < b["until_age"]:
                bz[i] = a[i] * b["income"] * cpi[:, t]
                work[i] = a[i] & bool(b.get("workplace", False))
        rent = house.rent_annual * cpi[:, t] * live
        official = house.official * cpi[:, t]

        # 2) 증여 전략 (10년마다)
        realized = np.zeros(n); gift_tax_now = np.zeros(n)
        if tc.enabled and tc.gift_per_child_10y > 0 and t % 10 == 0:
            want = tc.gift_per_child_10y * tc.n_children * cpi[:, t] * live
            buf = np.maximum(Wt - buffer_years * hh.annual_spending * cpi[:, t], 0)
            g = np.minimum(want, buf)
            amt, gain = sell_from_taxable(g); realized += gain
            per_child = amt / max(tc.n_children, 1)
            gtax = gift_tax(np.maximum(per_child - 5000, 0)) * tc.n_children
            gifts_hist.append((t, amt, gtax)); gift_val += amt - gtax; gift_tax_now = gtax

        # 3) 계좌 이전 (생활비 버퍼 유지)
        buf = np.maximum(Wt - buffer_years * hh.annual_spending * cpi[:, t], 0)
        if tc.enabled and tc.use_isa:
            room = np.minimum(tc.isa_annual * na, np.maximum(tc.isa_total * na - isa_in, 0))
            amt, gain = sell_from_taxable(np.minimum(room, buf) * live); realized += gain
            Wi += amt; isa_in += amt; buf -= amt
        if tc.enabled and tc.use_pension:
            amt, gain = sell_from_taxable(np.minimum(tc.pen_annual * na, buf) * live); realized += gain
            Wp += amt; Pp += amt

        # 4) 과세계좌 금융소득
        fin_tot = Wt * (w * tc.div_yield + (1 - w) * np.maximum(eco["rate"][:, t], 0))
        fin_i = [fshare[i] * fin_tot for i in range(k)]

        # 5) 지출·간병비
        ratio = np.where(na >= 2, 1.0, hh.survivor_spending_ratio)
        spend = hh.annual_spending * ratio * cpi[:, t] * live
        care = np.zeros(n)
        if cfg.care.enabled:
            c = cfg.care
            for i, m in enumerate(hh.members):
                new = a[i] & (care_left[i] == 0) & (m.age + t >= c.start_age) & (rng.random(n) < c.lam)
                care_cost[i] = np.where(new, c.cost_median * np.exp(c.cost_log_sigma * rng.standard_normal(n)), care_cost[i])
                care_left[i] = np.where(new, c.duration_years, care_left[i])
                act = a[i] & (care_left[i] > 0); care += act * care_cost[i] * cpi[:, t]
                care_left[i] = np.where(act, care_left[i] - 1, 0)

        # 6) 연금계좌 인출
        need = spend + care - nps.sum(0) - priv.sum(0) - rent - bz.sum(0)
        pw = np.zeros(n); pw_taxable = np.zeros(n)
        if tc.enabled and tc.use_pension and t >= tc.pen_wait_years:
            pw = np.clip(np.minimum(need, tc.pen_withdraw_cap * na), 0, Wp)
            frac = np.where(Wp > 0, np.clip(1 - Pp / np.maximum(Wp, 1e-9), 0, 1), 0)
            Pp = np.maximum(Pp - pw * (1 - frac), 0); Wp -= pw; pw_taxable = pw * frac

        # 7) 세금·건보료·보유세
        taxes = np.zeros(n); hi = np.zeros(n); prop = np.zeros(n)
        if tc.enabled:
            prop_base = [hshare[i] * official * (0.45 if house.n_houses == 1 else house.fmv_ratio_prop) for i in range(k)]
            rent_i = [hshare[i] * rent for i in range(k)]
            for i, m in enumerate(hh.members):
                age = m.age + t
                sep_rent, comb_rent = rent_tax(rent_i[i], nps[i] * tc.nps_taxable_ratio)
                taxes += a[i] * (income_tax_person(nps[i], fin_i[i], age, tc, extra=comb_rent + bz[i]) + sep_rent)
                taxes += a[i] * private_pension_tax(priv[i] + pw_taxable * fshare[i], age)
            if tc.use_isa:
                taxes += np.maximum(Wi * (w * tc.div_yield + (1 - w) * np.maximum(eco["rate"][:, t], 0))
                                    - tc.isa_free_per_year * na, 0) * .099
            # 해외주식 양도세(작년 실현분, 1인 250만원 공제) + 올해 증여세
            taxes += np.maximum(cg_due - 250 * na, 0) * .22 * 1.1 * live 
            if house.official > 0:
                oldest = max(m.age for m in hh.members) + t
                # 재산세: 물건별(가구 합산 근사) / 종부세: 인별 과세(명의 비율대로 1인 공제 9억)
                prop = property_tax(official, house.n_houses, house.fmv_ratio_prop)
                co_owned = (hshare > 0).sum(0) >= 2
                per_person = sum(a[i] * comprehensive_property_tax(hshare[i] * official, max(house.n_houses, 2),
                                 hh.members[i].age + t, house.years_held + t, house.fmv_ratio_cjs) for i in range(k))
                if house.n_houses == 1:   # 1주택: 단독명의 12억 공제+고령자·장기보유 공제 / 공동명의는 1인 9억 공제와 특례 중 유리한 쪽
                    single = comprehensive_property_tax(official, 1, oldest, house.years_held + t, house.fmv_ratio_cjs)
                    cjs = np.where(co_owned, np.minimum(per_person, single), single)
                else:
                    cjs = per_person
                prop = (prop + cjs) * live
            pm = (house.hi_property_monthly or 0) * cpi[:, t]
            rate = tc.hi_rate * (1 + tc.ltc_ratio)
            any_work = work.any(0)
            # 직장가입자(사업장 대표): 사업소득 전액(대표자 전액 부담) + 보수 외 소득 2,000만원 초과분
            hi_w = np.zeros(n)
            for i in range(k):
                other = .5 * nps[i] + np.where(fin_i[i] > 1000, fin_i[i], 0) + .5 * rent_i[i]
                hi_w += work[i] * rate * (bz[i] + np.maximum(other - 2000, 0))
            # 지역가입자(직장 아닌 생존자): 소득분 + 재산분(직장가입자 몫 주택 지분 제외)
            nonw = [a[i] & ~work[i] for i in range(k)]
            reg_inc = sum(nonw[i] * (.5 * nps[i] + np.where(fin_i[i] > 1000, fin_i[i], 0) + .5 * rent_i[i] + bz[i]) for i in range(k))
            reg_prop_share = sum(nonw[i] * hshare[i] for i in range(k))
            all_reg = health_premium([nps[i] for i in range(k)], fin_i, tc, rent_i, prop_base, pm)
            mixed_reg = reg_inc * rate + pm * 12 * reg_prop_share
            hi = (hi_w + np.where(any_work, mixed_reg, all_reg + sum(bz[i] for i in range(k)) * rate)) * live
        tax_y[:, t] = taxes / cpi[:, t]; hi_y[:, t] = hi / cpi[:, t]; prop_y[:, t] = prop / cpi[:, t]

        # 8) 인출: 과세계좌 → ISA → (부족 시) 연금계좌 연금외수령
        rest = need - pw + taxes + hi + prop
        amt, gain = sell_from_taxable(np.maximum(rest, 0)); realized += gain
        from_i = np.minimum(np.maximum(rest, 0) - amt, Wi); Wi -= from_i
        surplus = np.maximum(-rest, 0); Wt += surplus; Bt += surplus
        short = np.maximum(rest, 0) - amt - from_i
        if tc.enabled and tc.use_pension:
            frac2 = np.where(Wp > 0, np.clip(1 - Pp / np.maximum(Wp, 1e-9), 0, 1), 0)
            gross = np.minimum(short / np.maximum(1 - .165 * frac2, 1e-9), Wp)
            Pp = np.maximum(Pp - gross * (1 - frac2), 0); Wp -= gross
            tax_y[:, t] += gross * frac2 * .165 / cpi[:, t]
            short = np.maximum(short - gross * (1 - .165 * frac2), 0)
        newly = (short > 1e-6) & (dep_at < 0) & live; dep_at[newly] = t
        cg_due = realized

        # 9) 수익률
        r = w * eco["stock"][:, t] + (1 - w) * eco["bond"][:, t]
        Wt *= (1 + r); Wi *= (1 + r); Wp *= (1 + r); gift_val *= (1 + r)
        W_hist[:, t + 1] = (Wt + Wi + Wp) * (dep_at < 0)

        # 10) 마지막 생존자 사망 → 상속세(10년 내 증여 합산, 기납부 증여세 공제)
        died_all = live & (alive[:, :, t + 1].sum(0) == 0)
        if died_all.any():
            fin_est = (Wt + Wi + Wp) * (dep_at < 0)
            est = fin_est + house.market * cpi[:, t + 1]
            recent = sum(g for (tg, g, _) in gifts_hist if t + 1 - tg < 10) if gifts_hist else 0
            recent_gt = sum(gt for (tg, _, gt) in gifts_hist if t + 1 - tg < 10) if gifts_hist else 0
            fin_ded = np.where(fin_est <= 2000, fin_est, np.where(fin_est <= 10000, 2000, np.minimum(fin_est * .2, 20000)))
            lump = max(50000, 20000 + 5000 * tc.n_children)
            et = np.maximum(progressive_estate(est + recent - lump - fin_ded) - recent_gt, 0)
            estate_real = np.where(died_all, (est - et) / cpi[:, t + 1], estate_real)
            etax_real = np.where(died_all, et / cpi[:, t + 1], etax_real)
            transfer_real = np.where(died_all, (est - et + gift_val) / cpi[:, t + 1], transfer_real)

    return {"depleted_at": dep_at, "youngest_age": youngest, "T": T, "tax_y": tax_y, "hi_y": hi_y, "prop_y": prop_y,
            "estate_real": estate_real, "estate_tax_real": etax_real, "transfer_real": transfer_real,
            "W_nominal": W_hist, "W_real": W_hist / cpi, "hh_alive": hh_alive,
            "last_alive_t": hh_alive.sum(1) - 1, "cpi": cpi}


def progressive_estate(base):
    from .tax import progressive, ESTATE
    return progressive(base, ESTATE)


def summarize(res):
    d = res["depleted_at"]; p = float((d >= 0).mean()); se = float(np.sqrt(p * (1 - p) / len(d)))
    y = min(20, res["T"])
    med = lambda x: float(np.median(x[~np.isnan(x)])) if np.any(~np.isnan(x)) else 0.0
    return {"고갈확률": p, "±": 1.96 * se,
            "20년 세금": float(res["tax_y"][:, :y].sum(1).mean()),
            "20년 건보료": float(res["hi_y"][:, :y].sum(1).mean()),
            "20년 보유세": float(res["prop_y"][:, :y].sum(1).mean()),
            "상속세(중앙값)": med(res["estate_tax_real"]),
            "가족 이전 총액(중앙값)": med(res["transfer_real"])}
