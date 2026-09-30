"""세금·계좌·부동산 반영 엔진 (v7).

계좌: 과세계좌(Wt, 취득가 Bt 추적) / ISA(Wi) / 연금계좌(Wp, 원금 Pp)
주택: 유동성 없음(생활비로 못 씀), 물가만큼 가치 상승, 보유세·건보 재산분·상속재산에 반영
순서(매년): 연금수입 → 증여 → 계좌 이전 → 지출 → 연금계좌 인출 → 세금·건보료 → 과세계좌·ISA 인출 → 수익률 → 상속
"""
import numpy as np
from .config import Household, SimConfig, AddOns
from . import mortality, pension
from .economy_v2 import EconomyV2, generate
from .tax import (TaxConfig, HouseConfig, income_tax_person, private_pension_tax, health_premium, estate_tax,
                  property_tax, comprehensive_property_tax, rent_tax, gift_tax)


def run(hh: Household, cfg: SimConfig, e: EconomyV2 = None, tc: TaxConfig = None,
        house: HouseConfig = None, qx_table=None, buffer_years=5, biz=None, addons=None):
    """biz: 구성원별 사업 정보 리스트 [dict(income=연 사업소득(만원), until_age=폐업 나이, workplace=직장가입자 여부)] 또는 None"""
    e, tc, house = e or EconomyV2(), tc or TaxConfig(), house or HouseConfig()
    biz = biz or [None] * len(hh.members)
    rng = np.random.default_rng(cfg.seed)
    qx_table = qx_table or mortality.default_qx(cfg.max_age)
    k = len(hh.members); youngest = min(m.age for m in hh.members)
    T, n = cfg.max_age - youngest, cfg.n_paths
    eco = generate(e, T, n, rng); cpi = eco["cpi"]
    from .config import CareMarkov
    markov = isinstance(cfg.care, CareMarkov) and cfg.care.enabled
    if markov:
        lives = [mortality.simulate_life(qx_table[m.sex], m.age, T, n, rng, cfg.care, return_severe=True) for m in hh.members]
        alive = np.stack([l[0] for l in lives]); incare = np.stack([l[1] for l in lives]); severe = np.stack([l[2] for l in lives])
        care_level = cfg.care.cost_median * np.exp(cfg.care.cost_log_sigma * rng.standard_normal((k, n)))
        sev_level = getattr(cfg.care, 'cost_severe_median', cfg.care.cost_median) * np.exp(getattr(cfg.care, 'cost_severe_sigma', cfg.care.cost_log_sigma) * rng.standard_normal((k, n)))
    else:
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
    ess_base = hh.essential if hh.essential is not None else hh.annual_spending
    spend_base = ess_base + hh.lifestyle
    life_ratio = np.full((n, T), np.nan)

    # ── v15 상품 추가 (AddOns) ──
    ad = addons or AddOns()
    rng2 = np.random.default_rng((cfg.seed or 0) + 7)          # 추가 상품 전용 난수(기존 시나리오 불변 → 짝지은 비교)
    S = np.zeros(n); D = np.zeros(n)
    if ad.stock_amount > 0:
        mv = np.minimum(ad.stock_amount, Wt); Wt -= mv; Bt -= mv; S += mv
    if ad.deposit_amount > 0:
        mv = np.minimum(ad.deposit_amount, Wt); Wt -= mv; Bt -= mv; D += mv
    ann_pay = np.zeros(n); ann_t = max(0, ad.annuity_start_age - hh.members[0].age)
    ann_members = list(range(k)) if (ad.annuity_joint and k > 1) else [0]
    def annuity_factor(t0):
        v = 1 / (1 + ad.annuity_rate); surv = []
        for i in ann_members:
            m = hh.members[i]; q = np.minimum(qx_table[m.sex] * ad.annuity_mort_mult, 1.0)
            x = min(m.age + t0, len(q) - 1)
            surv.append(np.concatenate([[1.0], np.cumprod(1 - q[x:-1])]))
        L = min(len(x) for x in surv); p = surv[0][:L]
        if len(surv) > 1: p = surv[0][:L] + surv[1][:L] - surv[0][:L] * surv[1][:L]
        return float(np.sum(v ** np.arange(L) * p))
    ci_prem = 0.0; ci_on = ad.care_benefit > 0 and markov and ad.care_member < k
    if ci_on:                                                  # 수지상등: 모델 자체의 간병·사망 경로로 보험료 산출
        i0 = ad.care_member; v = (1 + ad.care_rate) ** -np.arange(T + 1)
        ages_i = hh.members[i0].age + np.arange(T + 1)
        trig = severe if (ad.care_trigger == "severe" and getattr(cfg.care, "two_level", False)) else incare
        pv_ben = (trig[i0] * v).sum(1).mean() * ad.care_benefit
        pay = alive[i0] & ~incare[i0] & (ages_i < ad.care_pay_until)[None, :]
        pv_prem = (pay * v).sum(1).mean()
        ci_prem = (1 + ad.care_loading) * pv_ben / max(pv_prem, 1e-9)
    stock_ret = np.full((n, T), np.nan); port_ret = np.full((n, T), np.nan)

    # ── v16 기본생활비 보호선: 생존확률 가중 현가 (계리적) ──
    # 상태 0 = 부부 모두 생존(1인 가구는 본인 생존), 1 = 본인만, 2 = 배우자만.
    # F[t, 상태] = Σ_s v^s Σ_상태' P(상태'|상태, t→t+s) · max(기본생활비×비율 − 보장소득_실질, 0)   (오늘 가치)
    floor_tab = None
    if hh.floor_method == "actuarial":
        v_r = 1 / (1 + hh.floor_real_rate); pibar = e.pi_theta
        qf = [np.minimum(qx_table[m.sex] * hh.floor_mort_mult, 1.0) for m in hh.members]
        def inc_real(i, yr):                               # 구성원 i의 yr년차 보장소득(실질): 국민연금(물가연동) + 사적연금(명목 정액)
            m = hh.members[i]; age = m.age + yr; x = 0.0
            if age >= m.nps_start_age: x += pension.nps_annual_amount(m, cfg.nps)
            if m.private_pension_start <= age < m.private_pension_start + m.private_pension_years:
                x += m.private_pension_annual / (1 + pibar) ** yr
            return x
        rent_real = house.rent_annual
        H = T + 1; floor_tab = np.zeros((H, 3))
        for t0 in range(H):
            horizon = H - t0
            surv = []
            for i, m in enumerate(hh.members):
                x0 = min(m.age + t0, len(qf[i]) - 1)
                q = qf[i][x0:x0 + horizon]; q = np.concatenate([q, np.ones(max(0, horizon - len(q)))])
                surv.append(np.concatenate([[1.0], np.cumprod(1 - q)[:-1]]))   # s년 뒤 생존확률
            vs = v_r ** np.arange(horizon)
            yrs = np.arange(t0, t0 + horizon)
            gap = lambda members, ratio: np.array([max(ess_base * ratio - sum(inc_real(i, y) for i in members) - rent_real, 0.0) for y in yrs])
            g0 = gap([0], 1.0 if k == 1 else hh.survivor_spending_ratio)
            if k == 1:
                floor_tab[t0, :] = np.sum(vs * surv[0] * g0); continue
            g1 = gap([1], hh.survivor_spending_ratio); gb = gap([0, 1], 1.0)
            p0, p1 = surv[0], surv[1]
            floor_tab[t0, 0] = np.sum(vs * (p0 * p1 * gb + p0 * (1 - p1) * g0 + (1 - p0) * p1 * g1))
            floor_tab[t0, 1] = np.sum(vs * p0 * g0)
            floor_tab[t0, 2] = np.sum(vs * p1 * g1)
    short_real = np.zeros(n)                                   # 생존 중 부족했던 기본생활비 누계(실질)

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
            buf = np.maximum(Wt - buffer_years * spend_base * cpi[:, t], 0)
            g = np.minimum(want, buf)
            amt, gain = sell_from_taxable(g); realized += gain
            per_child = amt / max(tc.n_children, 1)
            gtax = gift_tax(np.maximum(per_child - 5000, 0)) * tc.n_children
            gifts_hist.append((t, amt, gtax)); gift_val += amt - gtax; gift_tax_now = gtax

        # 3) 계좌 이전 (생활비 버퍼 유지)
        buf = np.maximum(Wt - buffer_years * spend_base * cpi[:, t], 0)
        if tc.enabled and tc.use_isa:
            room = np.minimum(tc.isa_annual * na, np.maximum(tc.isa_total * na - isa_in, 0))
            amt, gain = sell_from_taxable(np.minimum(room, buf) * live); realized += gain
            Wi += amt; isa_in += amt; buf -= amt
        if tc.enabled and tc.use_pension:
            amt, gain = sell_from_taxable(np.minimum(tc.pen_annual * na, buf) * live); realized += gain
            Wp += amt; Pp += amt

        # 4) 과세계좌 금융소득
        fin_tot = Wt * (w * tc.div_yield + (1 - w) * np.maximum(eco["rate"][:, t], 0)) + D * np.maximum(eco["rate"][:, t], 0)
        fin_i = [fshare[i] * fin_tot for i in range(k)]

        # 5) 지출·간병비
        ratio = np.where(na >= 2, 1.0, hh.survivor_spending_ratio)
        retired = True if hh.retire_age is None else (hh.members[0].age + t >= hh.retire_age)
        ess = ess_base * ratio * cpi[:, t] * live
        life_want = hh.lifestyle * ratio * cpi[:, t] * live
        care = np.zeros(n)
        if markov:
            care = ((incare[:, :, t] & ~severe[:, :, t]) * care_level + severe[:, :, t] * sev_level).sum(0) * cpi[:, t]
        elif cfg.care.enabled:
            c = cfg.care
            for i, m in enumerate(hh.members):
                new = a[i] & (care_left[i] == 0) & (m.age + t >= c.start_age) & (rng.random(n) < c.lam)
                care_cost[i] = np.where(new, c.cost_median * np.exp(c.cost_log_sigma * rng.standard_normal(n)), care_cost[i])
                care_left[i] = np.where(new, c.duration_years, care_left[i])
                act = a[i] & (care_left[i] > 0); care += act * care_cost[i] * cpi[:, t]
                care_left[i] = np.where(act, care_left[i] - 1, 0)

        # 6) 연금계좌 인출
        if ad.annuity_premium > 0 and t == ann_t:                  # 종신연금 일시납 가입
            prem = np.minimum(ad.annuity_premium * cpi[:, t] * live, Wt); Wt -= prem; Bt = np.maximum(Bt - prem, 0)
            ann_pay = prem / (annuity_factor(t) * (1 + ad.annuity_loading))
        ann_alive = np.zeros(n, bool)
        for i in ann_members: ann_alive |= a[i]
        ann_inc = ann_pay * ann_alive if t >= ann_t else np.zeros(n)
        ci_ben = np.zeros(n); ci_cost = np.zeros(n)
        if ci_on:
            i0 = ad.care_member
            ci_ben = ad.care_benefit * trig[i0, :, t]
            ci_cost = ci_prem * (a[i0] & ~incare[i0, :, t] & (hh.members[i0].age + t < ad.care_pay_until))
        income_all = nps.sum(0) + priv.sum(0) + rent + bz.sum(0) + ann_inc + ci_ben
        if retired:
            # 목표 우선순위: 기본생활(Essential) 보호선을 먼저 지키고, 남는 만큼만 여행·취미(Lifestyle) 지출
            if floor_tab is not None:
                st_idx = np.where(a[0] & (a[1] if k > 1 else True), 0, np.where(a[0], 1, 2)) if k > 1 else np.zeros(n, int)
                floor = floor_tab[t, st_idx] * cpi[:, t] * live
                if ad.annuity_premium > 0 and t >= ann_t:           # 종신연금 소득은 보장소득으로 보호선에서 차감
                    floor = np.maximum(floor - ann_inc * annuity_factor(t) / (1 + e.pi_theta), 0)
            else:
                yrs_left = max(5, hh.planning_age - (youngest + t))
                ann = (1 - 1.02 ** -yrs_left) / 0.02            # (v12) 실질 2%로 할인한 확정 연금현가계수
                floor = np.maximum(ess - income_all, 0) * ann
            life_paid = np.clip(Wt + Wi + Wp + S + D - floor, 0, life_want)
            spend = ess + life_paid
            if hh.lifestyle > 0:
                life_ratio[:, t] = np.where(live, life_paid / np.maximum(life_want, 1e-9), np.nan)
            saving = 0.0
        else:
            spend = np.zeros(n); saving = hh.annual_saving * cpi[:, t] * live
            if ad.pension_contrib_annual > 0:                     # 저축 일부를 연금계좌로, 세액공제 환급은 과세계좌로
                pc = ad.pension_contrib_annual * cpi[:, t] * live
                credit = np.minimum(pc, 900) * ad.pension_credit_rate
                Wp += pc; saving = saving - pc + credit
        need = spend + care + ci_cost - income_all - saving
        pw = np.zeros(n); pw_taxable = np.zeros(n)
        pen_active = tc.enabled and (tc.use_pension or ad.pension_contrib_annual > 0)
        pen_open = (tc.use_pension and t >= tc.pen_wait_years) or (ad.pension_contrib_annual > 0 and retired
                                                                   and hh.members[0].age + t >= 55)
        if pen_active and pen_open:
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
        from_d = np.minimum(short, D); D -= from_d; short -= from_d
        from_s = np.minimum(short, S); S -= from_s; short -= from_s
        if pen_active:
            frac2 = np.where(Wp > 0, np.clip(1 - Pp / np.maximum(Wp, 1e-9), 0, 1), 0)
            gross = np.minimum(short / np.maximum(1 - .165 * frac2, 1e-9), Wp)
            Pp = np.maximum(Pp - gross * (1 - frac2), 0); Wp -= gross
            tax_y[:, t] += gross * frac2 * .165 / cpi[:, t]
            short = np.maximum(short - gross * (1 - .165 * frac2), 0)
        newly = (short > 1e-6) & (dep_at < 0) & live; dep_at[newly] = t
        short_real += np.where(live, short, 0) / cpi[:, t]
        cg_due = realized

        # 9) 수익률
        r = w * eco["stock"][:, t] + (1 - w) * eco["bond"][:, t]
        Wt *= (1 + r); Wi *= (1 + r); Wp *= (1 + r); gift_val *= (1 + r)
        rt = eco["rate"][:, t]
        D *= (1 + np.maximum(rt, 0))
        if ad.stock_amount > 0:                                    # 단일지수(CAPM) + 금리 민감도
            lm = np.log1p(eco["stock"][:, t]) - np.log1p(-e.fee)  # 시장 로그수익(펀드 보수 제외)
            b = ad.stock_beta; s_e = ad.stock_idio_sigma
            li = (rt + b * (lm - rt) + 0.5 * b * (1 - b) * e.s_sigma ** 2 - 0.5 * s_e ** 2
                  + s_e * rng2.standard_normal(n) + ad.stock_rate_beta * (eco["rate"][:, t + 1] - rt))
            S *= np.exp(li); stock_ret[:, t] = np.expm1(li); port_ret[:, t] = r
        W_hist[:, t + 1] = (Wt + Wi + Wp + S + D) * (dep_at < 0)

        # 10) 마지막 생존자 사망 → 상속세(10년 내 증여 합산, 기납부 증여세 공제)
        died_all = live & (alive[:, :, t + 1].sum(0) == 0)
        if died_all.any():
            fin_est = (Wt + Wi + Wp + S + D) * (dep_at < 0)
            est = fin_est + house.market * cpi[:, t + 1]
            recent = sum(g for (tg, g, _) in gifts_hist if t + 1 - tg < 10) if gifts_hist else 0
            recent_gt = sum(gt for (tg, _, gt) in gifts_hist if t + 1 - tg < 10) if gifts_hist else 0
            fin_ded = np.where(fin_est <= 2000, fin_est, np.where(fin_est <= 10000, 2000, np.minimum(fin_est * .2, 20000)))
            lump = max(50000, 20000 + 5000 * tc.n_children)
            et = np.maximum(progressive_estate(est + recent - lump - fin_ded) - recent_gt, 0)
            estate_real = np.where(died_all, (est - et) / cpi[:, t + 1], estate_real)
            etax_real = np.where(died_all, et / cpi[:, t + 1], etax_real)
            transfer_real = np.where(died_all, (est - et + gift_val) / cpi[:, t + 1], transfer_real)

    return {"depleted_at": dep_at, "youngest_age": youngest, "T": T, "antithetic": eco.get("antithetic", False), "tax_y": tax_y, "hi_y": hi_y, "prop_y": prop_y,
            "estate_real": estate_real, "life_ratio": life_ratio, "stock_ret": stock_ret, "port_ret": port_ret, "short_real": short_real,
            "floor_tab": floor_tab, "annuity_pay": float(np.median(ann_pay[ann_pay > 0])) if (ann_pay > 0).any() else 0.0, "care_premium": ci_prem, "legacy_target": hh.legacy_target, "estate_tax_real": etax_real, "transfer_real": transfer_real,
            "W_nominal": W_hist, "W_real": W_hist / cpi, "hh_alive": hh_alive,
            "last_alive_t": hh_alive.sum(1) - 1, "cpi": cpi}


def progressive_estate(base):
    from .tax import progressive, ESTATE
    return progressive(base, ESTATE)


def summarize(res):
    from .metrics import prob_se
    d = res["depleted_at"]; p = float((d >= 0).mean()); se = prob_se(d >= 0, res.get("antithetic", False))
    y = min(20, res["T"])
    med = lambda x: float(np.median(x[~np.isnan(x)])) if np.any(~np.isnan(x)) else 0.0
    return {"고갈확률": p, "±": 1.96 * se,
            "20년 세금": float(res["tax_y"][:, :y].sum(1).mean()),
            "20년 건보료": float(res["hi_y"][:, :y].sum(1).mean()),
            "20년 보유세": float(res["prop_y"][:, :y].sum(1).mean()),
            "상속세(중앙값)": med(res["estate_tax_real"]),
            "가족 이전 총액(중앙값)": med(res["transfer_real"]),
            **goal_summary(res)}


def goal_summary(res):
    """목표별 달성 지표: 기본생활 유지 확률, 여행·취미 충족률, 유산 달성 확률."""
    d = res["depleted_at"]
    out = {"기본생활 유지 확률": float((d < 0).mean())}
    sr = res.get("short_real")
    if sr is not None:
        out["평균 부족액(실질)"] = float(sr.mean())
        out["부족 시 평균 부족액(실질)"] = float(sr[sr > 0].mean()) if (sr > 0).any() else 0.0
    lr = res.get("life_ratio")
    if lr is not None and np.any(~np.isnan(lr)):
        cnt = (~np.isnan(lr)).sum(1)
        per = np.where(cnt > 0, np.nansum(lr, 1) / np.maximum(cnt, 1), np.nan)
        per = per[~np.isnan(per)]
        out["여행·취미 평균 충족률"] = float(per.mean())
        out["여행·취미 90% 이상 유지 확률"] = float((per >= 0.9).mean())
    tgt = res.get("legacy_target", 0) or 0
    if tgt > 0:
        est = res["estate_real"]
        out["유산 목표 달성 확률"] = float(np.mean(np.nan_to_num(est, nan=0.0) >= tgt))
    return out
