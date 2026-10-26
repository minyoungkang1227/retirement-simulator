"""세금·계좌·부동산 반영 엔진 (v7).

계좌: 과세계좌(Wt, 취득가 Bt 추적) / ISA(Wi) / 연금계좌(Wp, 원금 Pp)
주택: 유동성 없음(생활비로 못 씀), 물가만큼 가치 상승, 보유세·건보 재산분·상속재산에 반영
순서(매년): 연금수입 → 증여 → 계좌 이전 → 지출 → 연금계좌 인출 → 세금·건보료 → 과세계좌·ISA 인출 → 수익률 → 상속
"""
import numpy as np
from .config import Household, SimConfig, AddOns, CareMarkov
from . import mortality, pension
from .economy_v2 import EconomyV2, generate
from .tax import (reverse_mortgage_monthly, acquisition_tax_rate, one_house_cgt, TaxConfig, HouseConfig, income_tax_person, private_pension_tax, health_premium,
                  property_tax, comprehensive_property_tax, rent_tax, gift_tax, hi_property_monthly,
                  retirement_income_tax, RET_INCOME_PENSION_RATIO)


def glide_path(hh: Household, T):
    """(v29) 나이별 주식 비중 경로. glide_end가 없으면 평생 같은 비중."""
    if hh.glide_end is None:
        return np.full(T + 1, float(hh.stock_weight))
    a0, a1 = hh.glide_ages
    ages = hh.members[0].age + np.arange(T + 1)
    return hh.stock_weight + (hh.glide_end - hh.stock_weight) * np.clip((ages - a0) / max(a1 - a0, 1), 0, 1)


def joint_pv(hh: Household, qx_table, t0):
    """부부 중 한 명이라도 생존하는 동안 연 1원의 현가 (실질 할인).
    주택연금·월세 절감처럼 '살아있는 동안 계속 들어오는 돈'을 보호선에서 빼줄 때 쓴다."""
    v = 1 / (1 + hh.floor_real_rate); surv = []
    for m in hh.members:
        q = np.minimum(qx_table[m.sex] * hh.floor_mort_mult, 1.0); x = min(m.age + t0, len(q) - 1)
        surv.append(np.concatenate([[1.0], np.cumprod(1 - q[x:-1])]))
    L = min(len(z) for z in surv); p = surv[0][:L]
    if len(surv) > 1:
        p = surv[0][:L] + surv[1][:L] - surv[0][:L] * surv[1][:L]
    return float(np.sum(v ** np.arange(L) * p))


def annuity_pv(hh: Household, cfg: SimConfig, ad: AddOns, qx_table, members, t0):
    """t0년에 가입할 때 연금 1원(첫해)의 현가. 부부형 생존자 비율·보증기간·체증형 반영.

    지급비율 = 보증기간 중이면 1, 이후 둘 다 생존 1 / 한 명만 생존 survivor_ratio / 아무도 없으면 0.
    보험료 = 일시납 / (이 현가 × (1 + 사업비율)) 로 쓰인다.
    """
    v = 1 / (1 + ad.annuity_rate); g = ad.annuity_escalation; r = ad.annuity_survivor_ratio
    surv = []
    for i in members:
        m = hh.members[i]
        q = np.minimum(mortality.project_qx(qx_table[m.sex], t0) * ad.annuity_mort_mult, 1.0) \
            if cfg.mortality_improvement else np.minimum(qx_table[m.sex] * ad.annuity_mort_mult, 1.0)
        x = min(m.age + t0, len(q) - 1)
        surv.append(np.concatenate([[1.0], np.cumprod(1 - q[x:-1])]))
    L = min(len(z) for z in surv)
    if len(surv) > 1:
        p0, p1 = surv[0][:L], surv[1][:L]
        pay = p0 * p1 + r * (p0 * (1 - p1) + (1 - p0) * p1)
    else:
        pay = surv[0][:L]
    s_ = np.arange(L)
    if ad.annuity_guarantee_years > 0:
        pay = np.where(s_ < ad.annuity_guarantee_years, 1.0, pay)
    return float(np.sum((v ** s_) * ((1 + g) ** s_) * pay))


def care_insurance_premium(hh: Household, ad: AddOns, alive, incare, trig, T):
    """간병보험료를 수지상등 원칙으로 산출: (1 + 사업비율) × 급부 현가 / 보험료 납입 현가.

    이 모델이 뽑은 간병·사망 경로(trig = 지급 트리거) 그대로를 쓰므로,
    보험료가 이 가구의 실제 위험을 반영한다.
    """
    i0 = ad.care_member; v = (1 + ad.care_rate) ** -np.arange(T + 1)
    ages_i = hh.members[i0].age + np.arange(T + 1)
    pv_ben = (trig[i0] * v).sum(1).mean() * ad.care_benefit
    pay = alive[i0] & ~incare[i0] & (ages_i < ad.care_pay_until)[None, :]
    pv_prem = (pay * v).sum(1).mean()
    return (1 + ad.care_loading) * pv_ben / max(pv_prem, 1e-9)


def life_paths(hh: Household, cfg: SimConfig, qx_table, T, n, rng, markov: bool):
    """가구원의 생존·간병 경로와 간병비 수준을 뽑는다.

    반환: alive, incare, severe (각 (구성원, 경로, 연도) 불리언), care_level, sev_level (경증·중증 연간 비용).
    markov=False면 간병 단계를 쓰지 않으므로 incare·severe·비용은 None.
    난수는 호출 순서대로 rng에서 뽑으므로, 이 함수의 호출 위치를 바꾸면 경로가 바뀐다.
    """
    k = len(hh.members)
    qmult = mortality.longevity_multiplier(n, T, cfg.longevity_sigma_level,
                                           cfg.longevity_sigma_trend, cfg.seed)   # (v30) 장수 불확실성
    if markov:
        lives = [mortality.simulate_life(qx_table[m.sex], m.age, T, n, rng, cfg.care, return_severe=True,
                                         improve=cfg.mortality_improvement, qmult=qmult) for m in hh.members]
        alive = np.stack([l[0] for l in lives])
        incare = np.stack([l[1] for l in lives]); severe = np.stack([l[2] for l in lives])
        care_level = cfg.care.cost_median * np.exp(cfg.care.cost_log_sigma * rng.standard_normal((k, n)))
        sev_level = cfg.care.cost_severe_median * np.exp(cfg.care.cost_severe_sigma * rng.standard_normal((k, n)))
    else:
        alive = np.stack([mortality.simulate_alive(qx_table[m.sex], m.age, T, n, rng,
                                                   improve=cfg.mortality_improvement, qmult=qmult)
                          for m in hh.members])
        incare = severe = care_level = sev_level = None
    if k == 2:                                                  # (v30) 사별 효과
        alive, incare, severe = mortality.apply_widowhood(
            alive, incare, severe, [qx_table[m.sex] for m in hh.members],
            [m.age for m in hh.members], [m.sex for m in hh.members],
            cfg.widow_excess, cfg.widow_tau, cfg.mortality_improvement, cfg.seed,
            qmult, cfg.widow_spike)
    return alive, incare, severe, care_level, sev_level


def dependent_care(hh: Household, cfg: SimConfig, qx_table, T, n, rng, markov: bool):
    """(v36) 더블케어: 부양하는 부모의 간병비 (경로, 연도), 실질 기준.

    수명·간병은 가구원과 같은 방식으로 돌리되 생활비는 넣지 않는다(따로 사시므로).
    비용은 share(형제간 분담 비율) 만큼만 이 가구가 댄다.
    """
    dep = np.zeros((n, T + 1))
    proc = cfg.care if markov else CareMarkov()        # 간병 전이 과정(비용 수준은 아래에서 cfg.care 기준)
    c = cfg.care
    sev_med = getattr(c, "cost_severe_median", c.cost_median)      # CareShock에는 중증 단계가 없다
    sev_sig = getattr(c, "cost_severe_sigma", c.cost_log_sigma)
    for d in (hh.dependents or []):
        share = float(d.get("share", 1.0))
        if share <= 0:
            continue
        _, d_in, d_sev = mortality.simulate_life(
            qx_table[d.get("sex", "F")], int(d["age"]), T, n, rng, proc,
            return_severe=True, improve=cfg.mortality_improvement)
        lvl = c.cost_median * np.exp(c.cost_log_sigma * rng.standard_normal(n))
        slv = sev_med * np.exp(sev_sig * rng.standard_normal(n))
        dep += share * ((d_in & ~d_sev) * lvl[:, None] + d_sev * slv[:, None])
    return dep


def floor_table(hh: Household, cfg: SimConfig, e: EconomyV2, house: HouseConfig, qx_table, T, ess_base):
    """기본생활비 보호선 F[t, 상태]: 생존확률로 가중한 '앞으로 필요한 기본생활비 부족분'의 현가.

    상태 0 = 부부 모두 생존(1인 가구는 본인 생존), 1 = 본인만 생존, 2 = 배우자만 생존.
    F[t, 상태] = Σ_s v^s Σ_상태' P(상태'|상태, t→t+s) · max(기본생활비×비율 − 보장소득_실질, 0)
    경로에 의존하지 않으므로(시뮬레이션 전에 한 번 계산) 루프 밖에서 표로 만들어 둔다.
    """
    k = len(hh.members)
    v_r = 1 / (1 + hh.floor_real_rate); pibar = e.pi_theta
    qf = [np.minimum(qx_table[m.sex] * hh.floor_mort_mult, 1.0) for m in hh.members]

    def inc_vec(i, yrs):                                   # 구성원 i의 보장소득(실질) 벡터
        m = hh.members[i]; ages = m.age + yrs
        x = np.where(ages >= m.nps_start_age, pension.nps_annual_amount(m, cfg.nps), 0.0)
        mask = (m.private_pension_start <= ages) & (ages < m.private_pension_start + m.private_pension_years)
        return x + np.where(mask, m.private_pension_annual / (1 + pibar) ** yrs, 0.0)

    def inc_alone(i, yrs):                                 # (v28) 배우자 사망 후: 유족연금 선택 반영
        base = inc_vec(i, yrs)
        if k < 2 or not cfg.nps.survivor_on:
            return base
        m = hh.members[i]; ages = m.age + yrs
        own = np.where(ages >= m.nps_start_age, pension.nps_annual_amount(m, cfg.nps), 0.0)
        S = pension.survivor_annual(hh.members[1 - i])
        o2, s2 = pension.survivor_split(own, S, cfg.nps.survivor_overlap)
        return base - own + o2 + s2

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
        gap = lambda members, ratio: np.maximum(
            ess_base * ratio - sum((inc_alone if len(members) == 1 and k > 1 else inc_vec)(i, yrs)
                                   for i in members) - rent_real, 0.0)
        g0 = gap([0], 1.0 if k == 1 else hh.survivor_spending_ratio)
        if k == 1:
            floor_tab[t0, :] = np.sum(vs * surv[0] * g0); continue
        g1 = gap([1], hh.survivor_spending_ratio); gb = gap([0, 1], 1.0)
        p0, p1 = surv[0], surv[1]
        floor_tab[t0, 0] = np.sum(vs * (p0 * p1 * gb + p0 * (1 - p1) * g0 + (1 - p0) * p1 * g1))
        floor_tab[t0, 1] = np.sum(vs * p0 * g0)
        floor_tab[t0, 2] = np.sum(vs * p1 * g1)
    return floor_tab


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
    markov = isinstance(cfg.care, CareMarkov) and cfg.care.enabled
    alive, incare, severe, care_level, sev_level = life_paths(hh, cfg, qx_table, T, n, rng, markov)
    hh_alive = alive.sum(0) > 0
    own = np.array(tc.ownership if tc.ownership else [1 / k] * k, float)
    hown = np.array(house.owner_share if house.owner_share else [1.0] + [0.0] * (k - 1), float)

    Wt = np.full(n, float(hh.liquid_assets)); Bt = Wt.copy(); Wi = np.zeros(n); Wp = np.full(n, float(hh.pension_balance)); Pp = np.zeros(n)
    isa_in = np.zeros(n); cg_due = np.zeros(n)
    gifts_hist = []                                   # (t, 명목 증여액, 증여세)
    gift_val = np.zeros(n)                          # 증여한 돈의 현재 가치(자녀도 같은 수익률로 운용 가정, 명목)
    care_left = np.zeros((k, n), int); care_cost = np.zeros((k, n))
    dep_care = dependent_care(hh, cfg, qx_table, T, n, rng, markov)
    # (v36) 간병비 전용 물가: 일반 물가 × (1 + 초과상승률)^t
    care_cpi = cpi * (1 + cfg.care_real_growth) ** np.arange(T + 1)[None, :]
    dep_at = np.full(n, -1)
    tax_y = np.zeros((n, T)); hi_y = np.zeros((n, T)); prop_y = np.zeros((n, T))
    estate_real = np.full(n, np.nan); etax_real = np.full(n, np.nan); transfer_real = np.full(n, np.nan)
    w = hh.stock_weight
    w_path = glide_path(hh, T)
    W_hist = np.zeros((n, T + 1)); W_hist[:, 0] = Wt
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
    annuity_factor = lambda t0: annuity_pv(hh, cfg, ad, qx_table, ann_members, t0)
    ci_on = ad.care_benefit > 0 and markov and ad.care_member < k
    trig = None                                                # 간병보험 지급 트리거(중증만 / 경증부터)
    if ci_on:
        trig = severe if (ad.care_trigger == "severe" and cfg.care.two_level) else incare
        ci_prem = care_insurance_premium(hh, ad, alive, incare, trig, T)
    else:
        ci_prem = 0.0
    stock_ret = np.full((n, T), np.nan); port_ret = np.full((n, T), np.nan)
    # (v36) 패닉 매도: 경로마다 '지금 실제로 들고 있는 주식 비중'을 따로 들고 간다
    panic_on = cfg.panic_drop > 0
    w_now = np.full(n, float(w_path[0]))
    prev_life = None                                           # 작년에 쓴 여행·취미 예산(실질) — 지출 경직성용

    floor_tab = floor_table(hh, cfg, e, house, qx_table, T, ess_base) if hh.floor_method == "actuarial" else None
    short_real = np.zeros(n)                                   # 생존 중 부족했던 기본생활비 누계(실질)
    cons_u = np.zeros(n); cons_a = np.zeros(n)                 # (v27) CRRA 효용 누적, 할인·생존 가중 합
    gamma = getattr(cfg, 'risk_aversion', 3.0); beta = 1 / 1.02

    # ── v19 주택 활용: 주택연금 / 집 줄이기 ──
    hscale = np.ones(n)                                        # 현재 집 가치 / 처음 집 가치 (다운사이징 등)
    hidx = eco.get("house")                                    # (v26) 실질 집값 지수 — 명목 = hidx × cpi
    if hidx is None: hidx = np.ones((n, T + 1))
    hp = lambda tt: hscale * hidx[:, tt] * cpi[:, tt]          # 집값 배율(처음 시세 기준, 명목)
    rm_t = None if house.reverse_mortgage_age is None else max(0, house.reverse_mortgage_age - youngest)
    ds_t = None if house.downsize_age is None else max(0, house.downsize_age - hh.members[0].age)
    if rm_t is not None and ds_t is not None and rm_t <= ds_t:
        ds_t = None                                            # 주택연금 가입 후에는 집을 팔 수 없음
    rm_pay = np.zeros(n); rm_bal = np.zeros(n); rm_on = np.zeros(n, bool)
    rm_basis = house.cost_basis if house.cost_basis is not None else 0.5 * house.market
    rm_monthly_real = 0.0; ds_cash_real = 0.0
    # ── v21 큰 지출 이벤트: 부족하면 은퇴 전에는 대출(금리 + 1.5%p), 은퇴 시점에 남은 빚은 자산으로 상환 ──
    debt = np.zeros(n); extra_house = np.zeros(n)             # extra_house: 이벤트로 산 집(오늘 가치)
    mort_bal = np.zeros(n); mort_pay = np.zeros(n); mort_rate = np.zeros(n)   # 주택담보대출(원리금균등, 명목)
    rent_save = np.zeros(n)                                    # (v24) 집을 사서 더 안 내게 된 월세·전세대출 이자(연, 오늘 가치)
    debt_retire_real = np.full(n, np.nan); debt_settled = False
    # (v48) 이미 갚고 있는 대출(주담대·신용대출 등): 원리금균등, 명목 고정 상환액.
    #   은퇴 전 상환은 소득에서(입력한 저축액은 상환 후 금액), 은퇴 후 남은 상환은 노후자금에서.
    od_bal = np.zeros(n); od_pay = np.zeros(n); od_rate = np.zeros(n)
    if getattr(hh, "debt_balance", 0) > 0:
        N0 = max(int(getattr(hh, "debt_years", 10) or 1), 1)
        i0 = hh.debt_rate if getattr(hh, "debt_rate", None) else np.maximum(eco["rate"][:, 0], 0) + 0.015
        od_rate = np.broadcast_to(np.asarray(i0, float), (n,)).copy()
        od_bal = np.full(n, float(hh.debt_balance))
        od_pay = od_bal * od_rate / (1 - (1 + od_rate) ** -N0)
    # (v49) 퇴직급여: DC 적립(Wdc, 은퇴 전) → 은퇴 때 IRP(Wr)로. Rr = 아직 안 꺼낸 이연퇴직소득 원금, tau = 퇴직소득세 실효율
    Wdc = np.zeros(n); Wr = np.zeros(n); Rr = np.zeros(n); tau = np.zeros(n); t_ret = None
    sev_real = np.zeros(n); sev_tax_real = np.zeros(n)
    joint_factor = lambda t0: joint_pv(hh, qx_table, t0)

    def sell_from_taxable(amount):
        """과세계좌에서 매도: 실현이익 반환(해외주식분만 과세 대상)."""
        nonlocal Wt, Bt
        amt = np.minimum(np.maximum(amount, 0), Wt)
        ratio = np.where(Wt > 0, Bt / np.maximum(Wt, 1e-9), 1)
        gain = amt * np.maximum(1 - ratio, 0)
        Bt = np.maximum(Bt - amt * np.minimum(ratio, 1), 0); Wt = Wt - amt
        return amt, gain * w * tc.overseas_share

    for t in range(T):
        w_target = float(w_path[t])
        if panic_on:
            if t > 0:                                           # 작년 주가가 많이 빠졌으면 줄이고, 아니면 천천히 복귀
                hit = eco["stock"][:, t - 1] <= -cfg.panic_drop
                w_now = np.where(hit, w_target * cfg.panic_cut,
                                 w_now + (w_target - w_now) * cfg.panic_recover)
            w = np.clip(w_now, 0.0, 1.0)
        else:
            w = w_target
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
        surv = np.zeros((k, n)); surv_flag = [np.zeros(n, bool) for _ in range(k)]
        if k == 2 and cfg.nps.survivor_on:                  # (v28) 유족연금: 배우자 사망 후 max(유족연금, 본인연금 + 30%×유족연금)
            for i in range(2):
                S = pension.survivor_annual(hh.members[1 - i]) * cpi[:, t] * (a[i] & ~a[1 - i])
                o2, s2 = pension.survivor_split(nps[i], S, cfg.nps.survivor_overlap)
                nps[i] = np.where(S > 0, o2, nps[i]); surv[i] = s2; surv_flag[i] = s2 > 0
        bz = np.zeros((k, n)); work = np.zeros((k, n), bool)
        for i, m in enumerate(hh.members):
            b = biz[i]
            if b and m.age + t < b["until_age"]:
                bz[i] = a[i] * b["income"] * cpi[:, t]
                work[i] = a[i] & bool(b.get("workplace", False))
        rent = house.rent_annual * cpi[:, t] * live
        if ds_t is not None and t == ds_t and house.market > 0:  # 집 줄이기: 팔고 작은 집 사기
            sell = house.market * hp(t)
            cgt = one_house_cgt(sell, rm_basis, house.years_held + t) if house.n_houses == 1 else 0.3 * np.maximum(sell - rm_basis, 0)
            newp = sell * house.downsize_ratio
            acq = acquisition_tax_rate(newp) * newp
            cash = np.maximum(sell * (1 - house.sell_cost) - cgt - newp - acq, 0) * live
            Wt += cash; Bt += cash; hscale = hscale * house.downsize_ratio
            ds_cash_real = float(np.median(cash[live] / cpi[live, t])) if live.any() else 0.0
        if rm_t is not None and t == rm_t and house.market > 0:  # 주택연금 가입 (공시 12억 이하 1주택)
            V = house.market * hp(t)
            elig = live & (house.n_houses == 1) & (house.official * hp(t) <= 120000)
            m_pay = reverse_mortgage_monthly(youngest + t, V)
            rm_pay = np.where(elig, 12 * m_pay, 0.0); rm_on = elig; rm_bal = np.where(elig, 0.01 * V, 0.0)
            rm_monthly_real = float(np.median(m_pay[elig] / cpi[elig, t])) if elig.any() else 0.0
        rm_inc = rm_pay * live
        ev_cost = np.zeros(n)
        for ev in hh.events:                                   # 큰 지출 이벤트 (본인 나이 기준)
            age0 = hh.members[0].age + t
            if ev.get("kind") == "house":
                if age0 == ev["age"]:
                    price = ev["amount"] * hp(t); ltv = ev.get("ltv", 0.6); N = ev.get("loan_years", 30)
                    loan = price * ltv * live
                    i_m = np.maximum(eco["rate"][:, t], 0) + 0.015                  # 고정금리 = 당시 금리 + 1.5%p
                    mort_bal = mort_bal + loan; mort_rate = np.where(loan > 0, i_m, mort_rate)
                    mort_pay = mort_pay + loan * i_m / (1 - (1 + i_m) ** -N)
                    down = price - loan + acquisition_tax_rate(price) * price
                    ev_cost += (down - ev.get("deposit_back", 0) * cpi[:, t]) * live   # 전세 보증금 회수분은 자기자본으로
                    extra_house = extra_house + ev["amount"] * live
                    rent_save = rent_save + ev.get("rent_saving", 0) * live
            elif ev["age"] <= age0 < ev["age"] + ev.get("years", 1):
                ev_cost += ev["amount"] * cpi[:, t] * live
        m_now = np.where(mort_bal > 1e-6, np.minimum(mort_pay, mort_bal * (1 + mort_rate)), 0.0) * live
        pre_ret = hh.retire_age is not None and hh.members[0].age + t < hh.retire_age
        mort_from_income = all(ev.get("mortgage_from_income", True) for ev in hh.events if ev.get("kind") == "house")
        if not (pre_ret and mort_from_income):                                # 은퇴 전 원리금은 근로소득(생활비)에서, 은퇴 후엔 노후자금에서
            ev_cost += m_now
        mort_bal = np.maximum(mort_bal * (1 + mort_rate) - m_now, 0)
        od_now = np.where(od_bal > 1e-6, np.minimum(od_pay, od_bal * (1 + od_rate)), 0.0) * live
        if not pre_ret:                                                       # (v48) 기존 대출: 은퇴 후 상환분만 노후자금에서
            ev_cost += od_now
        od_bal = np.maximum(od_bal * (1 + od_rate) - od_now, 0) * live
        official = (house.official * hscale + extra_house * 0.69) * hidx[:, t] * cpi[:, t]     # 새 집 공시가격 ≈ 시세의 69%
        # (v28) 기초연금: 65세 이상, 소득인정액(소득평가액 + 재산 소득환산) 기준
        basic_inc = np.zeros(n)
        if cfg.nps.basic_on:
            elig = [a[i] & (hh.members[i].age + t >= 65) for i in range(k)]
            if any(e.any() for e in elig):
                inc_m = (nps.sum(0) + surv.sum(0) + priv.sum(0) + rent + bz.sum(0)) / 12
                fin_now = np.maximum(Wt + Wi + Wp + Wr + Wdc + S + D - debt, 0)
                gross = np.maximum(official - np.where(rm_on, rm_bal, 0.0) - mort_bal - od_bal, 0)
                a_m = [pension.a_benefit_monthly(m) * cpi[:, t] for m in hh.members]      # A급여액(명목)
                basic_inc = pension.basic_pension_annual(inc_m, gross, fin_now, elig,
                                                         [(nps[i] + surv[i]) / 12 for i in range(k)],
                                                         a_m, surv_flag, cfg.nps, cpi[:, t]) * live

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
        care = dep_care[:, t] * care_cpi[:, t] * live          # (v36) 부모 간병 분담분
        if markov:
            care = care + ((incare[:, :, t] & ~severe[:, :, t]) * care_level
                           + severe[:, :, t] * sev_level).sum(0) * care_cpi[:, t]
        elif cfg.care.enabled:
            c = cfg.care
            for i, m in enumerate(hh.members):
                new = a[i] & (care_left[i] == 0) & (m.age + t >= c.start_age) & (rng.random(n) < c.lam)
                care_cost[i] = np.where(new, c.cost_median * np.exp(c.cost_log_sigma * rng.standard_normal(n)), care_cost[i])
                care_left[i] = np.where(new, c.duration_years, care_left[i])
                act = a[i] & (care_left[i] > 0); care += act * care_cost[i] * care_cpi[:, t]
                care_left[i] = np.where(act, care_left[i] - 1, 0)

        # 6) 연금계좌 인출
        if ad.annuity_premium > 0 and t == ann_t:                  # 종신연금 일시납 가입
            prem = np.minimum(ad.annuity_premium * cpi[:, t] * live, Wt); Wt -= prem; Bt = np.maximum(Bt - prem, 0)
            ann_pay = prem / (annuity_factor(t) * (1 + ad.annuity_loading))
        if t >= ann_t:
            if len(ann_members) > 1:
                both = a[ann_members[0]] & a[ann_members[1]]
                one = (a[ann_members[0]] | a[ann_members[1]]) & ~both
                ratio = np.where(both, 1.0, np.where(one, ad.annuity_survivor_ratio, 0.0))
            else:
                ratio = a[ann_members[0]].astype(float)
            if t - ann_t < ad.annuity_guarantee_years:
                ratio = np.maximum(ratio, 1.0)                 # 보증기간: 사망해도 지급(상속인)
            ann_inc = ann_pay * ratio * (1 + ad.annuity_escalation) ** (t - ann_t)
        else:
            ann_inc = np.zeros(n)
        ci_ben = np.zeros(n); ci_cost = np.zeros(n)
        if ci_on:
            i0 = ad.care_member
            ci_ben = ad.care_benefit * trig[i0, :, t]
            ci_cost = ci_prem * (a[i0] & ~incare[i0, :, t] & (hh.members[i0].age + t < ad.care_pay_until))
        # (v42) 정기보험: 보장기간 안에 피보험자가 사망한 바로 그 해에 보험금, 그 전까지는 보험료
        dl_ben = np.zeros(n); dl_cost = np.zeros(n)
        if ad.death_benefit > 0 and ad.death_member < k and t < ad.death_term_years:
            j = ad.death_member
            died_now = alive[j, :, t - 1] & ~a[j] if t > 0 else ~a[j]
            dl_ben = ad.death_benefit * died_now * live      # 남은 가족이 있어야 쓸 수 있다
            dl_cost = ad.death_premium * a[j]
        income_all = (nps.sum(0) + surv.sum(0) + basic_inc + priv.sum(0) + rent + bz.sum(0)
                      + ann_inc + ci_ben + dl_ben + rm_inc)
        if retired:
            # 목표 우선순위: 기본생활(Essential) 보호선을 먼저 지키고, 남는 만큼만 여행·취미(Lifestyle) 지출
            if floor_tab is not None:
                st_idx = np.where(a[0] & (a[1] if k > 1 else True), 0, np.where(a[0], 1, 2)) if k > 1 else np.zeros(n, int)
                floor = floor_tab[t, st_idx] * cpi[:, t] * live
                if ad.annuity_premium > 0 and t >= ann_t:           # 종신연금 소득은 보장소득으로 보호선에서 차감
                    floor = np.maximum(floor - ann_inc * annuity_factor(t) / (1 + e.pi_theta), 0)
                if rm_on.any():                                        # 주택연금(명목 정액)도 보장소득으로 차감
                    floor = np.maximum(floor - rm_inc * joint_factor(t) / (1 + e.pi_theta), 0)
                if rent_save.any():                                    # 자가 전환으로 줄어든 주거비만큼 보호선도 낮춤
                    floor = np.maximum(floor - rent_save * cpi[:, t] * live * joint_factor(t), 0)
            else:
                yrs_left = max(5, hh.planning_age - (youngest + t))
                ann = (1 - 1.02 ** -yrs_left) / 0.02            # (v12) 실질 2%로 할인한 확정 연금현가계수
                floor = np.maximum(ess - income_all, 0) * ann
            life_paid = np.clip(Wt + Wi + Wp + Wr + S + D - floor, 0, life_want)
            if cfg.lifestyle_floor > 0 and prev_life is not None:
                # (v36) 지출 경직성: 형편이 나빠져도 한 해에 이 비율 넘게는 못 줄인다.
                #       줄이지 못한 만큼은 그대로 쓰게 되므로 기본생활 쪽 여유가 더 빨리 깎인다.
                keep = prev_life * cpi[:, t] / np.maximum(cpi[:, t - 1] if t else cpi[:, 0], 1e-9)
                life_paid = np.minimum(np.maximum(life_paid, keep * (1 - cfg.lifestyle_floor)), life_want)
            prev_life = life_paid
            ess_eff = np.maximum(ess - rent_save * cpi[:, t] * live, 0)            # 자가 전환: 은퇴 후 주거비(월세 등) 절감
            spend = ess_eff + life_paid
            if hh.lifestyle > 0:
                life_ratio[:, t] = np.where(live, life_paid / np.maximum(life_want, 1e-9), np.nan)
            saving = 0.0
        else:
            spend = np.zeros(n)
            sal = sum(a[i] * m.salary * (1 + hh.wage_growth) ** t for i, m in enumerate(hh.members)) * cpi[:, t]
            saving = (sal * hh.saving_rate if hh.saving_rate > 0 else hh.annual_saving * cpi[:, t]) * live
            if rent_save.any():                                   # 은퇴 전: 안 내게 된 월세가 대출 원리금보다 많으면 그 차이만큼 추가 저축
                saving = saving + np.maximum(rent_save * cpi[:, t] - m_now, 0) * live
            if hh.retirement_contrib and getattr(hh, "ret_plan", "DC") == "DC":   # DC: 매년 연봉의 1/12 회사 적립(과세이연)
                Wdc += sal / 12 * live
            if ad.pension_contrib_annual > 0:                     # 저축 일부를 연금계좌로, 세액공제 환급은 과세계좌로
                pc = ad.pension_contrib_annual * cpi[:, t] * live
                credit = np.minimum(pc, 900) * ad.pension_credit_rate
                Wp += pc; saving = saving - pc + credit
        need = spend + care + ci_cost + dl_cost + ev_cost - income_all - saving
        # (v49) 은퇴 첫해: 퇴직급여를 IRP로. DB는 마지막 월급 × 근속연수, DC는 쌓인 적립금. 퇴직소득세는 사람별로 계산
        if retired and t_ret is None:
            t_ret = t
            if hh.retirement_contrib and hh.retire_age is not None:
                ty = t                                                 # 은퇴까지 일한 햇수
                R_i = []
                sal_t = [m.salary * (1 + hh.wage_growth) ** max(ty - 1, 0) for m in hh.members]
                if getattr(hh, "ret_plan", "DC") == "DB":
                    for i, m in enumerate(hh.members):
                        R_i.append(sal_t[i] / 12 * (m.tenure + ty) * cpi[:, t] * alive[i, :, max(t - 1, 0)])
                else:
                    tot = max(sum(sal_t), 1e-9)
                    R_i = [Wdc * sal_t[i] / tot for i in range(k)]
                R_tot = sum(R_i)
                tax_tot = sum(retirement_income_tax(R_i[i], hh.members[i].tenure + ty) for i in range(k))
                tau = np.where(R_tot > 0, tax_tot / np.maximum(R_tot, 1e-9), 0.0)
                Wr = Wr + R_tot; Rr = Rr + R_tot; Wdc = np.zeros(n)
                sev_real = R_tot / cpi[:, t]; sev_tax_real = tax_tot / cpi[:, t]
        ret_tax_now = np.zeros(n); g_taxable = np.zeros(n); ret_draw = np.zeros(n)
        if retired and t_ret is not None and Wr.any() and hh.members[0].age + t >= 55:
            kyr = t - t_ret + 1                                        # 연금수령연차
            cap = Wr * 1.2 / max(11 - kyr, 1) if kyr <= 10 else Wr     # 연금수령한도
            # 연금계좌(IRP 포함)에서는 v48까지와 같이 1인당 연 pen_withdraw_cap(1,500만 원)까지만 꺼내고 나머지는 과세계좌에서.
            # (필요한 만큼 IRP에서 한꺼번에 꺼내면 과세계좌가 그대로 남아 금융소득세·건보료가 늘어난다 — 측정으로 확인)
            draw = np.minimum(np.minimum(np.maximum(need, 0), np.minimum(cap, tc.pen_withdraw_cap * na)), Wr)
            part_r = np.minimum(draw, Rr); part_g = draw - part_r      # 이연퇴직소득부터, 그다음 운용수익
            ret_tax_now = part_r * tau * RET_INCOME_PENSION_RATIO[0 if kyr <= 10 else 1]
            g_taxable = part_g
            Wr = Wr - draw; Rr = Rr - part_r; need = need - draw; ret_draw = draw
        pw = np.zeros(n); pw_taxable = np.zeros(n)
        pen_self = ad.pension_contrib_annual > 0 or hh.pension_balance > 0 or hh.retirement_contrib
        # 세금을 끄는 것(tc.enabled=False)은 '세금을 안 매긴다'는 뜻이어야 한다.
        # 전에는 이 줄이 연금계좌 인출 자체를 막아, 세금을 끄면 퇴직연금이 동결돼 결과가 거꾸로 나왔다.
        pen_active = tc.use_pension or pen_self
        pen_open = (tc.use_pension and t >= tc.pen_wait_years) or (pen_self and retired and hh.members[0].age + t >= 55)
        if pen_active and pen_open:
            pw = np.clip(np.minimum(need, np.maximum(tc.pen_withdraw_cap * na - ret_draw, 0)), 0, Wp)   # (v49) IRP 인출과 합쳐 한도
            frac = np.where(Wp > 0, np.clip(1 - Pp / np.maximum(Wp, 1e-9), 0, 1), 0)
            Pp = np.maximum(Pp - pw * (1 - frac), 0); Wp -= pw; pw_taxable = pw * frac
        pw_taxable = pw_taxable + g_taxable                           # (v49) IRP 운용수익은 연금소득

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
            if house.official > 0 or extra_house.any():
                oldest = max(m.age for m in hh.members) + t
                # 재산세: 물건별(가구 합산 근사) / 종부세: 인별 과세(명의 비율대로 1인 공제 9억)
                prop = property_tax(official, house.n_houses, house.fmv_ratio_prop) * np.where(rm_on, 0.75, 1.0)  # 주택연금 재산세 25% 감면
                co_owned = (hshare > 0).sum(0) >= 2
                per_person = sum(a[i] * comprehensive_property_tax(hshare[i] * official, max(house.n_houses, 2),
                                 hh.members[i].age + t, house.years_held + t, house.fmv_ratio_cjs) for i in range(k))
                if house.n_houses == 1:   # 1주택: 단독명의 12억 공제+고령자·장기보유 공제 / 공동명의는 1인 9억 공제와 특례 중 유리한 쪽
                    single = comprehensive_property_tax(official, 1, oldest, house.years_held + t, house.fmv_ratio_cjs)
                    cjs = np.where(co_owned, np.minimum(per_person, single), single)
                else:
                    cjs = per_person
                prop = (prop + cjs) * live
            # 재산분 건보료: 입력값이 있으면 그 값(명목으로 환산), 없으면 별표4 점수표로 자동 계산.
            # 자동 계산은 명목 과세표준에 바로 등급을 적용하므로 따로 물가 환산을 하지 않는다.
            pm = (house.hi_property_monthly * hscale * hidx[:, t] * cpi[:, t]
                  if house.hi_property_monthly is not None
                  else hi_property_monthly(sum(prop_base), tc))
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
        if tc.enabled:
            taxes = taxes + ret_tax_now
        tax_y[:, t] = taxes / cpi[:, t]; hi_y[:, t] = hi / cpi[:, t]; prop_y[:, t] = prop / cpi[:, t]

        # 8) 인출: 과세계좌 → ISA → (부족 시) 연금계좌 연금외수령
        rest = need - pw + taxes + hi + prop
        if retired and not debt_settled:                       # 은퇴 첫해: 남은 빚은 금융자산으로 상환
            debt_retire_real = np.where(live, debt / cpi[:, t], debt_retire_real); rest = rest + debt; debt = np.zeros(n); debt_settled = True
        if hh.draw_order == "isa_first":                       # (v29) ISA(비과세·분리과세) 먼저 소진
            from_i = np.minimum(np.maximum(rest, 0), Wi); Wi -= from_i
            amt, gain = sell_from_taxable(np.maximum(rest, 0) - from_i); realized += gain
        else:
            amt, gain = sell_from_taxable(np.maximum(rest, 0)); realized += gain
            from_i = np.minimum(np.maximum(rest, 0) - amt, Wi); Wi -= from_i
        surplus = np.maximum(-rest, 0)
        repay = np.minimum(surplus, debt); debt -= repay; surplus -= repay   # 여유 저축은 빚부터 갚음
        Wt += surplus; Bt += surplus
        short = np.maximum(rest, 0) - amt - from_i
        from_d = np.minimum(short, D); D -= from_d; short -= from_d
        from_s = np.minimum(short, S); S -= from_s; short -= from_s
        if Wr.any():                                                  # (v49) IRP 연금외수령: 퇴직소득세 전액(감면 없음)
            rate_l = tau if tc.enabled else np.zeros(n)
            take = np.minimum(short / np.maximum(1 - rate_l, 1e-9), Wr)
            pr = np.minimum(take, Rr); pg = take - pr
            ltax = (pr * rate_l + pg * (0.165 if tc.enabled else 0.0))
            Wr = Wr - take; Rr = Rr - pr
            tax_y[:, t] += ltax / cpi[:, t]
            short = np.maximum(short - (take - ltax), 0)
        if pen_active:
            frac2 = np.where(Wp > 0, np.clip(1 - Pp / np.maximum(Wp, 1e-9), 0, 1), 0)
            gross = np.minimum(short / np.maximum(1 - .165 * frac2, 1e-9), Wp)
            Pp = np.maximum(Pp - gross * (1 - frac2), 0); Wp -= gross
            tax_y[:, t] += gross * frac2 * .165 / cpi[:, t]
            short = np.maximum(short - gross * (1 - .165 * frac2), 0)
        if not retired:                                        # 은퇴 전 부족분은 대출로 (고갈로 보지 않음)
            debt += short * live; short = np.zeros(n)
        newly = (short > 1e-6) & (dep_at < 0) & live; dep_at[newly] = t
        short_real += np.where(live, short, 0) / cpi[:, t]
        if retired:                                            # (v27) 실제 소비(실질)로 CRRA 효용 적립
            na_ = np.maximum(na, 1)
            floor_c = 0.15 * ess_base / np.maximum(na_, 1)            # 소비 0 근처에서 효용이 발산하지 않도록
            c_real = np.maximum((spend - short) / cpi[:, t] / na_, floor_c)   # 1인당 연 실질 소비(만원)
            w_ = beta ** t * live * na_
            cons_u += w_ * c_real ** (1 - gamma) / (1 - gamma)
            cons_a += w_
        cg_due = realized

        # 9) 수익률
        r = w * eco["stock"][:, t] + (1 - w) * eco["bond"][:, t]
        Wt *= (1 + r); Wi *= (1 + r); Wp *= (1 + r); Wr *= (1 + r); Wdc *= (1 + r); gift_val *= (1 + r)
        rt = eco["rate"][:, t]
        D *= (1 + np.maximum(rt, 0))
        debt *= (1 + np.maximum(rt, 0) + 0.015)
        rm_bal = np.where(rm_on, (rm_bal + rm_inc) * (1 + np.maximum(rt, 0) + 0.011 + 0.0075), 0.0)  # 대출이자(금리+1.1%p) + 연 보증료 0.75%
        if ad.stock_amount > 0:                                    # 단일지수(CAPM) + 금리 민감도
            lm = np.log1p(eco["stock"][:, t]) - np.log1p(-e.fee)  # 시장 로그수익(펀드 보수 제외)
            b = ad.stock_beta; s_e = ad.stock_idio_sigma
            li = (rt + b * (lm - rt) + 0.5 * b * (1 - b) * e.s_sigma ** 2 - 0.5 * s_e ** 2
                  + s_e * rng2.standard_normal(n) + ad.stock_rate_beta * (eco["rate"][:, t + 1] - rt))
            S *= np.exp(li); stock_ret[:, t] = np.expm1(li); port_ret[:, t] = r
        W_hist[:, t + 1] = (Wt + Wi + Wp + Wr + Wdc + S + D - debt) * (dep_at < 0)

        # 10) 마지막 생존자 사망 → 상속세(10년 내 증여 합산, 기납부 증여세 공제)
        died_all = live & (alive[:, :, t + 1].sum(0) == 0)
        if died_all.any():
            fin_est = np.maximum(Wt + Wi + Wp + Wr + Wdc + S + D - debt, 0) * (dep_at < 0)
            est = fin_est + np.maximum(house.market * hp(t + 1) - rm_bal, 0) + np.maximum(extra_house * hidx[:, t + 1] * cpi[:, t + 1] - mort_bal, 0)   # 주택연금·주담대 상환 후
            est = np.maximum(est - od_bal, 0)                                 # (v48) 남은 기존 대출은 상속재산에서 차감
            recent = sum(g for (tg, g, _) in gifts_hist if t + 1 - tg < 10) if gifts_hist else 0
            recent_gt = sum(gt for (tg, _, gt) in gifts_hist if t + 1 - tg < 10) if gifts_hist else 0
            fin_ded = np.where(fin_est <= 2000, fin_est, np.where(fin_est <= 10000, 2000, np.minimum(fin_est * .2, 20000)))
            lump = max(50000, 20000 + 5000 * tc.n_children)
            et = np.maximum(progressive_estate(est + recent - lump - fin_ded) - recent_gt, 0)
            estate_real = np.where(died_all, (est - et) / cpi[:, t + 1], estate_real)
            etax_real = np.where(died_all, et / cpi[:, t + 1], etax_real)
            transfer_real = np.where(died_all, (est - et + gift_val) / cpi[:, t + 1], transfer_real)

    return {"depleted_at": dep_at, "youngest_age": youngest, "T": T, "antithetic": eco.get("antithetic", False), "erp_path": eco.get("erp_path"), "tax_y": tax_y, "hi_y": hi_y, "prop_y": prop_y,
            "estate_real": estate_real, "life_ratio": life_ratio, "stock_ret": stock_ret, "port_ret": port_ret, "short_real": short_real, "cons_u": cons_u, "cons_a": cons_a, "risk_aversion": gamma,
            "floor_tab": floor_tab, "debt_retire": debt_retire_real, "rm_monthly": rm_monthly_real, "ds_cash": ds_cash_real, "annuity_pay": float(np.median(ann_pay[ann_pay > 0])) if (ann_pay > 0).any() else 0.0, "care_premium": ci_prem, "legacy_target": hh.legacy_target, "estate_tax_real": etax_real, "transfer_real": transfer_real,
            "W_nominal": W_hist, "W_real": W_hist / cpi, "death_t": np.where(alive.all(2), -1, alive.argmin(2)),   # (구성원, 경로) 사망 연도, 끝까지 생존이면 -1
            "hh_alive": hh_alive, "sev_real": sev_real, "sev_tax_real": sev_tax_real,
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


def cte(x, q: float) -> float:
    """조건부 꼬리기대값: 나쁜 쪽 (1−q) 비율의 평균. 미국 보험 자본규제의 CTE와 같은 정의."""
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(q * len(x)))
    return float(x[k:].mean()) if k < len(x) else float(x[-1])


def goal_summary(res):
    """목표별 달성 지표: 기본생활 유지 확률, 여행·취미 충족률, 유산 달성 확률."""
    d = res["depleted_at"]
    out = {"기본생활 유지 확률": float((d < 0).mean())}
    sr = res.get("short_real")
    if sr is not None:
        out["평균 부족액(실질)"] = float(sr.mean())
        out["부족 시 평균 부족액(실질)"] = float(sr[sr > 0].mean()) if (sr > 0).any() else 0.0
        out["CTE70 부족액(실질)"] = cte(sr, 0.70)          # 나쁜 30% 경우의 평균 부족액
        out["CTE90 부족액(실질)"] = cte(sr, 0.90)          # 나쁜 10% 경우의 평균 부족액
    cu, ca = res.get("cons_u"), res.get("cons_a")
    if cu is not None and ca is not None and ca.mean() > 0:
        g = res.get("risk_aversion", 3.0)
        u = cu.mean() / ca.mean()                                     # 할인·생존 가중 평균 효용
        out["확실성등가 소비(연, 1인)"] = float(((1 - g) * u) ** (1 / (1 - g)))
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
