"""연금보험 상품: 실제 상품설명서의 숫자로 적립·연금·해지를 시뮬레이션한다.

왜 필요한가
  모형의 `AddOns` 는 "종신연금"처럼 **유형** 수준이었다. 그런데 설계사가 파는 것은
  "A생명 ○○연금보험 2종"이고, 공시이율·사업비·최저보증이율·해지공제가 상품마다 다르다.
  같은 돈을 어디에 넣느냐를 비교하려면 그 숫자들이 들어와야 한다.

구조 (공시이율형 연금보험)
  ① 납입: 보험료에서 사업비(계약체결비용 + 계약관리비용)를 떼고 남은 돈이 적립금에 들어간다.
  ② 적립: 적립금은 매년 공시이율로 불어난다. 공시이율은 시장금리를 따라가되
     보험사 자산이 길어 천천히 움직이고(평활), **최저보증이율** 아래로는 내려가지 않는다.
  ③ 연금: 개시 시점의 적립금을 연금액으로 바꾼다. 적립금 ÷ 연금현가계수.
  ④ 해지: 중도 해지하면 해지공제가 붙어 초기에는 낸 돈보다 적게 돌려받는다.

계리적으로 중요한 지점
  최저보증이율은 **금리에 대한 풋옵션**이다. 금리가 떨어진 경로에서만 값을 한다.
  확률 모형이 있으니 그 가치를 몬테카를로로 직접 잴 수 있다 (`guarantee_value`).

기본값 출처 (2026-10 확인)
  · 공시이율 2.3% 수준 — 생명보험사 2025.12 적용 일반연금 2.31%·연금저축 2.28%
  · 최저보증이율 5년 미만 1.0% / 5~10년 0.75% / 10년 이상 0.5% — 표준적 구조
  · 계약체결비용 월납 보험료의 5.6~6.9%(7년 이하), 2.3~3.6%(8~10년), 일시납 4.19%
  · 계약관리비용 납입 중 4.2~9%, 납입 후 적립금의 0.3~2%
  · 해지공제 1년차 25.6%, 2년차 10.7%, 3년차 5.7%, 6년차 0.7%
    (보험연구원 「변액연금 최저보증 및 사업비 부과 현황 조사」)
  상품마다 다르므로 **전부 입력으로 바꿀 수 있게** 두었다. 실제 상담에서는 상품설명서의 숫자를 넣는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import mortality


@dataclass
class AnnuityProduct:
    """공시이율형 연금보험 한 건의 스펙. 금액 단위는 만 원(오늘 가치)."""
    name: str = "공시이율형 연금보험"
    monthly_premium: float = 50.0      # 월 보험료
    pay_years: int = 20                # 납입기간
    start_age: int = 65                # 연금개시 나이

    # ── 적립 이율 ──
    credit_spread: float = -0.008      # 공시이율 = 시장 명목금리 + 스프레드 (자산운용 마진·비용 반영)
    credit_smooth: float = 0.6         # 평활 계수: 공시이율은 시장금리보다 천천히 움직인다 (0=즉시, 1=고정)
    guarantee: tuple = ((5, 0.010), (10, 0.0075), (10 ** 9, 0.005))   # (경과연수 미만, 최저보증이율)

    # ── 사업비 ──
    acq_rate_early: float = 0.062      # 계약체결비용 (보험료 대비, 7년 이하)
    acq_rate_late: float = 0.030       # 계약체결비용 (8~10년)
    acq_years: int = 10
    admin_rate_pay: float = 0.055      # 계약관리비용 (납입 중, 보험료 대비)
    admin_rate_post: float = 0.008     # 계약관리비용 (납입 후, 적립금 대비 연)

    # ── 해지공제 (연차별, 적립금 대비) ──
    surrender: tuple = (0.256, 0.107, 0.057, 0.035, 0.018, 0.007)

    # ── 연금 전환 ──
    annuity_rate: float = 0.02         # 연금액 산출 예정이율
    annuity_mort_mult: float = 0.85    # 연금 가입자는 일반 인구보다 오래 산다(역선택)
    guarantee_years: int = 10          # 보증지급기간
    joint: bool = False                # 부부형
    survivor_ratio: float = 0.6        # 부부형일 때 한 명 사망 후 지급 비율

    def min_rate(self, elapsed: int) -> float:
        """경과 연수에 해당하는 최저보증이율."""
        for upto, r in self.guarantee:
            if elapsed < upto:
                return r
        return self.guarantee[-1][1]

    def surrender_rate(self, elapsed: int) -> float:
        """경과 연수의 해지공제율. 표를 넘어가면 0."""
        return self.surrender[elapsed] if elapsed < len(self.surrender) else 0.0


def credited_rate(prod: AnnuityProduct, nominal_rate, guarantee: bool = True):
    """경로별 공시이율 (n, T). 시장금리를 평활해 따라가고 최저보증이율이 바닥을 받친다.

    guarantee=False 로 두면 보증이 없는 같은 상품이 되어, 두 결과의 차이가 곧 보증의 가치다."""
    nominal_rate = np.asarray(nominal_rate, float)
    n, T = nominal_rate.shape
    target = nominal_rate + prod.credit_spread
    out = np.empty((n, T))
    cur = target[:, 0].copy()
    a = prod.credit_smooth
    for t in range(T):
        cur = a * cur + (1 - a) * target[:, t]           # 지수평활
        out[:, t] = cur
    out = np.maximum(out, 0.0)                            # 공시이율은 음수가 되지 않는다
    if guarantee:
        floor = np.array([prod.min_rate(t) for t in range(T)])[None, :]
        out = np.maximum(out, floor)
    return out


def accumulate(prod: AnnuityProduct, nominal_rate, cpi=None, guarantee: bool = True, until: int = None) -> dict:
    """납입·적립 단계를 경로별로 돌린다. until(연금개시 시점) 이후로는 적립도 비용도 없다 — 연금으로 전환되기 때문.

    반환 (모두 (n, T) 또는 (n, T+1), 명목 만 원)
      balance   : 연말 적립금
      paid      : 그해 낸 보험료
      cost      : 그해 떼인 사업비
      surrender : 그해 말 해지하면 받는 돈
    """
    rate = credited_rate(prod, nominal_rate, guarantee)
    n, T = rate.shape
    cpi = np.ones((n, T + 1)) if cpi is None else np.asarray(cpi, float)
    bal = np.zeros((n, T + 1)); paid = np.zeros((n, T)); cost = np.zeros((n, T)); sur = np.zeros((n, T + 1))
    annual = prod.monthly_premium * 12
    until = T if until is None else max(0, min(until, T))
    for t in range(until):
        prem = annual * cpi[:, t] if t < prod.pay_years else 0.0      # 보험료는 물가에 연동(실질 고정)
        if t < prod.pay_years:
            acq = prod.acq_rate_early if t < 7 else (prod.acq_rate_late if t < prod.acq_years else 0.0)
            fee = prem * (acq + prod.admin_rate_pay)
            paid[:, t] = prem; cost[:, t] = fee
            bal[:, t + 1] = (bal[:, t] + prem - fee) * (1 + rate[:, t])
        else:
            fee = bal[:, t] * prod.admin_rate_post
            cost[:, t] = fee
            bal[:, t + 1] = (bal[:, t] - fee) * (1 + rate[:, t])
        sur[:, t + 1] = bal[:, t + 1] * (1 - prod.surrender_rate(t))
    for t in range(until, T):                                         # 개시 이후: 연금으로 전환되므로 적립금을 그대로 둔다
        bal[:, t + 1] = bal[:, t]; sur[:, t + 1] = sur[:, t]
    return {"balance": bal, "paid": paid, "cost": cost, "surrender": sur, "credited": rate}


def annuity_factor(prod: AnnuityProduct, members, qx_table, start_t: int, improve: bool = True) -> float:
    """연금 1원(첫해)의 현가. 예정이율과 연금 가입자 사망률로 계산한다."""
    v = 1 / (1 + prod.annuity_rate)
    surv = []
    for m in members:
        q = mortality.project_qx(qx_table[m.sex], start_t) if improve else qx_table[m.sex]
        q = np.minimum(q * prod.annuity_mort_mult, 1.0)
        x = min(m.age + start_t, len(q) - 1)
        surv.append(np.concatenate([[1.0], np.cumprod(1 - q[x:-1])]))
    L = min(len(z) for z in surv)
    if len(surv) > 1:
        p0, p1 = surv[0][:L], surv[1][:L]
        pay = p0 * p1 + prod.survivor_ratio * (p0 * (1 - p1) + (1 - p0) * p1)
    else:
        pay = surv[0][:L]
    s = np.arange(L)
    if prod.guarantee_years > 0:
        pay = np.where(s < prod.guarantee_years, 1.0, pay)
    return float(np.sum((v ** s) * pay))


def run(prod: AnnuityProduct, members, qx_table, nominal_rate, cpi=None,
        improve: bool = True, guarantee: bool = True) -> dict:
    """적립 + 연금 전환을 한 번에. income[:, t] 는 t년차에 받는 연금(명목)."""
    n, T = np.asarray(nominal_rate).shape
    age0 = members[0].age
    t_start = max(0, min(prod.start_age - age0, T))
    acc = accumulate(prod, nominal_rate, cpi, guarantee, until=t_start)
    af = annuity_factor(prod, members, qx_table, t_start, improve)
    first = acc["balance"][:, t_start] / af                           # 연금개시 첫해 연금액(명목)
    income = np.zeros((n, T))
    income[:, t_start:] = first[:, None]                              # 공시이율형은 개시 후 정액(단순화)
    return {**acc, "income": income, "annuity_factor": af, "first_payment": first, "start_t": t_start}


def guarantee_value(prod: AnnuityProduct, members, qx_table, nominal_rate, cpi=None, improve: bool = True) -> dict:
    """최저보증이율의 가치 — 보증이 있을 때와 없을 때의 차이.

    보증은 금리가 낮게 흘러간 경로에서만 값을 한다. 그래서 평균보다
    '나쁜 경로에서 얼마나 받쳐 주는가'(하위 10%)가 이 보증의 성격을 더 잘 보여준다."""
    on = run(prod, members, qx_table, nominal_rate, cpi, improve, guarantee=True)
    off = run(prod, members, qx_table, nominal_rate, cpi, improve, guarantee=False)
    d = on["first_payment"] - off["first_payment"]
    binds = (on["credited"] > off["credited"] + 1e-12).any(axis=1)
    return {"평균 연금액 증가": float(d.mean()),
            "중앙값 증가": float(np.median(d)),
            "하위 10% 경로에서 증가": float(d[off["first_payment"] <= np.percentile(off["first_payment"], 10)].mean()),
            "보증이 작동한 경로 비율": float(binds.mean()),
            "보증 있음 연금액(중앙값)": float(np.median(on["first_payment"])),
            "보증 없음 연금액(중앙값)": float(np.median(off["first_payment"]))}


def accumulate_fund(monthly_premium: float, pay_years: int, until: int, stock, bond, cpi,
                    stock_weight: float = 0.5, fee: float = 0.003) -> np.ndarray:
    """같은 보험료를 펀드(주식+채권)에 넣었을 때의 적립금 (n, T+1), 명목.

    보험료에서 사업비를 떼지 않는 대신 운용보수 `fee` 를 매년 뗀다.
    보험사의 적립이율 대신 시장 수익률을 그대로 받으므로 더 오르기도 더 떨어지기도 한다."""
    stock = np.asarray(stock, float); bond = np.asarray(bond, float)
    n, T = stock.shape
    cpi = np.ones((n, T + 1)) if cpi is None else np.asarray(cpi, float)
    bal = np.zeros((n, T + 1))
    annual = monthly_premium * 12
    until = max(0, min(until, T))
    for t in range(T):
        prem = annual * cpi[:, t] if t < min(pay_years, until) else 0.0
        if t < until:
            r = stock_weight * stock[:, t] + (1 - stock_weight) * bond[:, t] - fee
            bal[:, t + 1] = (bal[:, t] + prem) * (1 + r)
        else:
            bal[:, t + 1] = bal[:, t]
    return bal


def years_until_empty(balance0, withdraw_first, stock, bond, cpi, t0: int,
                      stock_weight: float = 0.5, fee: float = 0.003) -> np.ndarray:
    """적립금을 직접 꺼내 쓸 때 몇 년 만에 바닥나는지 (경로별). 끝까지 남으면 남은 햇수를 준다.

    연금보험은 오래 살아도 계속 나오지만 펀드는 그렇지 않다 — 그 차이를 보기 위한 계산."""
    stock = np.asarray(stock, float); n, T = stock.shape
    bal = np.array(balance0, float).copy()
    out = np.full(n, T - t0, float)
    alive = np.ones(n, bool)
    for t in range(t0, T):
        w = withdraw_first * (cpi[:, t] / cpi[:, t0])                  # 물가만큼 올려 꺼낸다(실질 고정)
        bal = bal - w
        empty = alive & (bal <= 0)
        out[empty] = t - t0; alive &= ~empty
        r = stock_weight * stock[:, t] + (1 - stock_weight) * bond[:, t] - fee
        bal = np.maximum(bal, 0) * (1 + r)
    return out


def outlive_probability(empty_years, members, qx_table, t0: int, seed=7, improve: bool = True) -> float:
    """꺼내 쓰던 돈이 **살아 있는 동안** 바닥날 확률.

    연금보험은 오래 살수록 이득이고 펀드는 오래 살수록 위험하다. 그 차이를 한 숫자로 보여준다.
    사망 경로는 상품 비교에만 쓰는 별도 난수라 다른 계산을 흔들지 않는다."""
    rng = np.random.default_rng([int(seed), 77])
    n = len(empty_years)
    T = int(max(empty_years.max(), 1)) + 60
    alive = None
    for m in members:                                        # 가구는 마지막 한 명이 살아 있을 때까지
        a = mortality.simulate_alive(qx_table[m.sex], m.age + t0, T, n, rng, improve=improve)
        alive = a if alive is None else (alive | a)
    last = alive.sum(1) - 1                                  # 마지막 생존 연차
    return float((empty_years < last).mean())


def compare_vehicles(prod: AnnuityProduct, members, qx_table, eco, stock_weight: float = 0.5,
                     deposit_spread: float = -0.002, improve: bool = True) -> dict:
    """같은 돈을 ① 연금보험 ② 펀드 ③ 예금에 넣었을 때를 같은 난수로 비교한다.

    설계사가 실제로 받는 질문 — "월 OO만 원을 어디에 넣는 게 낫습니까" — 에 답하기 위한 것이다.
    공정한 비교가 되도록 세 경우 모두 같은 보험료·같은 기간·같은 시나리오를 쓴다."""
    ins = run(prod, members, qx_table, eco["rate"], eco["cpi"], improve)
    t0 = ins["start_t"]
    fund = accumulate_fund(prod.monthly_premium, prod.pay_years, t0,
                           eco["stock"], eco["bond"], eco["cpi"], stock_weight)
    dep_r = np.maximum(eco["rate"] + deposit_spread, 0.0)
    depo = accumulate_fund(prod.monthly_premium, prod.pay_years, t0,
                           dep_r, dep_r, eco["cpi"], 1.0, fee=0.0)
    pay = ins["first_payment"]                                         # 보험사가 주는 첫해 연금액
    q = lambda v: (float(np.percentile(v, 10)), float(np.median(v)), float(np.percentile(v, 90)))
    age0 = members[0].age
    ey_f = years_until_empty(fund[:, t0], pay, eco["stock"], eco["bond"], eco["cpi"], t0, stock_weight)
    ey_d = years_until_empty(depo[:, t0], pay, dep_r, dep_r, eco["cpi"], t0, 1.0, 0.0)
    return {
        "연금개시 적립금": {"연금보험": q(ins["balance"][:, t0]), "펀드": q(fund[:, t0]), "예금": q(depo[:, t0])},
        "연금보험 첫해 연금액": q(pay),
        "같은 금액을 꺼낼 때 버티는 햇수": {"펀드": q(ey_f), "예금": q(ey_d)},
        "살아 있는 동안 바닥날 확률": {"연금보험": 0.0,
                            "펀드": outlive_probability(ey_f, members, qx_table, t0, improve=improve),
                            "예금": outlive_probability(ey_d, members, qx_table, t0, improve=improve)},
        "개시 나이": age0 + t0,
        "낸 돈(명목 중앙값)": float(np.median(ins["paid"].sum(1))),
        "사업비(명목 중앙값)": float(np.median(ins["cost"].sum(1))),
    }
