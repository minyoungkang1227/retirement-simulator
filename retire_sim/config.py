"""입력 스키마. 모든 금액은 '현재가치 기준 만원/년'."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Person:
    age: int
    sex: str                      # "M" | "F"
    nps_monthly: float = 0.0      # 국민연금 예상 월액(정상수령 기준, 만원)
    nps_normal_age: int = 65      # 정상 수령 개시 나이
    nps_start_age: int = 65       # 실제 수령 개시 나이(조기/연기 반영)
    private_pension_annual: float = 0.0   # 사적연금 연액(명목 고정, 만원)
    private_pension_start: int = 60
    private_pension_years: int = 20


@dataclass
class Household:
    members: List[Person]
    liquid_assets: float          # 금융자산 합계(만원)
    stock_weight: float = 0.5     # 주식 비중(연 1회 리밸런싱)
    annual_spending: float = 3600 # 부부 기준 연 생활비(만원, 현재가치)
    survivor_spending_ratio: float = 0.7  # 한 명 사망 시 생활비 비율
    # ── v12 목표 기반(Goal-Based) 설정 ──
    essential: Optional[float] = None     # 기본생활비(연, 만원). None이면 annual_spending 사용
    lifestyle: float = 0.0                # 여행·취미 등 선택 지출(연, 만원). 자산이 부족하면 먼저 줄임
    legacy_target: float = 0.0            # 남기고 싶은 금액(만원, 현재가치)
    retire_age: Optional[int] = None      # 본인(첫 구성원) 은퇴 나이. None이면 이미 은퇴
    annual_saving: float = 0.0            # 은퇴 전 연 저축액(만원, 현재가치)
    planning_age: int = 95                # 기본생활비 보호선 계산용 계획 나이


@dataclass
class EconomyAssumptions:
    stock_mu: float = 0.06        # 주식 기대 산술수익률
    stock_sigma: float = 0.18
    bond_mu: float = 0.03
    bond_sigma: float = 0.05
    rho_stock_bond: float = 0.0
    # 물가: 이산 OU  pi_{t+1} = pi_t + kappa(theta - pi_t) + sigma*eps
    infl_theta: float = 0.02
    infl_kappa: float = 0.3
    infl_sigma: float = 0.01
    infl_init: float = 0.025


@dataclass
class NPSRules:
    """국민연금 조기/연기 조정률 — 개발 전 최신 기준 재확인 필요."""
    early_cut_per_year: float = 0.06
    defer_add_per_year: float = 0.072
    max_early_years: int = 5
    max_defer_years: int = 5


@dataclass
class CareShock:
    """간병비 점프: 나이 >= start_age인 생존자에게 연 lam 확률로 발생."""
    enabled: bool = True
    start_age: int = 75
    lam: float = 0.03
    cost_median: float = 2000     # 발생 시 연 비용 중앙값(만원)
    cost_log_sigma: float = 0.5
    duration_years: int = 3


@dataclass
class CareMarkov:
    """간병 다중상태 마르코프 모델 (v11): 건강(H) ⇄ 간병(C) → 사망(D).

    매년 전이: H→D q_H(x), H→C (1-q_H)·i(x), C→D min(1, m·q_x), C→H (1-q_C)·rec
    q_H는 생명표 전체 사망률 q_x가 유지되도록 간병 유병률로 보정.
    수치는 임시값 — 장기요양 인정률·사망 통계로 보정 필요.
    """
    enabled: bool = True
    incidence: tuple = ((65, 0.002), (75, 0.01), (85, 0.03), (999, 0.08))  # (이 나이 미만, 연 발생률)
    mort_mult: float = 2.5        # 간병 상태 사망률 배수
    recovery: float = 0.10        # 간병 → 건강 회복 확률(연)
    cost_median: float = 2000     # 간병 중 연 비용 중앙값(만원, 현재가치)
    cost_log_sigma: float = 0.5

    def inc(self, age):
        for lim, v in self.incidence:
            if age < lim:
                return v
        return self.incidence[-1][1]


@dataclass
class SimConfig:
    n_paths: int = 10_000
    max_age: int = 110
    seed: Optional[int] = 42
    economy: EconomyAssumptions = field(default_factory=EconomyAssumptions)
    nps: NPSRules = field(default_factory=NPSRules)
    care: object = field(default_factory=CareShock)   # CareShock(v2) 또는 CareMarkov(v11)
