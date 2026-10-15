"""보장설계 — 가장이 먼저 떠났을 때 남은 가족이 얼마가 있어야 하는가.

재무설계의 여섯 영역 중 이 모델이 마지막까지 비워 두었던 자리다(현금흐름·부채·목표자금·
세금·상속증여·은퇴는 이미 엔진이 다룬다). 새로 만든 계산은 거의 없다 —
`engine_tax.floor_table` 이 이미 **배우자만 생존하는 상태의 보호선**을,
그것도 **유족연금을 차감한 뒤** 계산하고 있어서, 필요보장액은 거기서 바로 나온다.

두 가지 방법을 모두 낸다. 실무에서 쓰는 두 방법이고, 서로 다른 답을 주는 게 정상이다.

  · 니즈분석법 (needs analysis) — 남은 가족이 **쓸 돈**에서 **이미 가진 것**을 뺀다.
    이 모델의 보호선이 곧 '쓸 돈'이라 추가 가정이 없다. 보수적이고 설명하기 쉽다.
  · 생애가치법 (human life value) — 고인이 **앞으로 벌었을 돈** 중 본인 몫을 뺀 것의 현가.
    소득이 높은 가구에서 니즈분석법보다 큰 값이 나온다.

금액 단위는 모델 전체와 같은 '만 원, 오늘 가치'.
"""
import numpy as np

from . import mortality
from .config import Household, SimConfig
from .economy_v2 import EconomyV2
from .engine_tax import floor_table
from .tax import HouseConfig

# floor_table 의 상태 번호: 0 = 부부 모두 생존, 1 = 구성원0만 생존, 2 = 구성원1만 생존.
# 구성원 i 가 사망하면 남는 사람은 1-i 이므로, 그 사람의 생존 상태를 본다.
_SURVIVOR_STATE = {0: 2, 1: 1}


def needs_analysis(hh: Household, cfg: SimConfig, e: EconomyV2 = None, house: HouseConfig = None,
                   qx_table=None, member: int = 0, horizon: int = None) -> dict:
    """니즈분석법 필요 사망보장액 — 나이별로.

    필요액(t) = 남은 배우자의 보호선(t) − 그때 쓸 수 있는 돈
      · 보호선: 기본생활비 중 공적·사적연금(유족연금 포함)으로 못 채우는 부분의 생존확률 가중 현가
      · 쓸 수 있는 돈: 금융자산 + 연금계좌 (집은 팔아야 쓸 수 있으므로 빼지 않는다)
      · 남은 부채는 갚아야 하므로 더한다

    반환: {"ages": [...], "need": [...], "floor": [...], "assets": float}
    need 가 0이면 이미 충분하다는 뜻이다.
    """
    if len(hh.members) < 2:
        raise ValueError("보장설계는 남는 가족이 있어야 의미가 있다 — 2인 가구에서만 쓴다")
    e = e or EconomyV2.enhanced()
    house = house or HouseConfig()
    qx_table = qx_table or mortality.default_qx(cfg.max_age)
    youngest = min(m.age for m in hh.members)
    T = horizon if horizon is not None else cfg.max_age - youngest
    ess = hh.essential if hh.essential is not None else hh.annual_spending

    tab = floor_table(hh, cfg, e, house, qx_table, T, ess)
    floor = tab[:, _SURVIVOR_STATE[member]]
    usable = float(hh.liquid_assets) + float(hh.pension_balance) + float(getattr(hh, "debt", 0.0) * -1)
    need = np.maximum(floor - usable, 0.0)
    return {"ages": (hh.members[member].age + np.arange(T + 1)).tolist(),
            "floor": floor.tolist(), "need": need.tolist(), "assets": usable,
            "방법": "니즈분석법"}


def human_life_value(hh: Household, cfg: SimConfig, e: EconomyV2 = None, qx_table=None,
                     member: int = 0, self_share: float = 0.3, real_rate: float = None) -> dict:
    """생애가치법 필요 사망보장액 — 고인이 앞으로 벌었을 소득 중 가족 몫의 현가.

    self_share: 소득 중 본인에게 쓰였을 비율(이 몫은 사망하면 함께 사라지므로 뺀다).
      부부 가구의 통상 가정은 0.3 내외. 실무 범위 0.25~0.4.
    real_rate: 실질 할인율. 기본값은 가구의 보호선 할인율을 그대로 쓴다(두 방법의 기준을 맞춘다).
    """
    e = e or EconomyV2.enhanced()
    qx_table = qx_table or mortality.default_qx(cfg.max_age)
    m = hh.members[member]
    r = hh.floor_real_rate if real_rate is None else real_rate
    years = max(hh.retire_age - m.age, 0)
    if years == 0 or not m.salary:
        return {"필요액": 0.0, "남은 근로기간": 0, "방법": "생애가치법"}

    q = np.minimum(qx_table[m.sex] * hh.floor_mort_mult, 1.0)
    x = min(m.age, len(q) - 1)
    surv = np.concatenate([[1.0], np.cumprod(1 - q[x:x + years])])[:years]
    v = (1 + r) ** -np.arange(years)
    g = getattr(hh, "wage_growth", 0.0)                     # 실질 임금상승
    income = m.salary * (1 + g) ** np.arange(years)
    pv = float(np.sum(v * surv * income * (1 - self_share)))
    return {"필요액": pv, "남은 근로기간": years, "본인 몫 제외율": self_share,
            "할인율": r, "방법": "생애가치법"}


def compare_methods(hh: Household, cfg: SimConfig, **kw) -> dict:
    """두 방법을 나란히. 설계 현장에서 쓰는 방식 — 큰 쪽을 상한, 작은 쪽을 하한으로 본다."""
    member = kw.pop("member", 0)
    hlv_kw = {k: kw.pop(k) for k in ("self_share", "real_rate") if k in kw}
    needs = needs_analysis(hh, cfg, member=member, **kw)
    hlv = human_life_value(hh, cfg, member=member,
                           e=kw.get("e"), qx_table=kw.get("qx_table"), **hlv_kw)
    lo, hi = sorted((needs["need"][0], hlv["필요액"]))
    return {"니즈분석법": needs["need"][0], "생애가치법": hlv["필요액"],
            "권장 범위": (lo, hi), "나이별 필요액": needs}


def term_life_premium(hh: Household, cfg: SimConfig, qx_table=None, member: int = 0,
                      benefit: float = 10_000.0, term_years: int = 20,
                      loading: float = 0.25, rate: float = 0.03, mort_mult: float = 1.0) -> float:
    """정기보험 연 보험료를 수지상등 원칙으로 산출 — 간병보험과 같은 방식.

    보험료 × 납입현가 × (1 − 사업비율) = 보장액 × 사망급부 현가
    실제 상품은 이보다 비싸다(해지·선택효과·이익마진). 비교용 기준선으로 쓴다.
    """
    qx_table = qx_table or mortality.default_qx(cfg.max_age)
    m = hh.members[member]
    q = np.minimum(qx_table[m.sex] * mort_mult, 1.0)
    x = min(m.age, len(q) - 1)
    qs = q[x:x + term_years]
    qs = np.concatenate([qs, np.ones(max(0, term_years - len(qs)))])
    surv = np.concatenate([[1.0], np.cumprod(1 - qs)[:-1]])   # 각 해 초 생존확률
    v = (1 + rate) ** -np.arange(term_years)
    pv_benefit = float(np.sum(v * surv * qs * benefit))       # 그 해에 사망 → 연말 지급
    pv_premium = float(np.sum(v * surv))                      # 생존 중 납입
    return (1 + loading) * pv_benefit / max(pv_premium, 1e-9)
