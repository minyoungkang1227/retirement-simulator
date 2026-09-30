"""한국 연금 제도 모듈 (v1: 국민연금 조기/연기 조정)."""
from .config import Person, NPSRules


def nps_annual_amount(p: Person, r: NPSRules) -> float:
    """조기/연기 반영한 국민연금 연액(현재가치, 물가연동)."""
    diff = p.nps_start_age - p.nps_normal_age
    if diff < 0:
        diff = max(diff, -r.max_early_years)
        factor = 1 + r.early_cut_per_year * diff
    else:
        diff = min(diff, r.max_defer_years)
        factor = 1 + r.defer_add_per_year * diff
    return p.nps_monthly * 12 * factor


# ── v20 국민연금 예상액 자동 계산 ──
# 2026 연금개혁: 소득대체율 43% (비례상수 1.2 × 43/40 = 1.29), 기준소득월액 상·하한 659만·41만 원(2026.7~)
# A값(전체 가입자 평균소득월액): 2025년 약 309만 원에 최근 변동률 3.4%를 반영한 근사 319만 원
NPS_A = 319.0
NPS_B_CAP, NPS_B_FLOOR = 659.0, 41.0
NPS_C = 1.29


def estimate_nps_monthly(salary_annual: float, years_past: float, age: int, retire_age: int,
                         contrib_end_age: int = 60) -> float:
    """정상수령 기준 국민연금 월액(만원, 오늘 가치) 근사.

    기본연금액(연) = c × (A + B) × (1 + 0.05 × (가입연수 − 20)),  가입 10년 미만은 0.
    B는 현재 월 소득(상·하한 적용)을 오늘 가치로 유지한다고 가정(임금과 A값이 같이 오른다고 봄).
    과거 가입기간의 더 높은 소득대체율, 크레딧, 부양가족연금은 반영하지 않음(보수적).
    """
    if salary_annual <= 0 and years_past <= 0:
        return 0.0
    B = min(max(salary_annual / 12, NPS_B_FLOOR), NPS_B_CAP) if salary_annual > 0 else NPS_B_FLOOR
    years = years_past + max(0, min(retire_age, contrib_end_age) - age)
    if years < 10:
        return 0.0
    annual = NPS_C * (NPS_A + B) * (1 + 0.05 * (min(years, 40) - 20))
    return round(annual / 12, 1)
