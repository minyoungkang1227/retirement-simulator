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
