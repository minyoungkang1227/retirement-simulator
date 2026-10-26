"""한국 세금·건보료 모듈 (2026-09 기준, 단순화 버전 — 세무 자문 아님).

반영: 종합소득세(공적연금 + 금융소득종합과세 비교과세), 사적연금 분리과세(1,500만원),
      지역가입자 건강보험료(소득분 + 재산분 법정 점수표), 상속세(2차 상속), ISA·연금계좌 절세 전략.
미반영: 배당세액공제(Gross-up), 고배당기업 분리과세 특례, 전월세보증금·자동차 부과점수, 해외주식 양도세,
        연금계좌 세액공제(보수적으로 미적용), 기초연금, 주택·부동산.
금액 단위: 만원(명목). 세법 기준금액은 물가연동하지 않음(실제 법과 동일).
"""
import numpy as np
from dataclasses import dataclass

# 종합소득세 누진세율 (과세표준 상한, 세율, 누진공제)
BRACKETS = [(1400, .06, 0), (5000, .15, 126), (8800, .24, 576), (15000, .35, 1544),
            (30000, .38, 1994), (50000, .40, 2594), (100000, .42, 3594), (np.inf, .45, 6594)]
ESTATE = [(10000, .10, 0), (50000, .20, 1000), (100000, .30, 6000), (300000, .40, 16000), (np.inf, .50, 46000)]


def _table_arrays(table):
    """(상한, 세율, 누진공제) 표 → searchsorted용 배열. 표 객체(id)별로 한 번만 만든다."""
    key = id(table)
    hit = _TABLE_CACHE.get(key)
    if hit is None or hit[0] is not table:
        his = np.array([t[0] for t in table], float)
        rs = np.array([t[1] for t in table], float)
        deds = np.array([t[2] for t in table], float)
        hit = (table, his, rs, deds); _TABLE_CACHE[key] = hit
    return hit[1], hit[2], hit[3]


_TABLE_CACHE: dict = {}


def progressive(base, table):
    """누진세: 구간 (lo, hi] 안에서 base × 세율 − 누진공제. base ≤ 0이면 0.

    (v47) 구간마다 np.where를 도는 대신 상한 배열에 searchsorted(side='left')를 써서
    구간 인덱스를 한 번에 찾는다. side='left'는 lo < base ≤ hi 와 같은 우닫힌 구간이라 결과가 동일하다."""
    base = np.maximum(np.asarray(base, float), 0)
    his, rs, deds = _table_arrays(table)
    idx = np.minimum(np.searchsorted(his, base, side="left"), len(his) - 1)
    out = base * rs[idx] - deds[idx]
    return np.where(base > 0, out, 0.0)


def pension_deduction(x):
    """연금소득공제 (한도 900만원)."""
    x = np.maximum(x, 0)
    d = np.where(x <= 350, x, np.where(x <= 700, 350 + .4 * (x - 350),
         np.where(x <= 1400, 490 + .2 * (x - 700), 630 + .1 * (x - 1400))))
    return np.minimum(d, 900)


def private_pension_rate(age):
    return np.where(age >= 80, .033, np.where(age >= 70, .044, .055))


@dataclass
class TaxConfig:
    enabled: bool = True
    nps_taxable_ratio: float = 0.75   # 국민연금 중 과세 대상 비율(2002년 이후 납입분) — 공단 원천징수 내역으로 확인
    div_yield: float = 0.02           # 주식 배당수익률(수익률 안에 포함, 과세계좌에서 과세)
    dependent_possible: bool = False  # 직장가입자 자녀 등의 피부양자 등록 가능 여부
    ownership: tuple = None           # 금융자산 명의 비율, None이면 균등
    overseas_share: float = 0.0       # 주식 중 해외주식 비율(양도차익 250만원 공제 후 22%)
    n_children: int = 2
    gift_per_child_10y: float = 0     # 전략: 10년마다 자녀 1인당 증여액(만원, 현재가치). 5,000까지 증여세 없음
    use_isa: bool = False             # 전략: 과세계좌 → ISA (1인 연 2,000, 총 1억)
    use_pension: bool = False         # 전략: 과세계좌 → 연금저축 (1인 연 1,800), 55세·가입 5년 후 연 1,500 이내 인출
    isa_annual: float = 2000; isa_total: float = 10000; isa_free_per_year: float = 200 / 3
    pen_annual: float = 1800; pen_withdraw_cap: float = 1500; pen_wait_years: int = 5
    hi_rate: float = 0.0719; ltc_ratio: float = 0.9448 / 7.19


def income_tax_person(nps, fin, age, tc: TaxConfig, extra=0):
    """공적연금 + 금융소득(비교과세) + 기타 종합소득(임대 등) 종합소득세, 지방세 포함."""
    other = np.maximum(nps * tc.nps_taxable_ratio - pension_deduction(nps * tc.nps_taxable_ratio), 0) + extra
    base = np.maximum(other - 150 - np.where(age >= 70, 100, 0), 0)
    sep = progressive(base, BRACKETS) + fin * .14
    comb = progressive(base + np.maximum(fin - 2000, 0), BRACKETS) + np.minimum(fin, 2000) * .14
    return np.where(fin > 2000, np.maximum(sep, comb), sep) * 1.1


# ── (v49) 퇴직소득세 (소득세법 제48·55조, 근속연수공제는 2023년 이후 퇴직분 기준) ──
#   ① 근속연수공제  ② 환산급여 = (퇴직소득 − 근속연수공제) × 12 / 근속연수  ③ 환산급여공제
#   ④ 과세표준 = 환산급여 − 환산급여공제  ⑤ 산출세액 = 기본세율(BRACKETS)(과세표준) × 근속연수 / 12
#   ⑥ 지방소득세 10%. 금액 단위 만원.
#   이연퇴직소득(IRP로 받아 연금으로 수령)은 연금수령 1~10년차 산출세액의 70%, 11년차부터 60%를 낸다.
RET_INCOME_PENSION_RATIO = (0.7, 0.6)


def retirement_service_deduction(years):
    """근속연수공제(만원). 근속연수는 1년 미만 절상."""
    n = np.maximum(np.ceil(np.asarray(years, float)), 1)
    return np.where(n <= 5, 100 * n,
           np.where(n <= 10, 500 + 200 * (n - 5),
           np.where(n <= 20, 1500 + 250 * (n - 10), 4000 + 300 * (n - 20))))


def retirement_conv_deduction(conv):
    """환산급여공제(만원)."""
    c = np.maximum(np.asarray(conv, float), 0)
    return np.where(c <= 800, c,
           np.where(c <= 7000, 800 + (c - 800) * .60,
           np.where(c <= 10000, 4520 + (c - 7000) * .55,
           np.where(c <= 30000, 6170 + (c - 10000) * .45, 15170 + (c - 30000) * .35))))


def retirement_income_tax(amount, years, local: bool = True):
    """퇴직소득세(만원). amount = 퇴직급여 총액, years = 근속연수. local=True면 지방소득세 10% 포함."""
    R = np.maximum(np.asarray(amount, float), 0)
    n = np.maximum(np.ceil(np.asarray(years, float)), 1)
    conv = np.maximum(R - retirement_service_deduction(n), 0) * 12 / n
    base = np.maximum(conv - retirement_conv_deduction(conv), 0)
    tax = progressive(base, BRACKETS) * n / 12
    return np.where(R > 0, tax * (1.1 if local else 1.0), 0.0)


def private_pension_tax(taxable, age):
    return np.where(taxable <= 1500, taxable * private_pension_rate(age), taxable * .165)


def health_premium(nps_list, fin_list, tc: TaxConfig, rent_list=None, prop_base_list=None, prop_monthly=0.0):
    """지역가입자 건보료(가구): 소득분(공적연금 50%, 금융소득 1,000만원 초과 시 전액, 임대소득금액 100%)
    + 재산분(별표4 점수표 자동 계산 또는 공단 모의계산값). 피부양자: 소득 2,000만원·재산과표 5.4억/9억·과세 임대소득 기준."""
    z = np.zeros_like(nps_list[0])
    rent_list = rent_list or [z] * len(nps_list); prop_base_list = prop_base_list or [z] * len(nps_list)
    inc_fin = [np.where(f > 1000, f, 0) for f in fin_list]
    if tc.dependent_possible:
        ok = np.ones_like(z, dtype=bool)
        for n, f, r, pb in zip(nps_list, inc_fin, rent_list, prop_base_list):
            inc = n + f + r * .5
            ok &= (inc <= 2000) & (r <= 0) & (pb <= 90000) & ~((pb > 54000) & (inc > 1000))
    else:
        ok = np.zeros_like(z, dtype=bool)
    base = sum(.5 * n + f + r * .5 for n, f, r in zip(nps_list, inc_fin, rent_list))
    return np.where(ok, 0, base * tc.hi_rate * (1 + tc.ltc_ratio) + prop_monthly * 12)


# ───────────── v45: 지역가입자 재산분 건보료 (법정 산식) ─────────────
# 국민건강보험법 시행령 [별표 4] 「재산보험료부과점수의 산정방법」(개정 2024. 5. 7.)
# (재산등급 상한(만원, 기본공제 차감 후), 부과점수) — 60등급. 상한 이하이면 그 등급.
HI_PROPERTY_GRADES = (
    (450, 22), (900, 44), (1350, 66), (1800, 97), (2250, 122), (2700, 146), (3150, 171),
    (3600, 195), (4050, 219), (4500, 244), (5020, 268), (5590, 294), (6220, 320), (6930, 344),
    (7710, 365), (8590, 386), (9570, 412), (10700, 439), (11900, 465), (13300, 490),
    (14800, 516), (16400, 535), (18300, 559), (20400, 586), (22700, 611), (25300, 637),
    (28100, 659), (31300, 681), (34900, 706), (38800, 731), (43200, 757), (48100, 785),
    (53600, 812), (59700, 841), (66500, 881), (74000, 921), (82400, 961), (91800, 1001),
    (103000, 1041), (114000, 1091), (127000, 1141), (142000, 1191), (158000, 1241),
    (176000, 1291), (196000, 1341), (218000, 1391), (242000, 1451), (270000, 1511),
    (300000, 1571), (330000, 1641), (363000, 1711), (399300, 1781), (439230, 1851),
    (483153, 1921), (531468, 1991), (584615, 2061), (643077, 2131), (707385, 2201),
    (778124, 2271), (np.inf, 2341),
)
_HI_GRADE_HI = np.array([g[0] for g in HI_PROPERTY_GRADES], float)   # 등급 상한 (searchsorted용)
_HI_GRADE_PTS = np.array([g[1] for g in HI_PROPERTY_GRADES], float)  # 등급별 부과점수
HI_POINT_VALUE = 211.5        # 재산보험료부과점수당 금액(원) — 시행령 제44조제2항, 2026년 고시
HI_PROPERTY_DEDUCT = 10000    # 재산 기본공제액(만원) — 1억원 (2024년 도입)


def hi_property_points(taxable_base, deduct: float = HI_PROPERTY_DEDUCT):
    """재산세 과세표준(만원, 명목) → 재산보험료부과점수. 배열 지원.

    기본공제를 뺀 뒤 별표 4 등급표를 적용한다. 공제 후 금액이 0 이하이면 부과점수 0
    (= 재산분 없음). 공단 실무도 재산과표가 기본공제 이하면 재산보험료를 부과하지 않는다.
    전월세보증금(30% 산입)·자동차는 반영하지 않는다 — 자가 보유 가구를 전제로 한 보수적 근사.
    """
    net = np.maximum(np.asarray(taxable_base, float) - deduct, 0.0)
    # (v47) 등급 60개를 np.where로 훑는 대신 상한 배열에 searchsorted. (lo, hi] 우닫힌 구간이라 결과 동일.
    idx = np.minimum(np.searchsorted(_HI_GRADE_HI, net, side="left"), len(_HI_GRADE_HI) - 1)
    return np.where(net > 0, _HI_GRADE_PTS[idx], 0.0)


def hi_property_monthly(taxable_base, tc: TaxConfig, deduct: float = HI_PROPERTY_DEDUCT,
                        point_value: float = HI_POINT_VALUE):
    """재산분 건보료(월, 만원, 명목) — 장기요양보험료 포함. 배열 지원.

    점수 × 점수당 금액(원) ÷ 10,000 → 만원. 장기요양은 건강보험료에 비례하므로 같은 비율을 쓴다.
    기본공제액·등급표·점수당 금액은 모두 명목 기준이고 물가연동되지 않는다(실제 제도와 동일).
    그래서 집값이 명목으로 오르면 등급이 올라가 재산분이 저절로 늘어난다.
    """
    pts = hi_property_points(taxable_base, deduct)
    return pts * point_value / 10000.0 * (1 + tc.ltc_ratio)
# ───────────── v7: 부동산·임대·해외주식·증여 (근사, 2026-09 기준) ─────────────
PROP_TAX = [(6000, .001, 0), (15000, .0015, 3), (30000, .0025, 18), (np.inf, .004, 63)]          # 주택 재산세(일반)
PROP_TAX_1H = [(6000, .0005, 0), (15000, .001, 3), (30000, .002, 18), (np.inf, .0035, 63)]       # 1주택 특례(공시 9억 이하)
CJS_2 = [(30000, .005, 0), (60000, .007, 60), (120000, .010, 240), (250000, .013, 600),
         (500000, .015, 1100), (940000, .020, 3600), (np.inf, .027, 10180)]                        # 종부세 2주택 이하
CJS_3 = [(30000, .005, 0), (60000, .007, 60), (120000, .010, 240), (250000, .020, 1080),
         (500000, .030, 3580), (940000, .040, 8580), (np.inf, .050, 17980)]                        # 종부세 3주택 이상


@dataclass
class HouseConfig:
    official: float = 0          # 보유 주택 공시가격 합계(만원)
    market: float = 0            # 시세 합계(만원) — 상속재산 평가용
    n_houses: int = 1
    owner_share: tuple = None    # 주택 명의 비율(구성원 순서), None이면 본인 100%
    years_held: int = 10         # 보유 기간(종부세 장기보유공제)
    hi_property_monthly: float = None  # 건보료 재산분(월, 만원). None이면 별표4 점수표로 자동 계산(v45)
    rent_annual: float = 0       # 연 임대수입(만원)
    fmv_ratio_prop: float = .60  # 재산세 공정시장가액비율(1주택 특례는 43~45%) — 근사
    fmv_ratio_cjs: float = .60   # 종부세 공정시장가액비율
    # ── v19 주택 활용 전략 ──
    reverse_mortgage_age: int = None   # 주택연금 가입 나이(부부 중 나이 적은 사람 기준). None이면 안 함
    downsize_age: int = None           # 집 줄이기(매도 후 작은 집 매수) 나이(본인 기준). None이면 안 함
    downsize_ratio: float = 0.5        # 새 집 가격 / 기존 집 가격
    cost_basis: float = None           # 기존 집 취득가(만원). None이면 현재 시세의 50%로 가정
    sell_cost: float = 0.006           # 매도 중개수수료 등


# 주택연금 월지급금 (한국주택금융공사, 2026-03-01, 종신지급방식 정액형, 만원/월)
# 나이: (주택가격 1억원당 월지급금, 12억원 주택 월지급금 = 연령별 상한)
RM_TABLE = {55: (15.6, 187.2), 60: (21.0, 252.8), 65: (25.2, 303.5), 70: (30.7, 341.4), 75: (38.1, 366.6), 80: (48.3, 406.0)}


def reverse_mortgage_monthly(age: float, house_value):
    """연소자 나이·주택가격(만원, 시세)으로 월지급금(만원). house_value는 스칼라/배열 모두 가능.
    표 사이 나이는 선형 보간, 12억 초과는 12억으로 계산."""
    ages = sorted(RM_TABLE); age = min(max(age, ages[0]), ages[-1])
    lo = max(a for a in ages if a <= age); hi = min(a for a in ages if a >= age)
    w = 0 if hi == lo else (age - lo) / (hi - lo)
    rate = RM_TABLE[lo][0] * (1 - w) + RM_TABLE[hi][0] * w
    cap = RM_TABLE[lo][1] * (1 - w) + RM_TABLE[hi][1] * w
    return np.minimum(rate * np.minimum(house_value, 120000) / 10000, cap)


def acquisition_tax_rate(price):
    """1주택 취득세율(지방교육세 포함) 근사: 6억 이하 1.1%, 9억 이하 2.2%, 초과 3.3%. 배열 지원."""
    return np.where(np.asarray(price) <= 60000, 0.011, np.where(np.asarray(price) <= 90000, 0.022, 0.033))


def one_house_cgt(price, basis: float, years: int):
    """1세대 1주택 양도세 근사: 12억 초과분 비율만 과세, 장기보유특별공제 최대 80%, 실효세율 38.5%. 배열 지원."""
    price = np.asarray(price, float)
    gain = np.maximum(price - basis, 0) * np.maximum(price - 120000, 0) / np.maximum(price, 1e-9)
    return gain * (1 - min(0.08 * years, 0.8)) * 0.385


def property_tax(official, n_houses, fmv):
    base = official * (0.45 if n_houses == 1 else fmv)
    tbl = PROP_TAX_1H if (n_houses == 1 and np.all(official <= 90000)) else PROP_TAX
    t = progressive(base, tbl)
    return t * 1.2 + base * .0014                      # 지방교육세 20% + 도시지역분 0.14%


def comprehensive_property_tax(official, n_houses, age, years_held, fmv=.60):
    ded = 120000 if n_houses == 1 else 90000
    base = np.maximum(official - ded, 0) * fmv
    t = progressive(base, CJS_2 if n_houses <= 2 else CJS_3)
    if n_houses == 1:                                    # 1주택 고령자·장기보유 세액공제(합산 80% 한도)
        a = 0.4 if age >= 70 else 0.3 if age >= 65 else 0.2 if age >= 60 else 0
        h = 0.5 if years_held >= 15 else 0.4 if years_held >= 10 else 0.2 if years_held >= 5 else 0
        t = t * (1 - min(a + h, 0.8))
    return t * 1.2                                       # 농어촌특별세 20%


def rent_tax(rent_person, other_income_person):
    """주택임대소득: 2,000만원 이하 분리과세(필요경비 50%, 기본공제 200만원) 근사. 초과분은 종합과세용 소득금액 반환."""
    sep = np.maximum(rent_person * .5 - np.where(other_income_person <= 2000, 200, 0), 0) * .14 * 1.1
    return np.where(rent_person <= 2000, sep, 0), np.where(rent_person > 2000, rent_person * .5, 0)


def gift_tax(amount_over_deduction):
    return progressive(amount_over_deduction, ESTATE)
