"""이 모델이 쓰는 바깥 자료의 **유일한 목록**.

전에는 같은 출처가 README·MODEL·CHANGELOG·앱에 흩어져 있어서(ECOS만 25군데),
자료가 갱신되면 여섯 군데를 고쳐야 했고 실제로 서로 어긋나기도 했다.
이제 출처는 여기 한 곳에만 적고, 앱 화면과 docs/SOURCES.md 는 여기서 만들어 낸다.

항목을 고치는 법: 아래 SOURCES 를 고치고 `python scripts/make_sources_doc.py` 를 돌린다.
(테스트가 docs/SOURCES.md 와 이 파일이 어긋나면 잡아 준다.)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """바깥에서 가져온 자료 한 건.

    as_of   : 이 모델이 반영한 시점 (언제 기준 값인가)
    cycle   : 원자료가 갱신되는 주기
    where   : 어디서 다시 받는가
    how     : 갱신 절차 (스크립트가 있으면 그 이름)
    stale   : 갱신하지 않고 두면 어떤 결과가 생기는가 — 급한지 판단하는 기준
    """
    key: str
    name: str                  # 자료 이름
    org: str                   # 만든 곳
    used_for: str              # 모델의 어디에 쓰이는가
    as_of: str
    cycle: str
    where: str
    how: str
    stale: str


SOURCES = (
    Source(
        key="rate_cpi", name="소비자물가지수·국고채 3년 금리 (월별 2000-01~2026-08)",
        org="한국은행 경제통계시스템(ECOS)", used_for="물가·금리 시나리오(OU 과정)의 장기평균·변동성·회귀속도",
        as_of="2026-08", cycle="매월", where="ecos.bok.or.kr (물가 901Y009, 국고채 3년 721Y001)",
        how="scripts/calibrate_*.py 또는 Colab 노트북에서 ECOS 키로 받아 재추정 → economy_v2.CALIBRATED_2026_08 교체",
        stale="몇 달 늦어도 장기평균은 거의 안 움직인다. 금리 수준이 1%p 넘게 달라진 국면에서는 다시 받는 게 좋다."),
    Source(
        key="kospi", name="KOSPI 월별 수익률 (2000-02~2026-08, 319개월)",
        org="한국거래소 (FinanceDataReader 'KS11')", used_for="주식 변동성·폭락 빈도·금리와의 상관",
        as_of="2026-08", cycle="매월", where="FinanceDataReader 패키지",
        how="노트북의 주식 보정 셀 실행 → economy_v2.CALIBRATED_EQUITY_2026_08 교체",
        stale="변동성 추정은 표본이 길어 안정적이다. 큰 폭락장을 겪은 뒤에는 다시 받는 게 좋다."),
    Source(
        key="erp", name="성숙시장 주식 위험프리미엄 4.2% (사전분포)",
        org="Damodaran (NYU Stern), 2026", used_for="한국 ERP의 베이즈 사전분포(+국가위험 약 0.7%p)",
        as_of="2026", cycle="연 1회(연초)", where="Damodaran 홈페이지의 ERP 자료",
        how="retire_sim/economy_v2.py 의 ERP 사전분포 주석과 erp 값 갱신",
        stale="**이 모델에서 가장 불확실한 값**(1σ 폭 13.7%p). 다만 해마다 조금씩만 움직인다."),
    Source(
        key="house", name="전국주택가격동향조사 (월간, 2013-01~)",
        org="한국부동산원 (ECOS 경유)", used_for="집값의 실질 상승률·변동성",
        as_of="2026-08", cycle="매월", where="ecos.bok.or.kr 또는 한국부동산원",
        how="노트북의 집값 보정 셀 실행",
        stale="2013년 작성기관 변경 이후만 쓴다. 표본이 상승기에 치우쳐 장기 상승률은 보수적으로 낮춰 뒀다."),
    Source(
        key="life_table", name="2024 완전생명표 (0~99세, 100~109세는 Gompertz 외삽)",
        org="통계청", used_for="사망률 — 모든 생존·연금현가 계산의 바탕",
        as_of="2024년 표(2025-12 공표)", cycle="연 1회(12월)",
        where="kosis.kr 생명표", how="retire_sim/data/life_table_2024.csv 교체 후 기대여명 테스트 확인",
        stale="1년 늦어도 기대여명은 0.1년 수준만 달라진다. 코호트 개선율이 이미 미래 연장을 반영한다."),
    Source(
        key="mortality_trend", name="생명표 추이 (2013→2024 기대수명 남 +2.3년, 여 +1.5년)",
        org="통계청", used_for="연령별 사망률 개선율(코호트) — 미국 SOA Scale MP와 같은 역할",
        as_of="2024", cycle="연 1회", where="kosis.kr 생명표",
        how="mortality.improvement_rate 재보정 (scripts/calibrate_mortality.py)",
        stale="개선율을 빼면 대표 가구에서 기본생활 유지 확률이 7~9%p 올라간다 — 빼면 안 되는 요소."),
    Source(
        key="ltc", name="장기요양 통계 (2025 인정자 123.5만 명, 등급 구성), 2022 장기요양실태조사, 2025 고령자 통계",
        org="국민건강보험공단 · 통계청", used_for="간병 발생률·중증 비율(12.7%)·연령별 유병률",
        as_of="2025", cycle="연 1회", where="국민건강보험공단 통계자료실, kosis.kr",
        how="scripts/calibrate_care.py 로 발생률 재보정 → config.CareMarkov.incidence 교체",
        stale="간병은 영향이 두 번째로 큰 요소(대표 가구 15~16%p). 고령화로 유병률이 오르는 추세라 2년에 한 번은 보는 게 좋다."),
    Source(
        key="ltc_price", name="장기요양 수가 인상률 (2022~2026: 4.32/4.70/2.92/3.93/4.4%, 평균 4.05%)",
        org="보건복지부 장기요양위원회", used_for="간병비가 물가보다 빨리 오르는 폭(연 +1.0%p)",
        as_of="2026", cycle="연 1회(연말 고시)", where="보건복지부 보도자료",
        how="최근 5년 평균 − 같은 기간 물가 평균 → config.SimConfig.care_real_growth",
        stale="장기 간병비를 좌우한다. 0으로 두면 30대 가구의 꼬리위험(CTE90)이 절반으로 과소평가된다."),
    Source(
        key="nps", name="국민연금 2026 개혁 (소득대체율 43%, 기준소득월액 상·하한 659만·41만 원, A값 319.35만 원(3,193,511원))",
        org="국민연금공단 · 보건복지부", used_for="국민연금 예상액 추정, 유족연금",
        as_of="2026-07", cycle="A값·상하한은 매년 7월", where="국민연금공단 공시",
        how="retire_sim/pension.py 상단 상수 교체",
        stale="해마다 바뀐다. **앱의 추정치는 근사이고, 정확한 값은 공단 조회**라고 화면에 안내하고 있다."),
    Source(
        key="basic_pension", name="기초연금 2026 (기준연금액 월 34.97만 원, 선정기준액 단독 247만·부부 395.2만 원, 기본재산액 대도시 1.35억·중소도시 8,500만·농어촌 7,250만 원)",
        org="보건복지부", used_for="기초연금 수급 여부와 연계감액(법정 산식)",
        as_of="2026-01", cycle="연 1회(1월)", where="보건복지부 기초연금 안내",
        how="config.NPSRules 의 basic_* 값 교체",
        stale="**영향이 가장 큰 제도**(60대 부부 25.1%p). 매년 1월에 반드시 갱신해야 한다."),
    Source(
        key="hf", name="주택연금 월지급금표 (종신지급 정액형)",
        org="한국주택금융공사", used_for="주택연금 선택지 비교",
        as_of="2026-03-01", cycle="연 1~2회", where="hf.go.kr 예상연금 조회",
        how="retire_sim/tax.py 의 월지급금표 교체 후 해당 테스트 확인",
        stale="집이 자산의 대부분인 가구에서는 선택을 좌우한다. 개정 고시가 나오면 바로 갱신."),
    Source(
        key="tax", name="소득세·상속세 세율표, 건강보험료율 7.19%(장기요양 0.9448%)",
        org="국세청 · 국민건강보험공단", used_for="종합소득세·금융소득종합과세·건보료·상속세",
        as_of="2026-09", cycle="연 1회(세법개정안 후)", where="국세청, 건보공단 공시",
        how="retire_sim/tax.py 의 BRACKETS·ESTATE·요율 교체",
        stale="세금 전체를 빼면 대표 가구에서 8~10%p 움직인다. 세법 개정 때 확인."),
    Source(
        key="hi_property", name="재산보험료부과점수 등급표 60등급, 점수당 금액 211.5원, 재산 기본공제 1억 원",
        org="국민건강보험법 시행령 [별표 4] (개정 2024. 5. 7.) · 시행령 제44조제2항",
        used_for="지역가입자 재산분 건보료 — 자가 보유 가구의 은퇴 후 고정비",
        as_of="2026", cycle="점수당 금액은 연 1회(고시), 등급표는 개정 시",
        where="law.go.kr 「국민건강보험법 시행령」 별표 4 · 공단 보험료 고시",
        how="retire_sim/tax.py 의 HI_PROPERTY_GRADES·HI_POINT_VALUE·HI_PROPERTY_DEDUCT 교체 "
            "→ scripts/measure_hi_property.py 로 손검산 재확인",
        stale="공시 5~14억 1주택 가구에서 기본생활 유지 확률을 6~7%p 낮춘다. 20년 건보료가 "
              "1,500만 원에서 4,100~5,600만 원으로 늘어나므로, 재산분을 빼면 은퇴 후 고정비를 "
              "크게 과소평가한다. 점수당 금액 고시가 바뀌면 바로 갱신."),
    Source(
        key="retirement_tax", name="퇴직소득세 — 근속연수공제(2023년 이후 퇴직분), 환산급여공제, 이연퇴직소득 연금수령 시 70%(1~10년차)·60%(11년차~)",
        org="소득세법 제48조(퇴직소득공제)·제55조(세율)·제146조, 시행령 제40조의2(연금수령한도)",
        used_for="퇴직금(DB)·퇴직연금(DC)을 은퇴 때 IRP로 받아 연금으로 나눠 쓸 때의 세금",
        as_of="2026", cycle="세법 개정 시",
        where="law.go.kr 「소득세법」 제48·55조, 국세청 퇴직소득세 계산 안내",
        how="retire_sim/tax.py 의 retirement_service_deduction·retirement_conv_deduction·RET_INCOME_PENSION_RATIO 교체 "
            "→ tests 의 test_retirement_income_tax_hand_calc_v49 손검산 재확인",
        stale="공제표가 물가연동이 아니라 명목 퇴직금이 커질수록 실효세율이 오른다. 대표 30대 가구에서 퇴직소득세는 "
              "퇴직급여의 3~4%(연금수령 시 그 70%)로 결과를 1%p 안쪽으로 움직인다."),
    Source(
        key="earned_income", name="근로소득공제(총급여 구간 70/40/15/5/2%, 한도 2,000만)·근로소득세액공제(130만 이하 55%, 초과 30%, 한도 74/66/50만)",
        org="소득세법 제47조·제59조",
        used_for="은퇴 후 재취업·부분 근로 소득의 세금",
        as_of="2026", cycle="세법 개정 시",
        where="law.go.kr 「소득세법」 제47·59조",
        how="retire_sim/tax.py 의 earned_income_deduction·earned_income_credit 교체 → test_earned_income_rules_v50 손검산",
        stale="은퇴 후 근로는 대부분 저소득 구간이라 세금 영향은 작다(월 200만 원 근로에서 연 수십만 원)."),
    Source(
        key="nps_work_cut", name="국민연금 재직자 노령연금 감액 기준 — 월 소득 A값 + 200만 원(2026년 약 519만 원) 이하면 감액 없음",
        org="국민연금법 개정(2026. 6. 17. 시행, 2026. 1. 1. 발생 소득부터 적용)",
        used_for="은퇴 후 근로가 있어도 국민연금을 깎지 않는 근거",
        as_of="2026-06", cycle="A값은 매년 7월",
        where="국민연금공단 공지, 보건복지부 보도자료",
        how="기준을 넘는 근로를 모델링하려면 retire_sim/pension.py 의 nps_annual_amount 에 감액 구간을 추가",
        stale="월 519만 원을 넘는 은퇴 후 근로는 드물어 지금은 감액을 넣지 않았다."),
    Source(
        key="hi_jeonse", name="건강보험 지역가입자 재산 — 전월세 보증금(월세는 연 환산 × 40)의 30%를 재산과표에 더함",
        org="국민건강보험법 시행령 [별표 4] 재산보험료부과점수 산정 · 공단 안내",
        used_for="세 들어 사는 가구의 건보료 재산분",
        as_of="2026", cycle="개정 시",
        where="law.go.kr 「국민건강보험법 시행령」 별표 4, 건강보험공단 지역보험료 안내",
        how="retire_sim/tax.py 의 HI_JEONSE_RATIO 교체",
        stale="보증금 5억 가구에서 20년 건보료가 약 1,600만 원 늘어난다. 월세 환산분은 아직 입력받지 않는다."),
    Source(
        key="old_age_spending", name="고령가구 연령별 소비 — 60~64세 대비 균등화 실질소비 65~69세 75%, 70~74세 57%, 75세 이상 43%, 보건만 유지·증가",
        org="자본시장연구원 연구보고서 25-03(2025), 재정패널(NaSTaB) 2008~2021 · Blanchett(2014) JFP, 60~90세 실질지출 연 −0.96%",
        used_for="나이에 따른 지출 변화(spend_profile): 여행·취미 75세부터 감소, 선택 시 기본생활비 연 1% 감소",
        as_of="2025-02", cycle="부정기",
        where="kcmi.re.kr 보고서 1822, Journal of Financial Planning 2014-05",
        how="retire_sim/engine_tax.py 의 spend_profile 배수 교체",
        stale="단면 자료라 '돈이 없어 못 쓰는' 효과가 섞여 있어 그대로 쓰지 않고 보수적으로 줄여 적용했다. "
              "기본값(여행·취미만)은 기본 가구 +3.7%p, 60세 부부 +6.3%p."),
    Source(
        key="early_exit", name="고령층 부가조사 — 주된 일자리를 그만둔 평균 53.0세, 사유: 사업부진·폐업 24.9%, 건강 22.1%, 가족 돌봄 15.2%, 정년퇴직 13.2%, 권고사직·명퇴·정리해고 9.9%",
        org="국가데이터처(통계청), 2026년 5월 경제활동인구조사 고령층(55~79세) 부가조사",
        used_for="은퇴 시점 위험(retire_early_prob): 계획보다 1~7년 일찍 주된 직장을 잃을 확률 — 안정적 10% · 보통 30% · 불안정 50%",
        as_of="2026-08", cycle="매년 8월",
        where="kostat.go.kr 보도자료 '2026년 5월 경제활동인구조사 고령층 부가조사 결과'",
        how="retire_sim/inputs.py 의 EARLY_EXIT 확률과 EARLY_EXIT_YEARS 교체",
        stale="조사는 '그만둔 나이'와 '사유'만 주고 '계획 대비 몇 년 일찍'은 주지 않는다. 세 단계 확률은 판단값이고, "
              "기본 '보통' 30%는 기본 가구 −6.0%p, 45세 부부 −6.1%p(재취업 없을 때)."),
    Source(
        key="kid_cost", name="자녀 1인 양육·교육비 — 대학 졸업까지 3억 896만 원 (영아 3,064 · 유아 3,696 · 초등 7,596 · 중 4,122 · 고 4,719 · 대학 7,709만 원, 2012년 가격)",
        org="한국보건사회연구원, 2012년 전국 출산력 및 가족보건·복지실태조사",
        used_for="자녀 나이별 연간 양육·교육비표 KID_COST(0~21세) — 2012→2026 소비자물가 ×1.27 환산, 1인 약 3.9억 원",
        as_of="2012", cycle="부정기(같은 조사의 양육비 문항은 2012년이 마지막)",
        where="보사연 연구보고서 2012-54; 머니투데이 2016-06-25 보도 등에 단계별 금액",
        how="retire_sim/inputs.py 의 KID_COST 교체(새 조사가 나오면 단계별 금액을 그 해 물가로)",
        stale="2024년 YuWa 인구연구소 추정(한국 1인당 GDP의 7.79배 ≈ 3.6억, 18세까지)과 같은 크기다. 식비·주거 몫이 든 총액이라 "
              "지금 있는 자녀는 지금 수준과의 차이만 넣고, 저축이 0이 될 때까지만 줄인다(CHANGELOG v52). 교육비 비중은 영아 12% → 대학 45%."),
    Source(
        key="widowhood", name="사별 후 사망위험 메타분석 (남 RR 1.22 / 여 1.03, 6개월 이내 1.41)",
        org="Moon et al. (2011) PLOS ONE — 226만 명, 전향 코호트 15편, 연령·성별 보정",
        used_for="배우자 사별 직후 생존자의 사망위험 상승",
        as_of="2011", cycle="갱신 거의 없음",
        where="PLOS ONE 2011;6(8):e23465 (제목으로 검색)",
        how="새 메타분석이 나오면 config.SimConfig.widow_excess 교체",
        stale="영향이 작다(0.5~0.9%p). 국내 혼인상태별 사망률(사별/유배우 4.2배)은 연령 보정이 없어 쓰지 않았다."),
    Source(
        key="insurance_cost", name="연금보험 사업비 (월납 계약체결 5.6~6.9%, 계약관리 4.2~9%, 해지공제 1년차 25.6%)",
        org="보험연구원 「변액연금 최저보증 및 사업비 부과 현황 조사」",
        used_for="연금보험 비교의 기본값과 '상품 선택의 폭'(가장 싼 상품 ↔ 가장 비싼 상품)",
        as_of="2026-10 확인", cycle="부정기", where="보험연구원 보고서",
        how="retire_sim/product.py 의 기본값과 COST_RANGE 교체",
        stale="상품 선택의 폭은 6% 수준으로 작다. 늦게 갱신해도 결론(개시 나이가 더 크다)은 바뀌지 않는다."),
    Source(
        key="credited_rate", name="생명보험사 공시이율 (2025.12 적용 일반연금 2.31%·연금저축 2.28%)",
        org="생명보험협회 공시실", used_for="연금보험 비교의 공시이율 기본값",
        as_of="2025-12", cycle="매월", where="pub.insure.or.kr 상품비교공시",
        how="앱의 '공시이율' 기본값과 product.py 주석 교체",
        stale="±0.5%p면 적립금이 ±4% 달라진다. 화면에서 사용자가 직접 바꿀 수 있어 급하지 않다."),
)

BY_KEY = {s.key: s for s in SOURCES}


def as_rows():
    """화면·문서에서 표로 뿌리기 좋은 형태."""
    return [{"자료": s.name, "만든 곳": s.org, "쓰이는 곳": s.used_for,
             "반영 시점": s.as_of, "갱신 주기": s.cycle} for s in SOURCES]


def refresh_order():
    """갱신이 급한 순서. '갱신 안 하면 생기는 일'에 굵게 표시해 둔 것을 앞으로."""
    return sorted(SOURCES, key=lambda s: (0 if "**" in s.stale else 1, s.name))
