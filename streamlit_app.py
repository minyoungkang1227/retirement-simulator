"""노후나침반 — 목표 기반 확률 은퇴자금 진단 (Streamlit 웹 앱, v12)

흐름: 단계별 입력(Task) → 최종 확인 → 목표별 달성 결과 → '무엇을 바꾸면' 비교
실행: streamlit run streamlit_app.py
입력값은 계산에만 쓰이며 서버에 저장하지 않습니다(계산 결과는 10분간 메모리 캐시).
"""
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from retire_sim.config import Person, Household, SimConfig, CareMarkov, CareShock, AddOns
from retire_sim.economy_v2 import EconomyV2
from retire_sim import engine_tax, mortality
from retire_sim.tax import TaxConfig, HouseConfig

st.set_page_config(page_title="노후나침반 · 은퇴 목표 진단", page_icon="🧭", layout="centered")
TEAL, CORAL, INK, MUTED = "#1F6F78", "#C8553D", "#1C2B33", "#5B6B73"
PROFILES = {"보수형": 0.3, "균형형": 0.5, "성장형": 0.7}
st.markdown(f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
html, body, [class*="css"], .stMarkdown, button, input, label {{ font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif; }}
.hero {{ font-size: 1.8rem; line-height: 1.35; font-weight: 700; color: {INK}; margin: .3rem 0 .5rem; }}
.hero .n {{ color: {CORAL}; }}
.sub {{ color: {MUTED}; font-size: 1rem; }}
.gap {{ background: #E6EDEF; border-radius: 14px; padding: 1rem 1.2rem; margin: .4rem 0 1rem; }}
.gap .big {{ font-size: 2rem; font-weight: 700; color: {TEAL}; }}
.gap .eq {{ color: {MUTED}; font-size: .95rem; }}
.dots {{ display: flex; gap: 10px; margin: .8rem 0 .4rem; flex-wrap: wrap; }}
.dot {{ width: 32px; height: 32px; border-radius: 50%; }}
.dot.bad {{ background: {CORAL}; }} .dot.ok {{ background: {TEAL}; opacity: .85; }}
.legend, .note {{ color: {MUTED}; font-size: .88rem; line-height: 1.6; }}
.card {{ border: 1px solid #CBD7DB; border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; background: white; }}
.card h4 {{ margin: 0 0 .3rem; font-size: 1rem; color: {TEAL}; }}
.card p {{ margin: 0; line-height: 1.6; }}
</style>""", unsafe_allow_html=True)

ss = st.session_state
if "p" not in ss:
    ss.p = dict(age=52, sex="M", retire_age=60, spouse=True, s_age=50, s_sex="F",
                deposit=1.5, invest=2.0, ret_acct=1.5, other=0.0, saving=150,
                essential=300, lifestyle=1200, nps=110, s_nps=50, nps_start=65,
                priv=0, priv_start=60, priv_years=20, legacy=1.0, children=2, profile="균형형",
                h_official=0.0, h_market=0.0, h_n=1, h_joint=False, h_years=10, h_hi=0.0, rent=0,
                biz=0, biz_until=65, biz_work=False, overseas=20, own_equal=True, survivor=70,
                enhanced=True, paths=10000)
    ss.step = 0
p = ss.p


# ─────────────────────────── 계산 ───────────────────────────
def total_assets(p):
    return p["deposit"] + p["invest"] + p["ret_acct"] + p["other"]


def build(p: dict, v: dict):
    nps_start = v.get("nps_start", p["nps_start"])
    retire = p["retire_age"] + v.get("retire_delta", 0)
    mk = lambda age, sex, nps, priv=0, ps=0, py=0: Person(
        age=age, sex=sex, nps_monthly=nps, nps_start_age=nps_start,
        private_pension_annual=priv, private_pension_start=ps, private_pension_years=py)
    members = [mk(p["age"], p["sex"], p["nps"], p["priv"], p["priv_start"], p["priv_years"] if p["priv"] > 0 else 0)]
    if p["spouse"]:
        members.append(mk(p["s_age"], p["s_sex"], p["s_nps"]))
    m = v.get("spend_mult", 1.0)
    hh = Household(members=members, liquid_assets=v.get("assets", total_assets(p)) * 10000,
                   stock_weight=v.get("stock_weight", PROFILES[p["profile"]]),
                   annual_spending=p["essential"] * 12 * m, essential=p["essential"] * 12 * m,
                   lifestyle=p["lifestyle"] * m, legacy_target=p["legacy"] * 10000,
                   retire_age=retire if p["age"] < retire else None,
                   annual_saving=(p["saving"] + v.get("saving_delta", 0)) * 12,
                   survivor_spending_ratio=p["survivor"] / 100,
                   floor_method="fixed95" if p.get("floor") == "보수적" else "actuarial")
    care = CareMarkov() if p["enhanced"] else CareShock()
    if v.get("care_mult"):
        care = CareMarkov(incidence=tuple((a, x * v["care_mult"]) for a, x in CareMarkov().incidence)) \
            if p["enhanced"] else CareShock(lam=0.03 * v["care_mult"])
    cfg = SimConfig(n_paths=v.get("paths", p["paths"]), seed=42, care=care)
    ekw = dict(v.get("eco", {}))
    for key in ("stock_shocks", "infl_shocks"):
        if key in ekw: ekw[key] = {int(t): x for t, x in ekw[key].items()}
    eco = EconomyV2.enhanced(**ekw) if p["enhanced"] else EconomyV2(**ekw)
    tkw = dict(overseas_share=p["overseas"] / 100, n_children=p["children"],
               ownership=None if (p["own_equal"] or not p["spouse"]) else (1, 0))
    tkw.update(v.get("tax", {}))
    k = len(members)
    house = HouseConfig(official=p["h_official"] * 10000, market=p["h_market"] * 10000, n_houses=p["h_n"],
                        owner_share=(tuple([1 / k] * k) if p["h_joint"] else None), years_held=p["h_years"],
                        hi_property_monthly=p["h_hi"] or None, rent_annual=p["rent"])
    biz = [dict(income=p["biz"], until_age=p["biz_until"], workplace=p["biz_work"])] if p["biz"] > 0 else [None]
    biz += [None] * (k - 1)
    q = None
    if v.get("q_mult"):
        q = {s: np.minimum(a * v["q_mult"], 1) for s, a in mortality.default_qx().items()}
        for a in q.values(): a[-1] = 1
    return hh, cfg, eco, TaxConfig(**tkw), house, biz, q, AddOns(**v.get("addons", {}))


@st.cache_data(ttl=600, max_entries=128, show_spinner=False)
def simulate(p_json: str, v_json: str) -> dict:
    p, v = json.loads(p_json), json.loads(v_json)
    hh, cfg, eco, tc, house, biz, q, ad = build(p, v)
    res = engine_tax.run(hh, cfg, e=eco, tc=tc, house=house, qx_table=q, biz=biz, addons=ad)
    s = engine_tax.summarize(res)
    d = res["depleted_at"]; y = res["youngest_age"]; ages = np.arange(res["T"] + 1) + y
    W, alive = res["W_real"], res["hh_alive"]
    fan = [[ages[t], *np.percentile(W[alive[:, t], t], [5, 25, 50, 75, 95])]
           for t in range(W.shape[1]) if alive[:, t].sum() > 200]
    curve = [(int(ages[t]), float(((d >= 0) & (d <= t)).mean())) for t in range(res["T"] + 1)]
    t85 = max(0, min(85 - y, res["T"]))
    w85 = W[alive[:, t85], t85]
    corr = None
    if np.any(~np.isnan(res["stock_ret"])):
        sr, pr = res["stock_ret"].ravel(), res["port_ret"].ravel(); ok = ~np.isnan(sr)
        corr = float(np.corrcoef(sr[ok], pr[ok])[0, 1])
    return {"s": s, "dep": (d >= 0), "anti": bool(res.get("antithetic")),
            "dep_age": float(np.median(d[d >= 0] + y)) if (d >= 0).any() else None, "fan": fan, "curve": curve,
            "p50_85": float(np.percentile(w85, 50)) if w85.size else 0.0, "annuity_pay": res.get("annuity_pay", 0.0),
            "care_premium": res.get("care_premium", 0.0), "stock_corr": corr}


@st.cache_data(ttl=86400, max_entries=64, show_spinner=False)
def stock_lookup(code: str):
    from retire_sim.market_data import stock_stats
    try:
        key = st.secrets.get("ECOS_AUTH_KEY", None)
    except Exception:
        key = None
    return stock_stats(code, years=10, ecos_key=key)


def run(p, v=None):
    return simulate(json.dumps(p, sort_keys=True), json.dumps(v or {}, sort_keys=True))


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def required_assets(p_json: str, target: float = 0.9):
    """기본생활 유지 확률이 target이 되는 금융자산(억 원) — 이분법, 같은 시나리오 사용."""
    p = json.loads(p_json); paths = min(p["paths"], 5000)
    ok = lambda a: run(p, {"assets": a, "paths": paths})["s"]["기본생활 유지 확률"] >= target
    lo, hi = 0.0, max(total_assets(p) * 3, 20.0)
    if not ok(hi): return None, hi
    for _ in range(10):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi, None


def paired(a, b):
    x = a["dep"].astype(float) - b["dep"].astype(float); n = len(x)
    if a["anti"] and b["anti"]:
        h = n // 2; x = (x[:h] + x[h:2 * h]) / 2; n = h
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(n)


def pct(x): return f"{x * 100:.0f}%"


def income_gap(p):
    need = p["essential"] + p["lifestyle"] / 12
    pension = p["nps"] + (p["s_nps"] if p["spouse"] else 0) + p["priv"] / 12
    return need, pension, need - pension


# ─────────────────────────── 단계별 입력 ───────────────────────────
STEPS = ["은퇴 시점", "현재 자산", "기본생활", "여행·취미", "연금·소득", "남길 자산", "투자 성향", "추가 정보", "최종 확인"]


def nav(i, form_key):
    c1, c2 = st.columns(2)
    back = c1.form_submit_button("이전", disabled=(i == 0), width="stretch")
    nxt = c2.form_submit_button("다음", type="primary", width="stretch")
    return back, nxt


def go(delta):
    ss.step = max(0, min(len(STEPS) - 1, ss.step + delta)); st.rerun()


st.markdown("### 🧭 노후나침반")
if ss.step < len(STEPS) and not ss.get("done"):
    st.markdown(f'<p class="sub">내가 원하는 은퇴생활을 지키려면 무엇이 필요한지, 1만 가지 미래로 확인합니다. '
                f'단계 {ss.step + 1}/{len(STEPS)} · {STEPS[ss.step]}</p>', unsafe_allow_html=True)
    st.progress((ss.step + 1) / len(STEPS))

    with st.sidebar:
        st.markdown("**지금까지 확인한 내용**")
        if ss.step > 0: st.caption(f"은퇴: 현재 {p['age']}세 → {p['retire_age']}세 은퇴")
        if ss.step > 1: st.caption(f"금융자산: {total_assets(p):.1f}억 원" + (f", 은퇴 전 월 {p['saving']}만 원 저축" if p['age'] < p['retire_age'] else ""))
        if ss.step > 2: st.caption(f"기본생활비: 월 {p['essential']}만 원")
        if ss.step > 3: st.caption(f"여행·취미: 연 {p['lifestyle']:,}만 원")
        if ss.step > 4: st.caption(f"연금: 월 {income_gap(p)[1]:.0f}만 원")
        if ss.step > 5: st.caption(f"남길 자산: {p['legacy']:.1f}억 원")
        if ss.step > 6: st.caption(f"투자 성향: {p['profile']}")

    i = ss.step
    with st.form(f"step{i}"):
        if i == 0:
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("현재 나이", 30, 90, p["age"]); sex = c2.radio("성별", ["남", "여"], index=0 if p["sex"] == "M" else 1, horizontal=True)
            retire = c3.number_input("은퇴 (예정) 나이", 40, 90, p["retire_age"], help="이미 은퇴했다면 현재 나이 이하로 입력")
            spouse = st.checkbox("배우자와 함께", value=p["spouse"])
            c1, c2 = st.columns(2)
            s_age = c1.number_input("배우자 나이", 30, 90, p["s_age"]); s_sex = c2.radio("배우자 성별", ["남", "여"], index=0 if p["s_sex"] == "M" else 1, horizontal=True)
            st.caption("계획은 95세까지를 기준으로 기본생활비를 보호합니다. 실제 수명은 생명표로 확률적으로 계산합니다.")
            back, nxt = nav(i, "s0")
            if nxt or back:
                p.update(age=int(age), sex="M" if sex == "남" else "F", retire_age=int(retire), spouse=bool(spouse),
                         s_age=int(s_age), s_sex="M" if s_sex == "남" else "F")
                go(1 if nxt else -1)
        elif i == 1:
            st.markdown("집은 빼고, 꺼내 쓸 수 있는 금융자산만 적어주세요. (단위: 억 원)")
            c1, c2 = st.columns(2)
            dep = c1.number_input("예금·적금", 0.0, 500.0, p["deposit"], step=0.1)
            inv = c2.number_input("주식·펀드·ETF", 0.0, 500.0, p["invest"], step=0.1)
            c1, c2 = st.columns(2)
            ret = c1.number_input("퇴직연금·연금저축 적립금", 0.0, 500.0, p["ret_acct"], step=0.1)
            oth = c2.number_input("기타 금융자산", 0.0, 500.0, p["other"], step=0.1)
            sav = st.number_input("은퇴 전까지 한 달 저축·투자액 (만 원)", 0, 5000, p["saving"], step=10,
                                  help="이미 은퇴했다면 0", disabled=p["age"] >= p["retire_age"])
            back, nxt = nav(i, "s1")
            if nxt or back:
                p.update(deposit=dep, invest=inv, ret_acct=ret, other=oth, saving=int(sav) if p["age"] < p["retire_age"] else 0)
                go(1 if nxt else -1)
        elif i == 2:
            st.markdown("**꼭 필요한 기본생활비**는 얼마인가요? 식비·주거·의료·보험료 등 줄이기 어려운 지출입니다.")
            ess = st.number_input("월 기본생활비 (만 원, 오늘 가치)", 0, 5000, p["essential"], step=10,
                                  help="재산세·건강보험료는 빼고 입력하세요 (따로 계산)")
            st.caption("이 금액은 가장 먼저 지키는 목표입니다. 자산이 부족해지면 다른 지출부터 줄입니다.")
            back, nxt = nav(i, "s2")
            if nxt or back:
                p.update(essential=int(ess)); go(1 if nxt else -1)
        elif i == 3:
            st.markdown("**여행·취미·외식** 등 원하는 은퇴생활에 1년에 얼마를 쓰고 싶으신가요?")
            life = st.number_input("연 여행·취미 예산 (만 원, 오늘 가치)", 0, 50000, p["lifestyle"], step=100)
            st.caption("자산이 충분하면 모두 쓰고, 부족해지면 기본생활비를 지키기 위해 이 예산을 먼저 줄입니다.")
            back, nxt = nav(i, "s3")
            if nxt or back:
                p.update(lifestyle=int(life)); go(1 if nxt else -1)
        elif i == 4:
            c1, c2, c3 = st.columns(3)
            nps = c1.number_input("국민연금 예상 월액 (만 원)", 0, 500, p["nps"], help="국민연금공단 '내 연금 알아보기'")
            s_nps = c2.number_input("배우자 국민연금 (만 원)", 0, 500, p["s_nps"], disabled=not p["spouse"])
            nps_start = c3.slider("받기 시작할 나이", 60, 70, p["nps_start"])
            c1, c2, c3 = st.columns(3)
            priv = c1.number_input("사적연금 연 수령액 (만 원)", 0, 10000, p["priv"], help="연금저축·IRP·연금보험")
            priv_start = c2.number_input("사적연금 시작 나이", 40, 90, p["priv_start"])
            priv_years = c3.number_input("받는 기간 (년)", 0, 50, p["priv_years"])
            back, nxt = nav(i, "s4")
            if nxt or back:
                p.update(nps=int(nps), s_nps=int(s_nps), nps_start=int(nps_start), priv=int(priv),
                         priv_start=int(priv_start), priv_years=int(priv_years)); go(1 if nxt else -1)
        elif i == 5:
            c1, c2 = st.columns(2)
            leg = c1.number_input("가족에게 남기고 싶은 금액 (억 원, 오늘 가치)", 0.0, 500.0, p["legacy"], step=0.5,
                                  help="없으면 0. 집도 포함해 계산합니다")
            ch = c2.number_input("자녀 수", 0, 10, p["children"])
            back, nxt = nav(i, "s5")
            if nxt or back:
                p.update(legacy=float(leg), children=int(ch)); go(1 if nxt else -1)
        elif i == 6:
            st.markdown("금융자산을 어떤 성향으로 운용하고 계신가요?")
            prof = st.radio("투자 성향", list(PROFILES), index=list(PROFILES).index(p["profile"]), horizontal=True,
                            captions=["주식 약 30%, 안정 위주", "주식 약 50%, 균형", "주식 약 70%, 장기 성장"])
            st.caption("감당할 수 있는 위험은 자산 규모와 연금 비중에 따라 다릅니다. 결과 화면에서 세 성향을 모두 비교할 수 있습니다.")
            back, nxt = nav(i, "s6")
            if nxt or back:
                p.update(profile=prof); go(1 if nxt else -1)
        elif i == 7:
            st.markdown("해당되는 것만 입력하세요. 없으면 그대로 **다음**을 누르면 됩니다.")
            with st.expander("주택", expanded=p["h_official"] > 0):
                c1, c2, c3 = st.columns(3)
                h_off = c1.number_input("공시가격 합계 (억 원)", 0.0, 500.0, p["h_official"], step=0.5)
                h_mkt = c2.number_input("시세 합계 (억 원)", 0.0, 800.0, p["h_market"], step=0.5)
                h_n = c3.number_input("주택 수", 0, 10, p["h_n"])
                c1, c2, c3 = st.columns(3)
                h_joint = c1.checkbox("부부 공동명의", value=p["h_joint"])
                h_years = c2.number_input("보유 기간 (년)", 0, 60, p["h_years"])
                h_hi = c3.number_input("건보료 재산분 (월, 만 원)", 0.0, 500.0, p["h_hi"], help="공단 '지역보험료 모의계산'에서 소득 0, 재산만 넣은 값")
                rent = st.number_input("연 임대수입 (만 원)", 0, 100000, p["rent"])
            with st.expander("사업", expanded=p["biz"] > 0):
                c1, c2, c3 = st.columns(3)
                biz = c1.number_input("연 사업소득 (만 원)", 0, 100000, p["biz"])
                biz_until = c2.number_input("그만둘 나이", 40, 90, p["biz_until"])
                biz_work = c3.checkbox("직원 있음 (직장가입자)", value=p["biz_work"])
            with st.expander("세금·명의·모델"):
                c1, c2 = st.columns(2)
                overseas = c1.slider("주식 중 해외주식 (%)", 0, 100, p["overseas"], step=10)
                own_equal = c2.checkbox("금융자산 부부 균등 명의", value=p["own_equal"])
                survivor = st.slider("한 명만 남았을 때 생활비 (%)", 40, 100, p["survivor"], step=5)
                floor = st.radio("기본생활비 보호 방식", ["계리적", "보수적"], index=0 if p.get("floor", "계리적") == "계리적" else 1, horizontal=True,
                                 captions=["생존확률로 가중한 앞으로의 부족분 (연금 개시 반영)", "95세까지 지금 부족분이 계속된다고 가정"])
                enhanced = st.toggle("강화 모델 (금리-물가 연결, 운용보수, 간병 다중상태 등)", value=p["enhanced"])
                paths = st.select_slider("시나리오 수", [2000, 5000, 10000], value=p["paths"])
            back, nxt = nav(i, "s7")
            if nxt or back:
                p.update(h_official=h_off, h_market=h_mkt, h_n=int(h_n), h_joint=bool(h_joint), h_years=int(h_years),
                         h_hi=float(h_hi), rent=int(rent), biz=int(biz), biz_until=int(biz_until), biz_work=bool(biz_work),
                         overseas=int(overseas), own_equal=bool(own_equal), survivor=int(survivor), floor=floor,
                         enhanced=bool(enhanced), paths=int(paths)); go(1 if nxt else -1)
        else:
            need, pension, gap = income_gap(p)
            st.markdown("앞에서 입력한 내용입니다. 이후 계산은 모두 이 가정을 바탕으로 합니다.")
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="card"><h4>은퇴</h4><p>현재 {p["age"]}세 · {p["retire_age"]}세 은퇴'
                        + (f'<br>배우자 {p["s_age"]}세' if p["spouse"] else '') + '</p></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><h4>자산</h4><p>금융자산 {total_assets(p):.1f}억 원'
                        + (f'<br>은퇴 전 월 {p["saving"]}만 원 저축' if p["age"] < p["retire_age"] else '')
                        + (f'<br>주택 시세 {p["h_market"]:.1f}억 원' if p["h_market"] > 0 else '') + '</p></div>', unsafe_allow_html=True)
            c1.markdown(f'<div class="card"><h4>목표</h4><p>기본생활: 월 {p["essential"]}만 원<br>여행·취미: 연 {p["lifestyle"]:,}만 원'
                        f'<br>남길 자산: {p["legacy"]:.1f}억 원</p></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><h4>소득 · 성향</h4><p>연금: 월 {pension:.0f}만 원 ({p["nps_start"]}세부터)'
                        f'<br>투자 성향: {p["profile"]}</p></div>', unsafe_allow_html=True)
            st.markdown(f"**이 조건으로 계산해 볼까요?** 투자자산이 매달 메워야 할 금액은 약 **{max(gap, 0):.0f}만 원**입니다.")
            c1, c2 = st.columns(2)
            back = c1.form_submit_button("이전", width="stretch")
            calc = c2.form_submit_button("이 조건으로 계산하기", type="primary", width="stretch")
            if back: go(-1)
            if calc:
                ss.done = True; st.rerun()
    st.markdown('<p class="note">입력하신 정보는 계산에만 쓰이고 저장되지 않습니다.</p>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────── 결과 ───────────────────────────
with st.spinner("1만 가지 미래를 계산하는 중"):
    base = run(p)
s = base["s"]
if st.button("입력 수정하기"):
    ss.done = False; ss.step = len(STEPS) - 1; st.rerun()

tab1, tab2, tab6, tab3, tab4, tab5 = st.tabs(["목표 달성", "무엇을 바꾸면", "상품 추가해 보기", "위기 상황", "세금·절세", "가정과 한계"])

with tab1:
    need, pension, gap = income_gap(p)
    st.markdown(f'<div class="gap"><div class="eq">은퇴 필요소득 월 {need:.0f}만 원 − 연금 월 {pension:.0f}만 원 =</div>'
                f'<div class="big">투자자산이 매달 메워야 할 돈 {max(gap, 0):.0f}만 원</div>'
                f'<div class="eq">오늘 가치 기준. 연금을 받기 전({p["nps_start"]}세 이전)에는 필요소득 전액을 자산에서 씁니다.</div></div>',
                unsafe_allow_html=True)

    ok = s["기본생활 유지 확률"]; n_bad = int(round((1 - ok) * 10))
    if n_bad == 0:
        st.markdown('<div class="hero">10번 중 1번도 안 되는 경우에만<br>기본생활비가 부족해집니다</div>', unsafe_allow_html=True)
    else:
        when = f"{base['dep_age']:.0f}세 무렵 " if base["dep_age"] else ""
        st.markdown(f'<div class="hero">10번 중 <span class="n">{n_bad}번</span>은<br>{when}기본생활비가 부족해집니다</div>', unsafe_allow_html=True)
    dots = "".join(f'<div class="dot {"bad" if k < n_bad else "ok"}"></div>' for k in range(10))
    st.markdown(f'<div class="dots">{dots}</div><div class="legend">주황 = 부족해지는 경우 · 청록 = 지켜지는 경우 '
                f'&nbsp;|&nbsp; 기본생활 유지 확률 {ok * 100:.1f}% (±{s["±"] * 100:.1f}%p)</div>', unsafe_allow_html=True)

    goals = [("기본생활 유지", ok)]
    if p["lifestyle"] > 0: goals.append(("여행·취미 충족", s.get("여행·취미 평균 충족률", 0)))
    if p["legacy"] > 0: goals.append(("남길 자산 달성", s.get("유산 목표 달성 확률", 0)))
    gdf = pd.DataFrame(goals, columns=["목표", "달성"]); gdf["달성(%)"] = (gdf["달성"] * 100).round(0)
    st.markdown("**목표별 달성**")
    bars = alt.Chart(gdf).mark_bar(color=TEAL, cornerRadiusEnd=4).encode(
        y=alt.Y("목표:N", sort=None, title=None), x=alt.X("달성(%):Q", scale=alt.Scale(domain=[0, 100]), title="%"))
    st.altair_chart((bars + bars.mark_text(align="left", dx=4, color=INK).encode(text="달성(%):Q")).properties(height=40 * len(goals) + 30),
                    width="stretch")
    st.caption("기본생활: 살아있는 동안 기본생활비를 끝까지 쓸 수 있는 확률 · 여행·취미: 은퇴 후 원하는 예산을 쓴 비율의 평균 · "
               "남길 자산: 세후 상속액(집 포함)이 목표 이상일 확률")

    with st.spinner("필요 자산 계산 중"):
        req, over = required_assets(json.dumps(p, sort_keys=True))
    if req is not None:
        st.markdown(f"기본생활을 **10번 중 9번** 지키려면 금융자산 약 **{req:.1f}억 원**이 필요합니다 (지금 {total_assets(p):.1f}억 원).")
    else:
        st.markdown(f"금융자산을 {over:.0f}억 원까지 늘려도 기본생활을 10번 중 9번 지키기 어렵습니다. 생활비나 은퇴 시점을 먼저 조정해 보세요.")
    st.caption("권장 금액이 아니라, 지금 가정에서 '10번 중 9번'에 해당하는 참고 수치입니다.")

    fan = pd.DataFrame(base["fan"], columns=["나이", "p5", "p25", "p50", "p75", "p95"])
    for c in ["p5", "p25", "p50", "p75", "p95"]: fan[c] = fan[c] / 10000
    band = alt.Chart(fan).encode(x=alt.X("나이:Q", title="나이 (나이 적은 분 기준)"))
    st.markdown("**남은 금융자산의 범위**")
    st.altair_chart((band.mark_area(opacity=.18, color=TEAL).encode(y=alt.Y("p5:Q", title="억 원 (오늘 가치)"), y2="p95:Q")
                     + band.mark_area(opacity=.35, color=TEAL).encode(y="p25:Q", y2="p75:Q")
                     + band.mark_line(color=TEAL, strokeWidth=2.5).encode(y="p50:Q")).properties(height=260), width="stretch")
    st.caption("진한 띠: 가운데 50%의 경우, 연한 띠: 90%의 경우, 선: 중간값")

def whatif(rows):
    out = []
    for label, v in rows:
        r = run(p, v); rs = r["s"]; dlt, ci = paired(r, base)
        verdict = "지금 계획" if not v else ("차이 없음" if abs(dlt) <= ci else ("나빠짐" if dlt > 0 else "좋아짐"))
        row = {"바꾸는 것": label, "기본생활 유지": pct(rs["기본생활 유지 확률"]),
               "변화": "—" if not v else f"{-dlt * 100:+.1f}%p (±{ci * 100:.1f})", "판단": verdict}
        if p["lifestyle"] > 0: row["여행·취미 충족"] = pct(rs.get("여행·취미 평균 충족률", 0))
        if p["legacy"] > 0: row["남길 자산 달성"] = pct(rs.get("유산 목표 달성 확률", 0))
        out.append(row)
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")

with tab2:
    st.markdown("같은 1만 가지 미래 위에서 **한 가지만 바꿔** 비교합니다. 변화가 오차 범위 안이면 '차이 없음'입니다.")
    rows = [("지금 계획", {})]
    if p["age"] < p["retire_age"]:
        rows += [("은퇴 2년 늦추기", {"retire_delta": 2}), ("월 저축 50만 원 늘리기", {"saving_delta": 50})]
    rows += [("생활비 10% 줄이기", {"spend_mult": 0.9})]
    rows += [(f"투자 성향 {k}", {"stock_weight": w}) for k, w in PROFILES.items() if k != p["profile"]]
    for a in (p["nps_start"] - 5, p["nps_start"] + 3):
        if 60 <= a <= 70 and a >= min(p["age"], 70): rows.append((f"국민연금 {a}세부터", {"nps_start": a}))
    whatif(rows)

with tab6:
    st.markdown("주식·예금·연금·보험을 **더했을 때** 목표 달성과 위험이 어떻게 바뀌는지 봅니다. "
                "주식·예금·연금 일시납은 지금 금융자산에서 옮기는 것으로 계산합니다. **매수·가입 권유가 아닙니다.**")
    with st.form("addons"):
        st.markdown("**주식 (종목코드)**")
        c1, c2 = st.columns(2)
        code = c1.text_input("종목코드 6자리", value=ss.get("ad_code", ""), placeholder="예: 005930")
        s_amt = c2.number_input("투자 금액 (억 원)", 0.0, 100.0, float(ss.get("ad_samt", 0.0)), step=0.1)
        with st.expander("종목 통계를 직접 입력 (자동 조회가 안 될 때)"):
            c1, c2, c3 = st.columns(3)
            m_beta = c1.number_input("시장 베타", -1.0, 3.0, 1.0, step=0.1)
            m_idio = c2.number_input("고유 변동성 (%)", 0.0, 100.0, 25.0, step=1.0)
            m_rate = c3.number_input("금리 +1%p 때 수익률 변화 (%)", -20.0, 20.0, 0.0, step=0.5)
            manual = st.checkbox("직접 입력값 사용")
        st.markdown("**예금**")
        d_amt = st.number_input("예금으로 옮길 금액 (억 원)", 0.0, 100.0, 0.0, step=0.1)
        st.markdown("**종신연금 (일시납)**")
        c1, c2, c3 = st.columns(3)
        a_amt = c1.number_input("일시납 보험료 (억 원, 오늘 가치)", 0.0, 50.0, 0.0, step=0.1)
        a_age = c2.number_input("가입·개시 나이", max(p["age"], 45), 90, max(p["age"], 65))
        a_joint = c3.checkbox("부부형 (한 명이라도 살아있으면 지급)")
        st.markdown("**간병보험**")
        c1, c2 = st.columns(2)
        c_ben = c1.number_input("간병 시 연 보장액 (만 원)", 0, 10000, 0, step=100)
        c_who = c2.radio("피보험자", ["본인", "배우자"], horizontal=True, disabled=not p["spouse"])
        c_trig = st.radio("지급 조건", ["중증(1~2등급)", "모든 장기요양 등급"], horizontal=True)
        pc = 0
        if p["age"] < p["retire_age"]:
            st.markdown("**연금저축·IRP 추가 납입 (은퇴 전)**")
            pc = st.number_input("월 납입액 (만 원) — 저축액 중 일부를 연금계좌로", 0, 500, 0, step=10)
        go_cmp = st.form_submit_button("비교하기", type="primary", width="stretch")

    if go_cmp:
        ss.ad_code, ss.ad_samt = code.strip(), s_amt
        items, info = [], {}
        if s_amt > 0:
            if manual or not code.strip():
                stt = {"beta": m_beta, "idio_sigma": m_idio / 100, "rate_beta": m_rate, "name": "직접 입력"}
            else:
                try:
                    with st.spinner("종목 데이터 조회 중"):
                        stt = stock_lookup(code.strip())
                    stt["name"] = code.strip()
                except Exception as ex:
                    st.error(f"종목 데이터를 가져오지 못했습니다: {ex} — '직접 입력값 사용'으로 계산할 수 있습니다.")
                    stt = None
            if stt:
                info["stock"] = stt
                items.append((f"주식 {stt['name']} {s_amt:.1f}억", {"stock_amount": s_amt * 10000, "stock_beta": stt["beta"],
                              "stock_idio_sigma": stt["idio_sigma"], "stock_rate_beta": stt["rate_beta"]}))
        if d_amt > 0: items.append((f"예금 {d_amt:.1f}억", {"deposit_amount": d_amt * 10000}))
        if a_amt > 0: items.append((f"종신연금 {a_amt:.1f}억 ({a_age}세{', 부부형' if a_joint else ''})",
                                   {"annuity_premium": a_amt * 10000, "annuity_start_age": int(a_age), "annuity_joint": bool(a_joint)}))
        if c_ben > 0: items.append((f"간병보험 연 {c_ben:,}만 원 ({c_who})", {"care_benefit": float(c_ben), "care_member": 0 if c_who == "본인" else 1,
                                                                    "care_trigger": "severe" if c_trig.startswith("중증") else "any"}))
        if pc > 0: items.append((f"연금저축 월 {pc}만 원", {"pension_contrib_annual": float(pc * 12)}))
        if not items:
            st.info("추가할 상품의 금액을 하나 이상 입력하세요.")
        else:
            if len(items) > 1:
                allv = {}
                for _, v in items: allv.update(v)
                items.append(("모두 추가", allv))
            rows, results = [], {}
            with st.spinner("비교 계산 중"):
                for label, v in [("지금 계획", {})] + items:
                    r = run(p, {"addons": v} if v else {}); results[label] = r; rs = r["s"]
                    dlt, ci = paired(r, base) if v else (0.0, 0.0)
                    rows.append({"구성": label, "기본생활 유지": pct(rs["기본생활 유지 확률"]),
                                 "변화": "—" if not v else f"{-dlt * 100:+.1f}%p (±{ci * 100:.1f})",
                                 "부족할 때 평균 부족액": f"{rs.get('부족 시 평균 부족액(실질)', 0) / 10000:.2f}억",
                                 "85세 금융자산 (중간값)": f"{r['p50_85'] / 10000:.2f}억",
                                 "여행·취미 충족": pct(rs.get("여행·취미 평균 충족률", 0)) if p["lifestyle"] > 0 else "—",
                                 "남길 자산 달성": pct(rs.get("유산 목표 달성 확률", 0)) if p["legacy"] > 0 else "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("변화: 같은 1만 가지 미래에서의 기본생활 유지 확률 차이(95% 신뢰구간). "
                       "부족할 때 평균 부족액: 돈이 모자란 경우 평생 모자란 기본생활비 합계(오늘 가치) — 작을수록 덜 심각합니다.")
            if "stock" in info:
                stt = info["stock"]; r = next(v for k, v in results.items() if k.startswith("주식"))
                st.markdown(f"**종목 {stt['name']} 통계**" + (f" ({stt.get('start','')} ~ {stt.get('end','')}, {stt.get('months','')}개월)" if "months" in stt else ""))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("시장 베타", f"{stt['beta']:.2f}")
                c2.metric("기존 자산과 상관", f"{r['stock_corr']:.2f}" if r["stock_corr"] is not None else "—")
                c3.metric("연 변동성", f"{stt.get('sigma', np.sqrt(stt['beta']**2*0.18**2+stt['idio_sigma']**2)) * 100:.0f}%")
                c4.metric("금리 +1%p 때", f"{stt['rate_beta']:+.1f}%" if abs(stt["rate_beta"]) > 1e-9 else "자료 없음")
                st.caption("기대수익은 과거 수익률이 아니라 '금리 + 베타 × 위험프리미엄'으로 계산합니다. "
                           "금리 민감도는 ECOS 키가 설정된 경우에만 추정됩니다.")
            for k, r in results.items():
                if k.startswith("종신연금") and r["annuity_pay"] > 0:
                    st.caption(f"{k}: 예상 연금액 연 약 {r['annuity_pay']:,.0f}만 원 (가입 시점 명목 금액, 물가연동 아님, 사업비 5% 가정)")
                if k.startswith("간병보험") and r["care_premium"] > 0:
                    st.caption(f"{k}: 모델로 산출한 보험료 연 약 {r['care_premium']:,.0f}만 원 (80세까지 납입, 부가보험료 30% 가정, 지급 조건: {c_trig})")

with tab3:
    st.markdown("정해진 위기가 온다고 가정했을 때 목표가 얼마나 흔들리는지 봅니다.")
    ra = max(0, p["retire_age"] - p["age"])
    whatif([("평소", {}),
            ("은퇴 첫해 주가 −40%", {"eco": {"stock_shocks": {str(ra): -0.40}}}),
            ("은퇴 10년 뒤 주가 −40%", {"eco": {"stock_shocks": {str(ra + 10): -0.40}}}),
            ("3년 연속 물가 6%", {"eco": {"infl_shocks": {"0": .06, "1": .06, "2": .06}}}),
            ("예상보다 오래 삶 (사망률 −20%)", {"q_mult": 0.8}),
            ("간병 위험 2배", {"care_mult": 2.0})])
    st.caption("같은 폭락도 은퇴 직후에 올수록 타격이 큽니다 (수익률 순서 위험).")

with tab4:
    st.markdown("계좌와 명의를 바꾸면 세금·건보료와 목표 달성이 어떻게 달라지는지 비교합니다.")
    rows = [("지금 그대로", {}), ("ISA 활용", {"tax": {"use_isa": True}}), ("연금계좌 활용", {"tax": {"use_pension": True}}),
            ("ISA + 연금계좌", {"tax": {"use_isa": True, "use_pension": True}})]
    if p["children"] > 0:
        rows.append(("+ 자녀당 5천만 원씩 10년마다 증여", {"tax": {"use_isa": True, "use_pension": True, "gift_per_child_10y": 5000}}))
    out = []
    for label, v in rows:
        r = run(p, v); rs = r["s"]
        out.append({"방법": label, "기본생활 유지": pct(rs["기본생활 유지 확률"]),
                    "20년 세금+건보료": f"{(rs['20년 세금'] + rs['20년 건보료']) / 10000:.2f}억",
                    "가족에게 남는 돈(중앙값)": f"{rs['가족 이전 총액(중앙값)'] / 10000:.1f}억"})
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
    st.caption("증여는 가족에게 남는 돈을 늘리지만 내 기본생활 유지 확률을 낮출 수 있습니다.")

with tab5:
    st.markdown("""
**계산 방식**
- 금리·물가·주가가 서로 연결되어 움직이는 1만 가지 미래와, 통계청 생명표 기반 부부 각자의 수명·간병을 계산합니다.
- 목표는 **기본생활 → 여행·취미 → 남길 자산** 순으로 지킵니다. 매년 남은 금융자산이 "앞으로 기본생활비 부족분의 기대 현재가치"(생존확률 가중, 연금 개시 반영, 실질 2% 할인, 사망률 80%로 보수적 계산)보다 많을 때만 여행·취미 예산을 씁니다.
- 국민연금, 사적연금 분리과세, 금융소득종합과세, 건강보험료, 재산세·종부세, 상속·증여세를 반영합니다.

**알아두실 점**
- 사망률은 통계청 2024 완전생명표(미래 수명 연장 미반영), 금리·물가는 한국은행 ECOS 2000~2026년 데이터로 추정했습니다. 간병은 건강보험공단 장기요양 통계(연령별 인정률, 등급 구성)와 2026 본인부담·간병비 시세로 보정했습니다. 주식 수익률만 아직 **임시 가정값**입니다. 결과의 절대값보다 **선택지 사이의 차이**를 보세요.
- 세법은 2026년 9월 기준으로 단순화했습니다. **투자 추천이나 세무 자문이 아니며**, 실제 결정 전 전문가와 상의하세요.
- 입력하신 정보는 계산에만 쓰이고 서버에 저장하지 않습니다.
""")
    st.markdown("모델 수식과 검증: [GitHub 저장소](https://github.com/minyoungkang1227/retirement-simulator)")

st.markdown('<p class="note">추천이 아닌 비교 결과입니다 · 제도 기준 2026-09 · 가정값 임시 · © 2026 강민영</p>', unsafe_allow_html=True)
