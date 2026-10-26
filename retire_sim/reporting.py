"""상담 리포트 조립: 입력 → 계산 → 고객에게 남길 HTML 한 장.

`report.py` 가 '어떻게 그릴지'라면 여기는 '무엇을 담을지'다.
앱과 API가 같은 함수를 쓰므로 화면에서 본 숫자와 리포트의 숫자가 어긋날 수 없다.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from . import engine_tax, report
from .advice import action_ranking, assumption_band, run_case
from .branding import Brand
from .inputs import income_gap
from .session import calc_id
from . import __version__


def fan_rows(res, upto_age: int = 100) -> list:
    """나이별 (나이, 운 좋을 때, 보통, 나쁠 때) 실질자산 — 억 원. 생존 경로가 적은 구간은 뺀다.
    앱의 그래프와 리포트의 그래프가 같은 숫자를 쓰도록 여기 한 곳에서만 계산한다."""
    W, alive = res["W_real"], res["hh_alive"]
    base_age = res["youngest_age"]
    rows = []
    for t in range(W.shape[1]):
        age = base_age + t
        if age > upto_age or alive[:, t].sum() <= 200:
            continue
        v = np.percentile(W[alive[:, t], t], [25, 50, 75]) / 10000
        rows.append((int(age), float(v[2]), float(v[1]), float(v[0])))
    return rows


def post_retirement_average(res, p: dict, key: str) -> float:
    """은퇴 후 '살아 있는 해'의 평균 (세금·건보료·보유세). 20년 합계를 나누면 은퇴 전이 섞인다."""
    t0 = max(0, min(p.get("retire_age", 0) - p["age"], res["T"]))
    live = res["hh_alive"][:, t0:res["T"]]
    v = res[key][:, t0:res["T"]]
    return float(v[live].mean()) if live.any() else 0.0


def money_flow(res, p: dict, pension: float = None):
    """은퇴 후 평균 1년의 돈 흐름 — [(항목, 연 금액)]. 들어오는 돈은 +, 나가는 돈은 −.

    리포트의 폭포 그래프와 앱 결과 탭이 같은 숫자를 쓰도록 여기 한 곳에서 만든다.
    """
    if pension is None:
        pension = income_gap(p)[1]
    flow = [("연금 수입", pension * 12), ("생활비", -(p["essential"] * 12 + p["lifestyle"])),
            ("세금", -post_retirement_average(res, p, "tax_y")),
            ("건보료", -post_retirement_average(res, p, "hi_y"))]
    prop = post_retirement_average(res, p, "prop_y")
    if prop > 0:
        flow.append(("보유세", -prop))
    return flow


def build_report(p: dict, client_name: str = "", advisor: str = "", brand: Brand = None,
                 res=None, actions=None, band=None, today: date = None) -> str:
    """상담 리포트 HTML. res·actions·band 를 주면 다시 계산하지 않는다(앱은 이미 가진 것을 넘긴다)."""
    b = brand or Brand()
    today = today or date.today()
    res = res if res is not None else run_case(p)
    s = engine_tax.summarize(res)
    ok = s["기본생활 유지 확률"]; n_ok = int(round(ok * 10))
    need, pension, gap = income_gap(p)
    short = max(gap, 0)

    d = res["depleted_at"]; dep = d[d >= 0] + res["youngest_age"]
    when = f"{np.median(dep):.0f}세쯤 " if len(dep) else ""
    hero = (f'지금 계획대로면 10번 중 <span class="n">{10 - n_ok}번</span>은 {when}생활비가 모자랍니다.'
            if n_ok < 10 else "지금 계획대로면 10번 중 9번 이상 생활비를 끝까지 낼 수 있습니다.")

    goals = [("🏠 기본 생활", "먹고 사는 데 꼭 필요한 돈", f"10번 중 {n_ok}번 가능")]
    if p["lifestyle"] > 0:
        goals.append(("✈️ 여행·취미", "원하는 예산 중 쓸 수 있는 정도",
                      f"{s.get('여행·취미 평균 충족률', 0) * 100:.0f}% 정도"))
    if p["legacy"] > 0:
        goals.append(("👨‍👩‍👧 자녀", f"{p['legacy']:.1f}억 원 남기기",
                      f"10번 중 {int(round(s.get('유산 목표 달성 확률', 0) * 10))}번 가능"))

    lo, hi = band if band is not None else assumption_band(p)
    pct = lambda x: f"{x * 100:.0f}%"
    band_txt = (f"주식 수익률·수명·간병 발생률은 아무도 정확히 모릅니다. 세 가정을 각각 추정 폭만큼만 움직여도 위 "
                f"<b>{pct(ok)}</b>는 <b>{pct(lo)} ~ {pct(hi)}</b> 사이에서 움직입니다. "
                f"계산을 더 많이 해도 이 폭은 줄지 않으니, 숫자 하나보다 <b>선택을 바꿨을 때의 차이</b>를 보셔야 합니다.")

    if actions is None:
        actions = action_ranking(p) if n_ok < 9 else []
    acts = [(lab, how, f"{ok * 100:.0f}% → {min(ok + g, 1) * 100:.0f}% ({g * 100:+.0f}%p)")
            for lab, how, g, _ in actions[:3]]

    flow = money_flow(res, p, pension)

    assume = ("· 금리·물가·주가가 서로 연결되어 움직이는 1만 가지 미래를 계산했습니다(한국은행 ECOS 2000~2026, KOSPI).<br>"
              "· 수명은 통계청 2024 완전생명표에 앞으로의 수명 연장을 반영했고, 간병은 건강보험공단 장기요양 통계로 추정했습니다.<br>"
              + ("· 간병비는 일반 물가만큼만 오른다고 보았습니다"
                 if not p.get("care_growth") else
                 f"· 간병비는 임금을 따라가므로 일반 물가보다 연 {p['care_growth']:.1f}%p 빠르게 오른다고 보았습니다")
              + ("(부양 부모님의 간병비도 분담 비율만큼 포함).<br>" if p.get("parents") else ".<br>")
              + "· 국민연금·유족연금·기초연금·건강보험료·세금을 2026년 기준으로 반영했습니다.<br>"
              f"· 기본생활비 월 {p['essential']:,}만 원, 여행·취미 연 {p['lifestyle']:,}만 원, "
              f"국민연금 개시 {p['nps_start']}세로 계산했습니다.<br>"
              "· 결과는 가정에 따라 달라집니다. 절대값보다 선택지 사이의 차이를 보시기 바랍니다.")

    return report.build(dict(
        client=client_name or "고객", advisor_line=f" · 상담 {advisor}" if advisor else "",
        date=f"{today:%Y년 %m월 %d일}", calc_id=calc_id(p), version=__version__,
        hero=hero, goals=goals,
        gap_eq=f"필요한 생활비 월 {need:.0f}만 원 − 받는 연금 월 {pension:.0f}만 원 (오늘 돈 가치 기준)",
        gap_big=(f"매달 {short:.0f}만 원을 모아둔 돈에서 꺼내 써야 합니다" if short > 0
                 else "연금만으로 생활비가 충분합니다"),
        band=band_txt, actions=acts,
        fan_svg=report.line_chart_svg(fan_rows(res), brand=b),
        flow_svg=report.balance_svg(flow, brand=b), flow_text=report.balance_sentence(flow),
        assumptions=assume), brand=b)
