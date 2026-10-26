"""상담 리포트: 고객에게 남길 한두 장짜리 요약을 HTML로 만든다.

왜 PDF가 아니라 HTML인가
  한글 PDF는 글꼴 파일을 함께 배포해야 하고(용량·라이선스), 배포 환경마다 글꼴이 달라 깨지기 쉽다.
  HTML은 어디서든 같게 보이고, 브라우저의 '인쇄 → PDF로 저장'으로 PDF가 된다.
  표지·여백·쪽 나눔은 @page 와 @media print 로 맞춰 두었다.

그림은 외부 라이브러리 없이 SVG 문자열로 직접 그린다(의존성 0, 인쇄 시 깨지지 않음).
"""
from __future__ import annotations

import html
from datetime import date

from .branding import DEFAULT as DEFAULT_BRAND, Brand

# 그림에서 쓰는 기본색. 기관 브랜드가 들어오면 build() 에서 갈아끼운다.
# (색은 branding.Brand 가 정한다 — 이 상수는 더 쓰지 않는다)


def _mix(hex_color: str, t: float) -> str:
    """색을 흰색 쪽으로 t만큼 섞는다 (옅은 선용). 브랜드 색 하나로 색 계열을 만든다."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, bl = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda c: int(round(c + (255 - c) * t))
    return f"#{f(r):02X}{f(g):02X}{f(bl):02X}"


def _esc(x) -> str:
    return html.escape(str(x))


def _lines_svg(series, title, w=640, h=240, pad=46, top=24, brand: Brand = None) -> str:
    """나이-금액 선 그래프 한 장. 축·0선·범례를 그린다.

    series: [(범례 이름, [(나이, 억 원), ...], 색, 점선 패턴 또는 "")]
    팬차트(한 계획의 운 좋을/보통/나쁠 때)와 A/B 비교가 같은 축·범례 코드를 쓰도록 하나로 모았다.
    """
    series = [s for s in series if s[1]]
    if not series:
        return ""
    b = brand or DEFAULT_BRAND
    muted, grid = b.muted, b.line
    xs = [x for _, pts, _, _ in series for x, _ in pts]
    ys = [y for _, pts, _, _ in series for _, y in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys)
    if y1 <= y0:
        y1 = y0 + 1
    legend_h = 18
    bottom = h - pad                                   # 가로축 기준선
    sx = lambda x: pad + (x - x0) / max(x1 - x0, 1) * (w - pad - 14)
    sy = lambda y: bottom - (y - y0) / (y1 - y0) * (bottom - top - legend_h)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{_esc(title)}">']
    for i in range(5):                                 # 가로 눈금
        v = y0 + (y1 - y0) * i / 4
        y = sy(v)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w - 14}" y2="{y:.1f}" stroke="{grid}" stroke-width="1"/>')
        out.append(f'<text x="{pad - 6}" y="{y + 4:.1f}" font-size="11" fill="{muted}" text-anchor="end">{v:.0f}</text>')
    for _, pts, color, dash in series:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        pl = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{pl}" fill="none" stroke="{color}" stroke-width="2.4"{d}/>')
    for x in (x0, (x0 + x1) // 2, x1):                 # 나이 축
        out.append(f'<text x="{sx(x):.1f}" y="{h - 14}" font-size="11" fill="{muted}" text-anchor="middle">{x}세</text>')
    out.append(f'<text x="{pad - 34}" y="14" font-size="11" fill="{muted}">억 원</text>')
    cx = pad                                           # 범례 (전에는 라벨을 만들고 버려서 어느 선인지 알 수 없었다)
    for lab, _, color, dash in series:
        if not lab:                                    # 라벨이 없는 선은 범례에 넣지 않는다(설명은 캡션으로)
            continue
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{cx}" y1="{top - 10}" x2="{cx + 16}" y2="{top - 10}" '
                   f'stroke="{color}" stroke-width="2.4"{d}/>')
        out.append(f'<text x="{cx + 21}" y="{top - 6}" font-size="11" fill="{muted}">{_esc(lab)}</text>')
        cx += 27 + len(lab) * 11
    out.append("</svg>")
    return "".join(out)


def line_chart_svg(rows, brand: Brand = None, **kw) -> str:
    """한 계획의 나이별 남는 돈. rows: [(나이, 운이 좋을 때, 보통, 운이 나쁠 때)] — 억 원."""
    b = brand or DEFAULT_BRAND
    pick = lambda i: [(r[0], r[i]) for r in rows]
    return _lines_svg([("운이 좋을 때", pick(1), _mix(b.primary, 0.45), "4 3"),
                       ("보통일 때", pick(2), b.primary, ""),
                       ("운이 나쁠 때", pick(3), b.accent, "")],
                      "나이별 남는 돈", brand=b, **kw)


def compare_chart_svg(rows_a, rows_b, label_a="지금 계획", label_b="바꾼 뒤",
                      brand: Brand = None, **kw) -> str:
    """두 계획을 겹쳐 비교. **색이 계획, 선 모양이 시나리오**다 — 실선 보통일 때, 점선 운이 나쁠 때.

    rows_*: line_chart_svg 와 같은 [(나이, 좋을 때, 보통, 나쁠 때)].
    '좋을 때'는 빼서 선이 6개가 되지 않게 하고, 범례는 계획 둘만 둔다(선 모양은 캡션으로 설명).
    """
    b = brand or DEFAULT_BRAND
    grey = _mix(b.ink, 0.55)
    pick = lambda rows, i: [(r[0], r[i]) for r in rows]
    return _lines_svg([(label_a, pick(rows_a, 2), grey, ""),
                       (None, pick(rows_a, 3), grey, "4 3"),
                       (label_b, pick(rows_b, 2), b.primary, ""),
                       (None, pick(rows_b, 3), b.primary, "4 3")],
                      f"{label_a} vs {label_b} 비교", brand=b, **kw)


def waterfall_svg(items, w=640, h=250, pad=46, brand: Brand = None) -> str:
    """은퇴 후 평균 1년, 연금이 들어와 생활비·세금으로 빠지고 얼마가 모자라는지 보여주는 폭포 그래프.

    items: [(항목, 연 금액)] — 첫 항목은 들어오는 돈(+), 나머지는 나가는 돈(−).
    마지막에 '모아둔 돈에서' 막대를 더해 부족분이 어디서 메워지는지 눈으로 잇는다."""
    items = [(k, float(v)) for k, v in items if abs(v) > 1e-9]
    if not items:
        return ""
    b = brand or DEFAULT_BRAND
    teal, coral, muted, line = b.primary, b.accent, b.muted, b.line
    net = sum(v for _, v in items)
    bars, run = [], 0.0
    for k, v in items:
        bars.append((k, run, run + v, v))                 # (이름, 시작, 끝, 증감)
        run += v
    if net < 0:
        bars.append(("모아둔 돈에서", run, 0.0, -net))     # 부족분을 메우는 막대
    lo = min(min(b[1], b[2]) for b in bars); hi = max(max(b[1], b[2]) for b in bars)
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    if hi <= lo:
        hi = lo + 1
    top, bot = 26, h - 46
    sy = lambda y: bot - (y - lo) / (hi - lo) * (bot - top)
    bw = (w - pad - 14) / len(bars)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="은퇴 후 1년 돈의 흐름">']
    out.append(f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{w - 14}" y2="{sy(0):.1f}" stroke="{line}" stroke-width="1"/>')
    for i, (k, y0, y1, v) in enumerate(bars):
        x = pad + i * bw + bw * 0.2
        bwid = bw * 0.6
        ytop, ybot = sy(max(y0, y1)), sy(min(y0, y1))
        color = teal if v > 0 else coral
        if k == "모아둔 돈에서":
            color = "#7FB8BE"   # 모아둔 돈 = 옅은 청록 (나이별 그래프의 '운이 좋을 때'와 같은 계열)
        out.append(f'<rect x="{x:.1f}" y="{ytop:.1f}" width="{bwid:.1f}" height="{max(ybot - ytop, 1.5):.1f}" fill="{color}" rx="3"/>')
        out.append(f'<text x="{x + bwid / 2:.1f}" y="{ytop - 6:.1f}" font-size="13" fill="{b.ink}" text-anchor="middle">{v:+,.0f}</text>')
        out.append(f'<text x="{x + bwid / 2:.1f}" y="{bot + 17:.1f}" font-size="13" fill="{muted}" text-anchor="middle">{_esc(k)}</text>')
        if i + 1 < len(bars):                              # 막대를 잇는 점선
            xr = x + bwid; y = sy(y1)
            out.append(f'<line x1="{xr:.1f}" y1="{y:.1f}" x2="{x + bw:.1f}" y2="{y:.1f}" '
                       f'stroke="{line}" stroke-width="1" stroke-dasharray="3 2"/>')
    out.append(f'<text x="{pad}" y="{h - 8}" font-size="12" fill="{muted}">단위: 만 원 / 년 (오늘 돈 가치, 은퇴 후 평균)</text>')
    out.append("</svg>")
    return "".join(out)


def css(brand: Brand = None) -> str:
    """브랜드 색을 입힌 리포트 스타일."""
    b = brand or DEFAULT_BRAND
    return f"""
@page {{ size: A4; margin: 14mm 13mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Pretendard, -apple-system, 'Malgun Gothic', sans-serif; color: {b.ink};
        margin: 0; padding: 18px; background: #F4F7F8; line-height: 1.55; }}
.page {{ max-width: 820px; margin: 0 auto 18px; background: #fff; padding: 26px 30px;
         border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,.07); }}
h1 {{ font-size: 1.5rem; margin: 0 0 2px; }}
h2 {{ font-size: 1.05rem; margin: 22px 0 8px; color: {b.primary};
      border-bottom: 2px solid {b.line}; padding-bottom: 5px; }}
.meta {{ color: {b.muted}; font-size: .86rem; }}
.tagline {{ color: {b.primary}; font-size: .9rem; font-weight: 600; margin: 0 0 4px; }}
.logo {{ max-height: 34px; margin-bottom: 6px; display: block; }}
.hero {{ font-size: 1.3rem; font-weight: 700; line-height: 1.45; margin: 14px 0 4px; }}
.hero .n {{ color: {b.accent}; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0; }}
.card {{ border: 1px solid {b.line}; border-radius: 9px; padding: 11px 13px; }}
.card .h {{ font-weight: 700; color: {b.primary}; font-size: .95rem; }}
.card .t {{ color: {b.muted}; font-size: .8rem; margin: 1px 0 5px; }}
.card .v {{ font-size: 1.2rem; font-weight: 700; }}
.gap {{ background: #E6EDEF; border-radius: 9px; padding: 12px 15px; margin: 12px 0; }}
.gap .big {{ font-size: 1.25rem; font-weight: 700; color: {b.primary}; }}
.gap .eq {{ color: {b.muted}; font-size: .88rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid {b.line}; }}
th {{ color: {b.muted}; font-weight: 600; font-size: .83rem; }}
td.r, th.r {{ text-align: right; }}
.note {{ color: {b.muted}; font-size: .8rem; line-height: 1.6; }}
.band {{ background: #FAFCFC; border-left: 3px solid {b.primary}; padding: 9px 13px; margin: 10px 0;
         font-size: .88rem; }}
.foot {{ color: {b.muted}; font-size: .74rem; margin-top: 16px; border-top: 1px solid {b.line}; padding-top: 9px; }}
@media print {{
  body {{ background: #fff; padding: 0; }}
  .page {{ box-shadow: none; border-radius: 0; margin: 0; padding: 0; max-width: none; }}
  .pagebreak {{ page-break-before: always; }}
  .noprint {{ display: none; }}
}}
"""


def build(ctx: dict, brand: Brand = None) -> str:
    """ctx 키: client, advisor, calc_id, version, hero, goals, gap_eq, gap_big, band, actions,
    fan_svg, flow_svg, assumptions. brand 를 주면 기관 이름·색·로고·고지문이 바뀐다."""
    b = brand or DEFAULT_BRAND
    g = "".join(
        f'<div class="card"><div class="h">{_esc(h)}</div><div class="t">{_esc(t)}</div><div class="v">{_esc(v)}</div></div>'
        for h, t, v in ctx["goals"])
    acts = "".join(
        f'<tr><td>{i}</td><td>{_esc(lab)}</td><td>{_esc(how)}</td><td class="r">{_esc(eff)}</td></tr>'
        for i, (lab, how, eff) in enumerate(ctx["actions"], 1))
    logo = (f'<img src="{_esc(b.logo_url)}" alt="{_esc(b.name)}" class="logo">' if b.logo_url else "")
    tagline = (f'<div class="tagline">{_esc(b.tagline)}</div>' if b.tagline else "")
    act_block = (f'<h2>지금 할 수 있는 일 — 효과가 큰 순서</h2><table>'
                 f'<tr><th>순위</th><th>무엇을</th><th>어떻게</th><th class="r">기본 생활 지킬 가능성</th></tr>'
                 f'{acts}</table>') if ctx["actions"] else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(b.name)} 노후 점검 리포트 — {_esc(ctx['client'])}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>{css(b)}</style></head><body>
<div class="page">
  {logo}<h1>{_esc(b.name)} 노후 점검 리포트</h1>
  {tagline}
  <div class="meta">{_esc(ctx['client'])} 님 · 작성 {_esc(ctx['date'])}{_esc(ctx['advisor_line'])}</div>
  <div class="hero">{ctx['hero']}</div>
  <div class="gap"><div class="eq">{ctx['gap_eq']}</div><div class="big">{ctx['gap_big']}</div></div>
  <h2>세 가지 목표</h2>
  <div class="grid">{g}</div>
  <div class="band">{ctx['band']}</div>
  <h2>나이별로 남는 돈 (오늘 돈 가치)</h2>
  {ctx['fan_svg']}
  <div class="note">선이 0에 닿는 때가 모아둔 돈을 다 쓰는 때입니다. 집은 포함하지 않았습니다.</div>
  {act_block}
  <div class="foot">
    계산 ID {_esc(ctx['calc_id'])} · 모델 v{_esc(ctx['version'])} · 1만 가지 미래 시뮬레이션<br>
    같은 입력과 같은 모델 버전이면 언제 다시 계산해도 같은 결과가 나옵니다.
  </div>
</div>
<div class="page pagebreak">
  <h2>은퇴 후 1년, 돈은 이렇게 움직입니다</h2>
  {ctx['flow_svg']}
  <div class="note">연금으로 들어온 돈에서 생활비·세금·건강보험료가 빠지고, 모자라는 만큼을 모아둔 돈에서 꺼내 씁니다.</div>
  <h2>이 계산이 가정한 것</h2>
  <div class="note">{ctx['assumptions']}</div>
  <div class="foot">{_esc(b.disclaimer)}<br>{_esc(b.footer)}</div>
</div>
</body></html>"""
