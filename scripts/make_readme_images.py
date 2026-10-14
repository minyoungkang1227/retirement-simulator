import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""README 그림을 **지금 모델로** 다시 만든다.

    PYTHONPATH=. python scripts/make_readme_images.py

왜 필요한가: 이 그림들을 만드는 스크립트가 없어서 v27 무렵 숫자에 멈춰 있었다.
기준 가구의 기본생활 달성이 그림에는 48%, 바로 아래 README 표에는 69.8%로 서로 달랐다.
모델을 고칠 때마다 이걸 돌리면 그런 어긋남이 생기지 않는다.

ablate.py 와 다르다: ablate 는 **구성요소를 꺼서** '이 기능이 값을 하나'를 보고,
여기 민감도는 **가정값을 그럴듯한 범위 안에서 흔들어** '무엇이 결과를 가장 크게 움직이나'를 본다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from retire_sim import engine_tax, metrics, mortality
from retire_sim.config import CareMarkov, Household, Person, SimConfig
from retire_sim.economy_v2 import EconomyV2
from retire_sim.tax import TaxConfig

TEAL, CORAL, INK, MUTED, LINE = "#1F6F78", "#E2725B", "#1B2B2E", "#5A6B6E", "#CBD7DB"
IMG = os.path.join(os.path.dirname(__file__), "..", "docs", "images")

for cand in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
KO = plt.rcParams["font.family"] not in ("DejaVu Sans", ["DejaVu Sans"])   # 한글 글꼴이 있나

L = {  # 한글 글꼴이 없는 환경에서도 그림이 깨지지 않게
    "goals": ("목표별 달성 확률 (%)", "Goal achievement (%)"),
    "whatif": ("무엇을 바꾸면 (기본생활 지킬 확률, %p)", "What if (essential success, pp)"),
    "ess": ("기본 생활", "Essential"), "life": ("여행·취미", "Lifestyle"), "leg": ("남길 돈", "Legacy"),
    "r2": ("은퇴 2년 늦추기", "Retire 2y later"), "sp": ("생활비 10% 줄이기", "Spending −10%"),
    "gr": ("주식 70%", "Equity 70%"), "sv": ("월 50만 원 더 저축", "Save +0.5M/mo"),
    "cs": ("주식 30%", "Equity 30%"),
}
t = lambda k: L[k][0 if KO else 1]

ECO, TC, CFG, QX = EconomyV2.enhanced(), TaxConfig(overseas_share=0.2), SimConfig(care=CareMarkov()), mortality.default_qx()


def couple_60(**kw):
    base = dict(members=[Person(60, "M", nps_monthly=110, private_pension_annual=600, private_pension_start=60),
                         Person(58, "F", nps_monthly=50)],
                liquid_assets=50_000, stock_weight=0.4, annual_spending=3600,
                essential=3000, lifestyle=600, legacy_target=10_000)
    return Household(**{**base, **kw})


def run(hh):
    return engine_tax.run(hh, CFG, e=ECO, tc=TC, qx_table=QX)


base = run(couple_60())
s = engine_tax.summarize(base)
print(f"기준 가구: 기본생활 {s['기본생활 유지 확률']*100:.1f}% · "
      f"여행·취미 {s.get('여행·취미 평균 충족률', 0)*100:.1f}% · 유산 {s.get('유산 목표 달성 확률', 0)*100:.1f}%")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
vals = [s["기본생활 유지 확률"] * 100, s.get("여행·취미 평균 충족률", 0) * 100,
        s.get("유산 목표 달성 확률", 0) * 100]
names = [t("ess"), t("life"), t("leg")]
a1.barh(names[::-1], vals[::-1], color=TEAL, height=.55)
for i, v in enumerate(vals[::-1]):
    a1.text(v + 1.5, i, f"{v:.0f}%", va="center", color=INK, fontsize=11)
a1.set(xlim=(0, 100), title=t("goals"))

cases = [(t("r2"), {"retire_age": 62}), (t("sp"), {"annual_spending": 3240, "essential": 2700}),
         (t("gr"), {"stock_weight": 0.7}), (t("sv"), {}), (t("cs"), {"stock_weight": 0.3})]
rows = []
for lab, kw in cases:
    if not kw:
        continue
    r = run(couple_60(**kw))
    d = metrics.compare(r, base)
    rows.append((lab, (engine_tax.summarize(r)["기본생활 유지 확률"] - s["기본생활 유지 확률"]) * 100, d["±"] * 100))
rows.sort(key=lambda x: x[1])
a2.barh([r[0] for r in rows], [r[1] for r in rows],
        xerr=[r[2] for r in rows], color=[TEAL if r[1] > 0 else CORAL for r in rows],
        height=.55, error_kw=dict(ecolor=MUTED, capsize=3, lw=1))
a2.axvline(0, color=LINE, lw=1)
a2.set(title=t("whatif"))
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "goals_whatif.png"), dpi=130, facecolor="white")
print("goals_whatif.png 갱신:", [(r[0], round(r[1], 1)) for r in rows])


# ── ② 세 전략(ISA·연금계좌·증여) ────────────────────────────────────────
# 전략은 기본값이 꺼져 있으므로 **켜서** 재야 한다 (v37에서 거꾸로 재다 바로잡은 자리).
from dataclasses import replace

L.update({"taxfig": ("세 전략을 켰을 때 — 은퇴 후 연 세금·건보료와 기본생활 지킬 확률",
                     "Tax strategies — annual tax/premium after retirement, and essential success"),
          "none": ("전략 없음", "None"), "isa": ("ISA", "ISA"), "pen": ("연금계좌", "Pension acct"),
          "gift": ("증여", "Gifting"), "all3": ("셋 다", "All three"),
          "tax": ("소득세", "Income tax"), "hi": ("건강보험료", "Health premium"),
          "prop": ("보유세", "Property tax")})

rich = couple_60(liquid_assets=80_000, pension_balance=20_000, annual_spending=4800,
                 lifestyle=1800, legacy_target=30_000)
from retire_sim.reporting import post_retirement_average
P = {"age": 60, "retire_age": 60}
STRAT = [(t("none"), {}), (t("isa"), dict(use_isa=True)), (t("pen"), dict(use_pension=True)),
         (t("gift"), dict(gift_per_child_10y=5000)),
         (t("all3"), dict(use_isa=True, use_pension=True, gift_per_child_10y=5000))]
names, stacks, probs = [], [], []
for lab, kw in STRAT:
    r = engine_tax.run(rich, CFG, e=ECO, tc=replace(TC, **kw), qx_table=QX)
    names.append(lab)
    stacks.append([post_retirement_average(r, P, k) for k in ("tax_y", "hi_y", "prop_y")])
    probs.append(engine_tax.summarize(r)["기본생활 유지 확률"] * 100)

fig, ax = plt.subplots(figsize=(9.5, 4.4))
bottom = [0] * len(names)
for i, (key, color) in enumerate((("tax", TEAL), ("hi", "#E8A33D"), ("prop", MUTED))):
    vals = [st[i] for st in stacks]
    ax.bar(names, vals, bottom=bottom, label=t(key), color=color, width=.55)
    bottom = [b + v for b, v in zip(bottom, vals)]
for i, (b, pr) in enumerate(zip(bottom, probs)):
    ax.text(i, b * 1.03, f"{pr:.0f}%", ha="center", color=INK, fontsize=11, fontweight="bold")
ax.set(title=t("taxfig"), ylabel="만 원 / 년" if KO else "10k KRW per year")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(IMG, "tax_strategies.png"), dpi=130, facecolor="white")
print("tax_strategies.png 갱신:", [(n, round(p, 1)) for n, p in zip(names, probs)])

# ── ③ 민감도: 무엇이 결과를 가장 크게 흔드나 ─────────────────────────────
L.update({"sens": ("가정을 바꿨을 때 기본생활 지킬 확률이 움직이는 폭",
                   "How far each assumption moves essential success"),
          "erp": ("주식 위험프리미엄 3%↔5%", "Equity premium 3%/5%"),
          "vol": ("주식 변동성 15%↔21%", "Equity vol 15%/21%"),
          "rate": ("장기 금리 2%↔4%", "Long-run rate 2%/4%"),
          "infl": ("장기 물가 1.5%↔2.5%", "Long-run inflation 1.5%/2.5%"),
          "spend": ("생활비 −10%↔+10%", "Spending −10%/+10%"),
          "wt": ("주식 비중 20%↔60%", "Equity weight 20%/60%"),
          "care": ("간병비 상승 0↔2%p", "Care inflation 0/2pp")})

p0 = s["기본생활 유지 확률"] * 100
pr = lambda **kw: engine_tax.summarize(
    engine_tax.run(kw.pop("h", couple_60()), kw.pop("cfg", CFG), e=kw.pop("e", ECO),
                   tc=TC, qx_table=QX))["기본생활 유지 확률"] * 100
tests = [
    (t("erp"), pr(e=replace(ECO, erp=0.03)), pr(e=replace(ECO, erp=0.05))),
    (t("vol"), pr(e=replace(ECO, s_sigma=0.15)), pr(e=replace(ECO, s_sigma=0.21))),
    (t("rate"), pr(e=replace(ECO, q_theta=0.0)), pr(e=replace(ECO, q_theta=0.02))),
    (t("infl"), pr(e=replace(ECO, pi_theta=0.015)), pr(e=replace(ECO, pi_theta=0.025))),
    (t("spend"), pr(h=couple_60(annual_spending=3240, essential=2700)),
                 pr(h=couple_60(annual_spending=3960, essential=3300))),
    (t("wt"), pr(h=couple_60(stock_weight=0.2)), pr(h=couple_60(stock_weight=0.6))),
    (t("care"), pr(cfg=SimConfig(care=CareMarkov(), care_real_growth=0.0)),
                pr(cfg=SimConfig(care=CareMarkov(), care_real_growth=0.02))),
]
tests.sort(key=lambda x: abs(x[2] - x[1]))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (n, a, b) in enumerate(tests):
    ax.barh(i, a - p0, left=p0, color=CORAL, height=.5)
    ax.barh(i, b - p0, left=p0, color=TEAL, height=.5)
ax.axvline(p0, color=INK, lw=1)
ax.set_yticks(range(len(tests)), [x[0] for x in tests])
ax.set(title=t("sens"), xlabel="기본생활 지킬 확률 (%)" if KO else "Essential success (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "sensitivity.png"), dpi=130, facecolor="white")
print("sensitivity.png 갱신 (폭 작은 순):", [(n, round(abs(b - a), 1)) for n, a, b in tests])
