import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""간병 발생률·진행률 보정 (v17, 목표 갱신 v56).

목표 연령별 유병률(장기요양 인정자 / 의료보장 적용인구, 2023, 국민건강보험공단 「노인장기요양보험통계연보」(2024 발간)
  — 보험연구원 송윤아·정수진(2024) 「보험회사의 노인주거서비스 제공 방안」 <표 II-4>에서 인용):
  65~69 2.7%, 70~74 5.8%, 75~79 12.9%, 80~84 29.8%, 85세 이상 41.4% (65세 이상 전체 10.8%)
  85세 이상은 통계연보가 한 구간이라, 모델의 85~89 / 90+ 두 구간은 v17 추정치(42.3 / 55.4%)의 비율을 유지한 채
  인구 가중(약 7:3)으로 41.4%가 되도록 37.9% / 49.6%로 나눴다(판단값). 60~64세는 v17 추정 0.3% 유지.
  v17~v55까지는 2022 장기요양실태조사의 인정자 연령 분포 × 2025 인정자 수 ÷ 2025 인구로 역산한 추정치
  (65~69 1.5%, 70~74 3.9%, 75~79 9.9%, 80~84 24.0%)를 썼는데, 공식 표보다 65~79세가 낮았다.
목표 중증 비율: 1~2등급 12.7% (2025)
방법: 60세 남녀 코호트를 시뮬레이션해 연령대별 유병률과 중증 비율이 목표에 맞도록 발생률·진행률을 반복 조정.
"""
import numpy as np
from dataclasses import replace
from retire_sim.config import CareMarkov
from retire_sim import mortality

groups = [(60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 100)]
target = [0.003, 0.027, 0.058, 0.129, 0.298, 0.379, 0.496]
target_sev = 0.127
q = mortality.default_qx()

def simulate(c, n=60000):
    prev = np.zeros(len(groups)); sev_num = sev_den = 0; dur = []
    for sex in ("M", "F"):
        al, ic, sv = mortality.simulate_life(q[sex], 60, 45, n, np.random.default_rng(1 if sex == "M" else 2), c, return_severe=True)
        for g, (lo, hi) in enumerate(groups):
            cols = slice(lo - 60, hi - 60)
            prev[g] += ic[:, cols].sum() / max(al[:, cols].sum(), 1) / 2
        sev_num += sv[:, 5:].sum(); sev_den += ic[:, 5:].sum()
        ever = ic.any(1); dur.append(ic.sum(1)[ever].mean())
    return prev, sev_num / sev_den, float(np.mean(dur))

c = CareMarkov()
inc = [v for _, v in c.incidence]
for it in range(25):
    c = replace(c, incidence=tuple(zip([65, 70, 75, 80, 85, 90, 999], inc)))
    prev, sev, dur = simulate(c)
    inc = [min(max(v * (tg / max(pv, 1e-6)) ** 0.7, 1e-4), 0.6) for v, tg, pv in zip(inc, target, prev)]
    c = replace(c, progress=float(np.clip(c.progress * (target_sev / sev) ** 0.8, 0.005, 0.5)))
print("발생률:", [round(v, 4) for v in inc])
print("진행률:", round(c.progress, 4))
print("유병률 모델 vs 목표:", [(round(a, 3), b) for a, b in zip(prev, target)])
print(f"중증 비율 {sev:.3f} (목표 {target_sev}), 간병 경험자 평균 기간 {dur:.1f}년")
