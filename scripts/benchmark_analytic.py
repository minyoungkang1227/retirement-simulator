"""해석해 대조: 변동성·세금·사망·간병을 모두 끈 결정론적 극한에서
   W_{t+1} = (W_t − 지출_t) × (1+r) 와 정확히 일치하는지 확인.

설정은 retire_sim/verify.py 의 analytic_rows() 한 곳에 있다(v48.1) — 앱의 검증 장부와 같은 계산이다."""
from retire_sim.verify import analytic_rows

rows = analytic_rows(12)
print(f"{'연차':>4} {'모형':>12} {'닫힌식':>12} {'차이':>10}")
for t, a, b, d in rows:
    print(f"{t:>4} {a:12,.1f} {b:12,.1f} {d:10.4f}")
print("최대 오차:", max(abs(d) for *_, d in rows))
