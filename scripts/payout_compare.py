import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""(v59) 퇴직급여를 받는 방법별 결과 — 일시금 / 10년 / 20년 / 종신연금 / 모자라는 만큼만.

실행: PYTHONPATH=. python scripts/payout_compare.py
같은 1만 개 미래(같은 난수) 위에서 받는 방법만 바꿔 비교한다. docs/MODEL.md §퇴직급여 받는 방법의 표를 만든다.
"""
from retire_sim import __version__, advice
from retire_sim.inputs import defaults

CASES = {
    "52·50세 부부, 연봉 7,000만·근속 20년 (DB)": dict(salary=7000, ret_contrib=True, ret_plan="DB", tenure=20),
    "35·33세 맞벌이, 연봉 5,000만+3,600만·근속 5년 (DB)": dict(age=35, s_age=33, salary=5000, s_salary=3600, ret_contrib=True,
                                                    ret_plan="DB", tenure=5, s_tenure=3, nps_auto=True,
                                                    nps_years=7, s_nps_years=5),
}

if __name__ == "__main__":
    print(f"모델 v{__version__} · 1만 경로")
    for title, over in CASES.items():
        p = defaults(); p.update(paths=10_000, **over)
        print(f"\n{title}")
        print(f"  {'받는 방법':<16}{'기본생활 유지':>10}{'일시금 대비':>10}{'나쁜 30% 평균 부족액':>16}")
        for r in advice.payout_compare(p):
            print(f"  {r['name']:<16}{r['prob'] * 100:>9.1f}%{r['delta'] * 100:>+9.1f}%p{r['cte70'] / 10000:>13.2f}억")
