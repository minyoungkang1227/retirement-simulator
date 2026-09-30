# Nohu Compass — Probabilistic Retirement Simulator (Korea)

[한국어](README.md) | **English**

**An actuarial, probability-based retirement simulator built around Korea's pension, tax and health-insurance rules — asking not "how much have I saved?" but "can I sustain the retirement I want?"**

It evaluates three goals — essential living, lifestyle (travel, hobbies) and legacy — across 10,000 economic and longevity scenarios, and compares **what changes bring you closer to your goals**. Results are comparisons on the same scenarios, not recommendations.

- **Web app (no install, Korean UI):** https://<app-url>.streamlit.app
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/minyoungkang1227/retirement-simulator/blob/main/notebooks/retire_sim_colab.ipynb) (analysis notebook)

## Motivation

Most Korean retirement calculators fix returns and lifespan and ignore taxes and health-insurance premiums, so results are optimistic. International tools (Boldin, ProjectionLab) use Monte Carlo simulation but US rules. This project combines **Korean institutions, stochastic models and actuarial models** to fill that gap.

## How it works

```
① Step-by-step input: retirement timing → assets → essential spending → lifestyle budget → pensions → legacy → risk profile → final check
② 10,000 scenarios: rates, inflation and equities (linked stochastic processes) × each spouse's lifespan and long-term care (multi-state model)
③ Cash flow: savings before retirement; after retirement, essential spending first and lifestyle only when affordable
④ Taxes, health premiums, property taxes and inheritance tax
⑤ Output: income gap, goal-by-goal success, required assets, what-if comparisons, add-on products (stocks, deposits, annuities, insurance)
```

## What makes it different

1. **Goal-based structure** — essential (protected), lifestyle (flexible) and legacy goals, with a guardrail that cuts flexible spending first in bad scenarios (floor = survival-weighted present value of expected essential shortfalls)
2. **Korean institutions** — National Pension early/deferred claiming, KRW 15M private-pension threshold, comprehensive financial-income taxation, health premiums (regional/workplace, dependents), property taxes, inheritance and gift taxes
3. **Actuarial modeling** — joint-life survival from the Statistics Korea 2024 complete life table, multi-state long-term-care Markov model (healthy → mild → severe → dead, calibrated to national long-term-care statistics) calibrated to preserve life-table mortality
4. **Stochastic economy** — OU real rate and inflation estimated from Bank of Korea (ECOS) data, with a Fisher link, two-factor closed-form bond pricing, exact joint distribution of integrated rates, crash jumps, parameter uncertainty, fees
5. **Statistically honest comparisons** — paired-difference confidence intervals on common random numbers decide "different / not different"

## Example (couple aged 52 and 50, retiring at 60, KRW 500M, saving KRW 1.5M/month · placeholder assumptions)

- Essential KRW 3M/month + lifestyle KRW 12M/year − pensions KRW 1.6M/month → **income gap KRW 2.4M/month**
- Goal success: essential **39%**, lifestyle funding **53%**, KRW 100M legacy **28%**

| What if | Change in essential success (95% CI) |
|---|---|
| Retire 2 years later | **+12.8pp** (±0.7) |
| Spend 10% less | +10.2pp (±0.6) |
| Growth profile | +7.3pp (±0.5) |
| Save KRW 0.5M more per month | +4.9pp (±0.4) |
| Conservative profile | −11.9pp (±0.6) |

| Crisis | Change in essential success |
|---|---|
| Equities −40% in the first year of retirement | −15.4pp |
| Equities −40% ten years into retirement | −10.3pp |
| Three years of 6% inflation | −5.7pp |
| Longer life (mortality −20%) | −3.5pp |
| Long-term care incidence doubled | −1.0pp |

- For this household **retirement timing is the strongest lever**, and the same crash hurts most right after retirement (sequence-of-returns risk).
- Splitting spending into essential + flexible lowers the depletion probability versus fixed spending of the same total (61.9% → 55.4%).

![goals](docs/images/goals_whatif.png)
![tax strategies](docs/images/tax_strategies.png)
![sensitivity](docs/images/sensitivity.png)

## Validation

- Exact OU/Vasicek discretization matches theoretical moments (rate 2.99%/1.84% vs 3.00%/1.83%)
- Bond prices: Vasicek closed form 0.87706 vs Monte Carlo 0.87723; two-factor 0.87553 vs 0.87518
- Integrated-rate 5×5 joint covariance: max correlation error 0.005 vs fine-step simulation
- Mortality: Statistics Korea 2024 complete life table; model life expectancy matches published values (ages 0, 60, 65, 80, both sexes)
- Multi-state care model preserves male life expectancy at 60 (table 23.7 → 23.9 years)
- OU maximum likelihood recovers long-run mean and volatility on synthetic data (small-sample upward bias in mean-reversion speed confirmed and corrected for real data)
- Rates and inflation estimated from ECOS monthly data 2000-01 to 2026-08 (inflation long-run mean 2.47% ±1.02pp, real rate 0.73% ±0.74pp)
- Paired-difference CIs for comparisons (±1.38pp under independence → ±0.44pp)

## Structure

```
streamlit_app.py   web app (step-by-step input → goal results → what-if)
retire_sim/        model package
  config.py        input schema (three goals, accumulation phase, care model)
  economy_v2.py    rate, inflation, equity and bond scenarios (Fisher link, integrated rates, jumps, shocks)
  mortality.py     mortality and multi-state care Markov model
  engine_tax.py    tax, accounts, housing, business income and goal guardrail engine
  tax.py           Korean tax and health-insurance rules (as of 2026-09)
  calibrate.py     OU maximum likelihood and ECOS data helpers
  market_data.py   ticker → beta, correlation, rate sensitivity
  metrics.py       depletion probability, confidence intervals, paired comparisons
notebooks/         Colab analysis notebook
scripts/           validation and sensitivity scripts
docs/              equations and design notes (MODEL.md / MODEL.en.md), figures
CHANGELOG.md       version history (what / how / what changed)
```

## Usage

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py      # web app
python scripts/validate_v11.py      # validation
```

## Status

| Component | Status |
|---|---|
| Cash-flow engine, couple mortality, rate/inflation SDEs, validation | Done |
| Shock tests, sensitivity, confidence intervals | Done |
| Taxes, health premiums, property, inheritance/gift, business income (approximate) | Done |
| v10 enhancements (Fisher link, fees, parameter uncertainty, crash jumps, antithetic variates) | Done |
| v11 refinements (paired CIs, integrated rates, multi-state care, term premium, OU MLE) | Done |
| Web app (Streamlit) | Done |
| v12 goal-based rebuild (three goals, income gap, accumulation phase, model portfolios, step-by-step input) | Done |
| Statistics Korea 2024 complete life table (v13) | Done |
| Long-term-care model from real data v17: two severity levels calibrated to national LTC recognition rates and grade mix, 2026 out-of-pocket and caregiver costs | Done |
| Actuarial essential floor v16: survival-weighted present value reflecting pension start and couple survival states | Done |
| Add-on products v15: stocks by ticker (beta, correlation, rate sensitivity), deposits, life annuities, long-term-care insurance, pension contributions | Done |
| ECOS rate/inflation estimation (v14) | Done |
| Cohort mortality improvement, real data for equity returns | Planned |
| Home sale and reverse mortgage | Planned |
| Account/budget-app import, periodic re-measurement | Planned |

See [CHANGELOG.md](CHANGELOG.md) and [docs/MODEL.en.md](docs/MODEL.en.md) for details.

## Limitations

- Mortality uses the Statistics Korea 2024 period life table (no future improvement yet); rates and inflation are estimated from ECOS 2000–2026 data. Long-term care is calibrated to national statistics; only equity returns are still **placeholders**. Focus on **differences between choices**.
- Tax rules are simplified (Korea, as of 2026-09). **Not investment or tax advice.**
- Web-app inputs are used only for calculation and are not stored.

## Contact

- Email: doongss1@naver.com

## Copyright

© 2026 Minyoung Kang. All rights reserved. For usage inquiries, please contact me by email. See [COPYRIGHT](COPYRIGHT).
