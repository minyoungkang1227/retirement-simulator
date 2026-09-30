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
① Step-by-step input (from age 20): retirement timing → assets, income and savings → essential spending → lifestyle budget → pensions → legacy → risk profile → final check
② 10,000 scenarios: rates, inflation and equities (linked stochastic processes) × each spouse's lifespan and long-term care (multi-state model)
③ Cash flow: savings before retirement; after retirement, essential spending first and lifestyle only when affordable
④ Taxes, health premiums, property taxes and inheritance tax
⑤ Output: income gap, goal-by-goal success, required assets and required savings, what-if comparisons (including reverse mortgage and downsizing), add-on products (stocks, deposits, annuities, insurance)
```

## What makes it different

1. **Goal-based structure** — essential (protected), lifestyle (flexible) and legacy goals, with a guardrail that cuts flexible spending first in bad scenarios (floor = survival-weighted present value of expected essential shortfalls)
2. **Korean institutions** — National Pension early/deferred claiming, KRW 15M private-pension threshold, comprehensive financial-income taxation, health premiums (regional/workplace, dependents), property taxes, inheritance and gift taxes
3. **Actuarial modeling** — joint-life survival from the Statistics Korea 2024 complete life table, multi-state long-term-care Markov model (healthy → mild → severe → dead, calibrated to national long-term-care statistics) calibrated to preserve life-table mortality
4. **Stochastic economy** — OU real rate and inflation estimated from Bank of Korea (ECOS) data, with a Fisher link, two-factor closed-form bond pricing, exact joint distribution of integrated rates, crash jumps, parameter uncertainty, fees
5. **Statistically honest comparisons** — paired-difference confidence intervals on common random numbers decide "different / not different"

## Example (couple aged 52 and 50, retiring at 60, KRW 500M, saving KRW 1.5M/month · placeholder assumptions)

- Essential KRW 3M/month + lifestyle KRW 12M/year − pensions KRW 1.6M/month → **income gap KRW 2.4M/month**
- Goal success: essential **48%**, lifestyle funding **60%**, KRW 100M legacy **37%**

| What if | Change in essential success (95% CI) |
|---|---|
| Retire 2 years later | **+11.8pp** (±0.6) |
| Spend 10% less | +9.2pp (±0.6) |
| Growth profile | +6.4pp (±0.5) |
| Save KRW 0.5M more per month | +4.8pp (±0.4) |
| Conservative profile | −13.9pp (±0.7) |

| Crisis | Change in essential success |
|---|---|
| Equities −40% in the first year of retirement | −15.7pp |
| Equities −40% ten years into retirement | −10.4pp |
| Three years of 6% inflation | −5.6pp |
| Longer life (mortality −20%) | −3.6pp |
| Long-term care incidence doubled | −1.0pp |

- For this household **retirement timing is the strongest lever**, and the same crash hurts most right after retirement (sequence-of-returns risk).
- Splitting spending into essential + flexible lowers the depletion probability versus fixed spending of the same total (53.9% → 47.6%).
- The largest uncertainty is the **equity risk premium**: within its estimated range (5.2% ± 1.4pp) essential success moves between 36% and 59%, which is why differences between choices matter more than levels.

**Young savers (dual-income couple aged 30 and 29, KRW 90M gross income, 15% saving rate, workplace pension, National Pension estimated automatically at KRW 2.43M/month)**

- Essential success **82%**, lifestyle funding **91%**
- To keep essentials 9 times out of 10: save **23%** of gross income (about KRW 1.71M/month in year one)
- What if: save KRW 0.5M more per month +8.1pp, retire 2 years later +6.7pp, spend 10% less +6.1pp · without the workplace pension −23.0pp

**House-rich, cash-poor household (couple aged 65 and 63, KRW 300M financial assets, home worth KRW 700M)**

| Housing strategy | Essential success | Lifestyle funding | KRW 300M legacy | Note |
|---|---|---|---|---|
| Keep as is | 13% | 33% | 100% | — |
| Reverse mortgage now (63) | 70% (+56.5pp) | 94% | 43% | KRW 1.65M/month |
| Reverse mortgage at 70 | 77% (+63.7pp) | 91% | 47% | KRW 2.15M/month (today's value) |
| Downsize to half price at 70 | 71% (+57.9pp) | 88% | 99% | KRW 340M released |

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
- Equities: volatility, crash frequency and rate correlation estimated from KOSPI monthly data 2000-02 to 2026-08; the risk premium combines the data with an external prior (Damodaran) in a Bayesian way
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
| Accumulation phase for ages 20–40s v20: income, saving rate and wage growth, workplace pension, automatic National Pension estimate (2026 reform), required-saving solver | Done |
| Housing v19: reverse mortgage (HF 2026 payment table, loan balance, non-recourse, property-tax relief) and downsizing (capital-gains and acquisition taxes) | Done |
| Equity returns from real data v18: KOSPI volatility, crash jumps, rate correlation, Bayesian risk premium | Done |
| Long-term-care model from real data v17: two severity levels calibrated to national LTC recognition rates and grade mix, 2026 out-of-pocket and caregiver costs | Done |
| Actuarial essential floor v16: survival-weighted present value reflecting pension start and couple survival states | Done |
| Add-on products v15: stocks by ticker (beta, correlation, rate sensitivity), deposits, life annuities, long-term-care insurance, pension contributions | Done |
| ECOS rate/inflation estimation (v14) | Done |
| Cohort mortality improvement | Planned |
| Account/budget-app import, periodic re-measurement | Planned |

See [CHANGELOG.md](CHANGELOG.md) and [docs/MODEL.en.md](docs/MODEL.en.md) for details.

## Limitations

- Mortality uses the Statistics Korea 2024 period life table (no future improvement yet); rates and inflation are estimated from ECOS 2000–2026 data. Long-term care is calibrated to national statistics and equities to KOSPI data. No placeholders remain, but the **equity risk premium is uncertain** (±1.4pp). Focus on **differences between choices**.
- Tax rules are simplified (Korea, as of 2026-09). **Not investment or tax advice.**
- Web-app inputs are used only for calculation and are not stored.

## Contact

- Email: doongss1@naver.com

## Copyright

© 2026 Minyoung Kang. All rights reserved. For usage inquiries, please contact me by email. See [COPYRIGHT](COPYRIGHT).
