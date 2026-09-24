# Probabilistic Retirement Simulator (Korea)

[한국어](README.md) | **English**

**An actuarial, probability-based retirement simulator built around Korea's pension, tax, and health-insurance rules.**

Instead of a fixed-return calculation, it answers "Will my retirement savings last?" with **10,000 economic and longevity scenarios**.
Results are presented as **comparisons between choices on the same scenarios**, not as recommendations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<GITHUB_ID>/retirement-simulator/blob/main/notebooks/retire_sim_colab.ipynb)

## Motivation

Most retirement calculators offered by Korean banks and insurers fix both returns and lifespan. International tools (Boldin, ProjectionLab) use Monte Carlo simulation, but their tax and pension rules are US-specific. This project fills the gap between the two.

## What makes it different

1. **Korean institutions** — National Pension early/deferred claiming, the KRW 15M private-pension separate-taxation threshold, comprehensive financial-income taxation, National Health Insurance premiums (regional/workplace, dependent eligibility), property and comprehensive real-estate holding taxes, inheritance and gift taxes
2. **Actuarial modeling** — joint-life survival of couples from life tables, long-term-care cost jumps, mortality credits
3. **Stochastic economy** — Vasicek short rate and Ornstein–Uhlenbeck inflation with exact discretization, closed-form bond pricing, rate-linked equity returns
4. **Choice comparison** — pension claiming age, ISA and pension-account strategies, ownership splitting, and gifting strategies compared with common random numbers

## Structure

```
Inputs → Economic scenarios (Vasicek, OU, equity, bonds) → Mortality (couple)
       → Institutions & tax module → Cash-flow engine (10,000 paths, 3 accounts + housing) → Metrics
       → Choice comparison / shock tests / sensitivity
```

```
retire_sim/        model package
  config.py        input schema
  economy_v2.py    rate, inflation, equity and bond scenarios (shock inputs supported)
  mortality.py     mortality (placeholder Gompertz; Statistics Korea life-table CSV loader)
  engine.py        base cash-flow engine
  engine_tax.py    tax, account, housing and business-income engine
  tax.py           Korean tax and health-insurance rules (as of 2026-09)
  metrics.py       depletion probability, confidence intervals, fan charts
notebooks/         Colab notebook (edit only the input cell)
scripts/           validation, sensitivity and example scripts
docs/              equations and design notes (MODEL.md / MODEL.en.md), figures
```

## Usage

**Colab:** click the badge above and select `Runtime > Run all`. Change only the numbers in the Step 2 input cell. The notebook is in Korean; its first cell has an English guide. Amounts are in units of KRW 10,000 (만원).

**Local:**
```bash
pip install -r requirements.txt
python scripts/validate.py      # checks against theoretical values
python scripts/sensitivity.py   # sensitivity analysis
```

## Example results
Couple aged 60 and 58, KRW 500M in financial assets, KRW 3M monthly spending. Assumptions are placeholders.

| National Pension start age | Depletion probability (±1.0pp) |
|---|---|
| 60 | 46.6% |
| 65 | 39.9% |
| 70 | 40.4% |

| Shock | Depletion probability |
|---|---|
| Baseline | 39.9% |
| Equities −40% in the first year of retirement | 65.3% |
| Equities −40% ten years later | 55.0% |

- The same crash is far more damaging **right after retirement** (sequence-of-returns risk).
- The most sensitive input is **spending**, not market assumptions (±10% → 19.7% to 62.4%).
- Adding taxes, health premiums and holding taxes (example: one home with a KRW 600M assessed value) raises depletion probability from 39.9% to 69.5%; ISA plus pension accounts bring it to 66.7%, while concentrating financial assets in one spouse's name raises it to 71.2%.

![results](docs/images/v2_results.png)
![sensitivity](docs/images/sensitivity.png)
![tax strategies](docs/images/tax_strategies.png)

## Validation

- Exact discretization of OU/Vasicek: long-run mean and standard deviation match theory (rate 2.99% / 1.84% vs 3.00% / 1.83%)
- Vasicek bond price: closed form 0.87706 vs Monte Carlo 0.87723
- Every probability comes with a 95% confidence interval; choice comparisons use common random numbers

## Status

| Component | Status |
|---|---|
| Cash-flow engine, couple mortality, care costs, rate/inflation SDEs, validation | Done |
| Shock tests, sensitivity analysis, confidence intervals | Done |
| Taxes, health premiums, holding taxes, inheritance/gift tax, business income (approximate) | Done |
| Parameters from Statistics Korea life tables and Bank of Korea (ECOS) data | Planned |
| Insurance module (life annuities, long-term-care insurance), home sale and reverse mortgage | Planned |
| Accumulation phase for people in their 40s, account/budget-app import, periodic re-measurement | Planned |

## Limitations

- Return, inflation, mortality and care assumptions are **placeholders**. Focus on **differences between choices** rather than absolute levels.
- Tax rules are simplified and reflect Korean law as of 2026-09. **This is not tax or investment advice**; consult a professional before making decisions.
- See [docs/MODEL.en.md](docs/MODEL.en.md) for equations and limitations.

## Contact

For questions, feedback, collaboration or licensing inquiries, please reach out by email.

- Email: <EMAIL>

## Copyright

© 2026 Minyoung Kang. All rights reserved.
The code, documents and figures in this repository are protected by copyright. You are welcome to view them, but copying, modifying, distributing or using them commercially without the author's prior written consent is prohibited. See [COPYRIGHT](COPYRIGHT).
