# Nohu Compass — Probabilistic Retirement Simulator (Korea)

> ### 👉 [**Try it now — retire-compass.streamlit.app**](https://retire-compass.streamlit.app)
> No input needed: press an **example household** button to see results first. Scan the QR code on your phone to open it. (If nobody has used the app for a while, the first visit may show a "wake up" button — press it and the app opens within ~30 seconds. The app is in Korean.)
>
> <img src="docs/images/qr_retire_compass.png" width="140" alt="QR code">


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
⑤ Output: income gap, goal-by-goal success, required assets and required savings, interactive what-if sliders (retirement timing, savings, spending, pension start, risk profile) with change curves, reverse mortgage and downsizing comparisons, add-on products (stocks, deposits, annuities, insurance)
```

## What makes it different

1. **Goal-based structure** — essential (protected), lifestyle (flexible) and legacy goals, with a guardrail that cuts flexible spending first in bad scenarios (floor = survival-weighted present value of expected essential shortfalls)
2. **Korean institutions** — National Pension early/deferred claiming, KRW 15M private-pension threshold, comprehensive financial-income taxation, health premiums (regional/workplace, dependents), property taxes, inheritance and gift taxes
3. **Actuarial modeling** — Statistics Korea 2024 complete life table with cohort mortality improvement (0.3–2.5% per year by age, validated against the 2013–2024 trend), joint-life survival, multi-state long-term-care Markov model, and annuity pricing by the equivalence principle (single/joint, guarantee periods, escalation, deferred) calibrated to preserve life-table mortality
4. **Stochastic economy** — OU real rate and inflation estimated from Bank of Korea (ECOS) data, with a Fisher link, two-factor closed-form bond pricing, exact joint distribution of integrated rates, crash jumps, parameter uncertainty, fees
5. **Statistically honest comparisons** — paired-difference confidence intervals on common random numbers decide "different / not different"

> **v36 (2026-10-14):** Three model refinements. **Care costs track wages, not consumer prices** — care is labour and admits no productivity growth (long-term-care fee schedule rose 4.05%/yr over 2022–2026 against 3.02% CPI, a 1.0pp gap). Compounded over a long horizon this more than **doubles** the tail risk for the younger reference household (CTE90 KRW 287M → 614M): the model had been systematically understating long-horizon care costs. Also added **double care** (a dependent parent's need for care simulated probabilistically, not as a fixed expense) and optional **behavioural assumptions** (panic selling after a crash, downward-sticky spending).

> **v35 (2026-10-14):** Added an **HTTP API** (`retire_sim/api.py`) so an institution can embed the engine, and **white-labelling** (name, colours, logo, disclaimer). The larger piece of work was what came first: input mapping, lever ranking and report assembly all lived inside the Streamlit script, so nothing could run without the UI. They now live in `inputs.py`, `advice.py` and `reporting.py`, shared by the app and the API — and a test fails if the app ever grows its own copy. Input validation was added so a bad value is rejected with a clear message (HTTP 422) instead of blowing up inside the engine.

> **v34 (2026-10-14):** Real annuity products can now be compared **using the numbers from their own disclosure** (`retire_sim/product.py`): credited rate (a smoothed follow of market rates, floored by the **minimum guaranteed rate**), acquisition and maintenance loadings, surrender charges, and annuitisation. The headline finding, for a 45-year-old paying KRW 0.5M/month for 20 years: the accumulated balance is **lowest for the annuity** (KRW 182M vs 210M on deposit and 255M in a fund), yet the probability of **running out of money while still alive is 0% for the annuity, 16% for the fund and 34% for the deposit**. What an annuity sells is not return but the transfer of longevity risk — comparing balances alone systematically undervalues it. The minimum guaranteed rate behaves as a **put on interest rates**: worth KRW 43k/year on average but KRW 169k/year in the worst decile of paths.

> **v33 (2026-10-13):** Added the layer that makes the model usable rather than more model: save/restore of a client's inputs, a **reproducible calculation ID** hashed from inputs plus model version, a **two-page advisory report** (self-contained HTML; print to PDF from the browser), and a **waterfall chart** showing where the money actually goes. In the B2B2C setting the unit of work is one advisory session and its output is a report, not a screen.

> **v32 (2026-10-13):** Replaced three approximations flagged as limitations in v31. The basic pension's offset against the national pension now follows the **statutory formula** (the larger of the 250% rule and the A-benefit rule, floored at the supplementary amount) plus the income-reversal adjustment; the property deduction is now a **three-way regional choice**; and the widowhood effect is recalibrated to the **Moon et al. (2011) meta-analysis** (2.26m people, age-adjusted). The Korean figure often quoted (widowed men die at 4.2x the rate of married men) is a crude rate with no age adjustment, so it was not used.

> **v31 (2026-10-13):** Instead of adding to the model, this release **removed complexity that did not change the answer and proved the arithmetic**. In a deterministic limit the engine now matches the closed form to a relative error of 1e-7 (`scripts/benchmark_analytic.py`); a component ablation (`scripts/ablate.py`) found one of the two economy generators made no difference beyond Monte Carlo noise, so it was deleted. Model code 1,874 → 1,643 lines (−12%), 14 → 11 modules. Error budget (`scripts/error_budget.py`): Monte Carlo noise ±0.7pp vs 14pp from a 1σ move in the equity risk premium — **read the result as a range, not a single number.**

## Example (model v33 · 10,000 paths · reproduce with `scripts/readme_numbers.py`)

**Couple aged 52 and 50, retiring at 60, KRW 500M financial assets, saving KRW 1.5M/month** (essential KRW 3M/month + lifestyle KRW 12M/year, pensions KRW 1.6M/month, KRW 100M legacy)

- Goal success: essential **56%**, lifestyle funding **66%**, KRW 100M legacy **42%**

| What if | Change in essential success (95% CI) |
|---|---|
| Retire 2 years later | **+10.2pp** (±0.6) |
| Spend 10% less | +8.0pp (±0.5) |
| Growth profile (70% equity) | +4.4pp (±0.5) |
| Save KRW 0.5M more per month | +4.1pp (±0.4) |
| Conservative profile (30% equity) | −12.3pp (±0.7) |

| Crisis | Change in essential success |
|---|---|
| Equities −40% in the first year of retirement | −15.0pp |
| Equities −40% ten years into retirement | −10.0pp |
| Three years of 6% inflation | −5.2pp |
| Longer life (mortality −20%) | −3.6pp |
| Long-term care incidence doubled | −3.0pp |

- For this household **retirement timing is the strongest lever**, and the same crash hurts most right after retirement (sequence-of-returns risk).
- The largest uncertainty is the **equity risk premium**: within its estimated range (5.2% ± 1.4pp) essential success moves between **45% and 66%**, while Monte Carlo noise is ±0.7pp — about one twentieth of it. That is why differences between choices matter more than levels.

**Young savers (dual-income couple aged 30 and 29, KRW 90M gross income, 15% saving rate, workplace pension, National Pension estimated automatically)**

- Essential success **83%**, lifestyle funding **92%**, KRW 100M legacy **75%**
- What if: save KRW 0.5M more per month +6.6pp · retire 2 years later +5.8pp · spend 10% less +5.1pp
- **Without the workplace pension −24.2pp** — at this age nothing else comes close.

**House-rich, cash-poor household (couple aged 65 and 63, KRW 300M financial assets, home worth KRW 700M, essential KRW 2.5M/month + lifestyle KRW 12M/year, KRW 300M legacy)**

| Housing strategy | Essential success | Lifestyle funding | KRW 300M legacy | Note |
|---|---|---|---|---|
| Keep as is | 23% | 32% | 95% | — |
| Reverse mortgage now (63) | 83% (+59.7pp) | 97% | 70% | KRW 1.65M/month |
| Reverse mortgage at 70 | 88% (+65.1pp) | 89% | 74% | KRW 2.25M/month (today's value) |
| Downsize to half price at 70 | 79% (+56.3pp) | 87% | 86% | KRW 350M released |

- When the home is most of the balance sheet, **whether to use it is the biggest single choice**. A reverse mortgage buys retirement security; downsizing protects the bequest as well.

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
- House prices: Korea Real Estate Board index via ECOS (2013 onwards) — smoothing-corrected volatility 10%, real growth 1.25% ± 1.18pp
- Rates and inflation estimated from ECOS monthly data 2000-01 to 2026-08 (inflation long-run mean 2.47% ±1.02pp, real rate 0.73% ±0.74pp)
- Paired-difference CIs for comparisons (±1.38pp under independence → ±0.44pp)
- 15 regression tests run on every commit (headline ranges, directional checks, theoretical values, life table, care model, reverse-mortgage table)

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
tests/             regression tests (pytest, run automatically by GitHub Actions)
docs/              equations and design notes (MODEL.md / MODEL.en.md), figures
CHANGELOG.md       version history (what / how / what changed)
```

## Usage

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py      # web app
python scripts/validate_v11.py      # theory checks
pytest -q                           # regression tests
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
| Interactive what-if v22: combine slider changes, before/after comparison, change curves, reading guide | Done |
| Big-expense planning v21: education, wedding support, home purchase (amortizing mortgage, jeonse deposit, acquisition tax), pre-retirement shortfalls as loans | Done |
| Accumulation phase for ages 20–40s v20: income, saving rate and wage growth, workplace pension, automatic National Pension estimate (2026 reform), required-saving solver | Done |
| Actuarial refinements v27: cohort mortality improvement, CTE70/90 and CRRA certainty equivalent, annuity product variants | Done |
| House price risk v26: stochastic house prices estimated from the national index (smoothing correction, Bayesian real growth), reflected in property taxes, reverse mortgage and inheritance | Done |
| Housing v19: reverse mortgage (HF 2026 payment table, loan balance, non-recourse, property-tax relief) and downsizing (capital-gains and acquisition taxes) | Done |
| Equity returns from real data v18: KOSPI volatility, crash jumps, rate correlation, Bayesian risk premium | Done |
| Long-term-care model from real data v17: two severity levels calibrated to national LTC recognition rates and grade mix, 2026 out-of-pocket and caregiver costs | Done |
| Actuarial essential floor v16: survival-weighted present value reflecting pension start and couple survival states | Done |
| Add-on products v15: stocks by ticker (beta, correlation, rate sensitivity), deposits, life annuities, long-term-care insurance, pension contributions | Done |
| ECOS rate/inflation estimation (v14) | Done |
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
