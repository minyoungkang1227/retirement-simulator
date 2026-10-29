# Nohu Compass — Probabilistic Retirement Simulator (Korea)

> ### 👉 [**Try it now — retire-compass.streamlit.app**](https://retire-compass.streamlit.app)
> No input needed: press an **example household** button to see results first. Scan the QR code on your phone to open it. (If nobody has used the app for a while, the first visit may show a "wake up" button — press it and the app opens within ~30 seconds. The app is in Korean.)
>
> <img src="docs/images/qr_retire_compass.png" width="140" alt="QR code">


[한국어](README.md) | **English**

**An actuarial, probability-based retirement simulator built around Korea's pension, tax and health-insurance rules — asking not "how much have I saved?" but "can I sustain the retirement I want?"**

It evaluates three goals — essential living, lifestyle (travel, hobbies) and legacy — across 10,000 economic and longevity scenarios, and compares **what changes bring you closer to your goals**. Results are comparisons on the same scenarios, not recommendations.

- **Web app (no install, Korean UI):** https://retire-compass.streamlit.app
- **Tablet app (light mode for advisor meetings):** https://minyoungkang1227.github.io/retirement-simulator/app/ — add to home screen; the same `retire_sim` engine runs inside the browser (Pyodide), so client inputs never leave the device (`docs/app/`, v50.2)
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

## Example (model v51 · 10,000 paths · reproduce with `scripts/readme_numbers.py`)

**Couple aged 52 and 50, retiring at 60, KRW 500M financial assets, saving KRW 1.5M/month** (app defaults as shipped: essential KRW 2.17M/month + lifestyle KRW 10M/year, pensions KRW 1.6M/month, KRW 100M legacy — KRW 2.17M is the 'minimum' couple living cost from the National Pension Research Institute panel; raising it to KRW 3M, close to their 'adequate' level, drops essential success to 50%)

- Goal success: essential **76%**, lifestyle funding **89%**, KRW 100M legacy **64%** (from v51 the default assumes a 30% chance of leaving the main job 1–7 years early; with retirement exactly at 60 it is 82%)

| What if | Change in essential success (95% CI) |
|---|---|
| Retire 2 years later | **+7.4pp** (±0.5) |
| Spend 10% less | +6.1pp (±0.5) |
| Retire exactly at 60 (no timing risk) | +6.2pp (±0.5) |
| Growth profile (70% equity) | +0.2pp (±0.4) |
| Save KRW 0.5M more per month | +2.6pp (±0.3) |
| Conservative profile (30% equity) | −4.0pp (±0.5) |

| Crisis | Change in essential success |
|---|---|
| Equities −40% in the first year of retirement | −12.0pp |
| Equities −40% ten years into retirement | −6.9pp |
| Three years of 6% inflation | −4.2pp |
| Longer life (mortality −20%) | −3.7pp |
| Long-term care incidence doubled | −7.1pp |

- For this household **retirement timing is the strongest lever**, and the same crash hurts most right after retirement (sequence-of-returns risk).
- **The assumption band is 69%–83%**: the minimum and maximum of moving the equity risk premium, longevity and long-term care incidence one at a time by their estimated ranges (v47). The width is effectively set by the **equity risk premium** — on its own (5.2% ± 1.4pp) it spans 69%–83%, while longevity spans 76%–78% and care incidence (±20%) 75%–79%. Monte Carlo noise of ±0.8pp is about one seventeenth of that, and unlike it this band does not shrink with more paths. That is why differences between choices matter more than levels.

**Young savers (dual-income couple aged 30 and 29, KRW 90M gross income, 15% saving rate, workplace pension, National Pension estimated automatically)**

- Essential success **74%**, lifestyle funding **89%**, KRW 100M legacy **68%**
- What if: save KRW 0.5M more per month +9.2pp · retire 2 years later +7.5pp · spend 10% less +6.0pp
- **Without the workplace pension −25.9pp** — at this age nothing else comes close.

**House-rich, cash-poor household (couple aged 65 and 63, KRW 300M financial assets, home worth KRW 700M, essential KRW 2.5M/month + lifestyle KRW 12M/year, KRW 300M legacy)**

| Housing strategy | Essential success | Lifestyle funding | KRW 300M legacy | Note |
|---|---|---|---|---|
| Keep as is | 14% | 35% | 94% | — |
| Reverse mortgage now (63) | 81% (+67.6pp) | 98% | 70% | KRW 1.65M/month |
| Reverse mortgage at 70 | 86% (+72.6pp) | 92% | 74% | KRW 2.25M/month (today's value) |
| Downsize to half price at 70 | 78% (+63.9pp) | 92% | 87% | KRW 350M released |

- When the home is most of the balance sheet, **whether to use it is the biggest single choice**. A reverse mortgage buys retirement security; downsizing protects the bequest as well.
- Since v45 counts the property component of health-insurance premiums, this household's figures are 2–5pp lower than before, and v47's spending rigidity takes another 5pp off (a household that owns a home carries higher fixed costs in retirement, so spending it cannot cut hurts it more).

![goals](docs/images/goals_whatif.png)
![tax strategies](docs/images/tax_strategies.png)
![sensitivity](docs/images/sensitivity.png)

## Recent changes

> **v50.1 (2026-10-26):** **Verification moved onto the screen.** A **verification ledger** (`retire_sim/verify.py`, 30 items across 11 areas) sets each tax, health-premium, retirement-income-tax and pension calculation beside a value worked out by hand from the statutory rates, deductions and published figures. The app's "About this calculation" tab and the test suite **read the same ledger**, so the on-screen "30 of 30 match" cannot be false while the tests pass. Four planted faults are each caught by exactly the item they should be. Comparison against the agencies' own online calculators is not done yet, and the screen says so. Tests 129 -> 163.

> **v50 (2026-10-26):** **Three gaps filled.** ① **Spending that changes with age** — essential and lifestyle spending used to be constant for life; in reality travel and leisure fall with age while health costs rise. Korean panel data (KCMI 2025, NaSTaB 2008–2021) put equivalised real consumption at 75+ at 43% of age 60–64, and Blanchett (2014) found −1%/year from 60 to 90, but cross-sections mix in households that cannot afford to spend, so the default is conservative: **only lifestyle spending falls, from age 75 to half by 85** (options: also trim essentials, or keep everything flat). The example couple goes 49% → **53%**. ② **Work after retirement** — earned-income deduction and credit, employee health premium, the Basic Pension earnings disregard. KRW 2M a month until 65 adds 6.6pp, until 70 adds 11.3pp; under the June 2026 reform the National Pension is not reduced below about KRW 5.19M a month, so no reduction is applied. ③ **Rental deposits** — counted in the estate, Basic Pension property and 30% in the health-premium property base. A KRW 500M deposit costs 10pp, which corrects households that were wrongly treated as Basic Pension recipients.

> **v49 (2026-10-26):** **Severance pay, workplace pensions and retirement income tax.** Previously only the DC type existed (1/12 of salary accrues each year). Now the user picks **DC or DB / statutory severance** (final monthly wage × years of service) and enters years of service (tier 1 gets one checkbox and the service years). The benefit is assumed to move into an IRP at retirement and be drawn as a pension; **retirement income tax** follows the Income Tax Act (service-years deduction → annualised pay → its deduction → basic rates × years/12, plus 10% local tax), and pension receipt pays 70% of it in years 1–10 and 60% from year 11. A hand calculation (KRW 100M, 10 years = KRW 4.2625M) is pinned by a test. A 52-year-old earning KRW 60M with 20 years of service gets KRW 150M, lifting essential success from 79.0% to **87.7%**. Drawing the IRP first initially *lowered* results by 2.4pp — the taxable account grew and raised financial-income tax and health premiums — so the withdrawal order was kept as in v48 (pension accounts together capped at KRW 15M per person per year); the only difference from v48 is then the tax rate (+0.4pp).

> **v48 (2026-10-26):** **Input now comes in three tiers**, chosen on the first screen — **1 Quick check** (on your own, 3 minutes), **2 My finances** and **3 With an adviser**. The order of questions was measured: moving each input by the error a default would typically make, retirement age (33.5pp), essential spending (29.7pp), lifestyle budget (24.0pp), financial assets (18.7pp), risk profile (18.3pp), the home and debt dominate, while spending after widowhood (6.2pp), parental care (2.9pp) and overseas equity share (1.2pp) are small. High-impact questions anyone can answer go in tier 1, details needed for tax go in tier 2, judgement calls go in tier 3. Unasked inputs take defaults and the results screen lists exactly what was assumed. Alongside: **existing debt** can now be entered (previously only future loans). Only repayments still due after retirement come out of retirement savings; KRW 200M with 20 years left costs −13.0pp. Without debt results equal v47. The full measurement table is in the CHANGELOG.

> **v47 (2026-10-26):** **Spending rigidity is now on by default.** That a household cannot cut its standard of living by more than 30% in a single year is a structural property of household spending, not an investment behaviour, so what v36 had built and left switched off inside the "behavioural assumptions" toggle is now the default (`lifestyle_floor=0.30`). Measured over 80,000 paths: **−1.3pp** for the reference household and **−3.0pp** for a couple just retired at 60. The example households above were re-run — the 52/50 couple 52% → **49%**, the house-rich cash-poor household 17% → **12%** (the larger the fixed costs already are, the more this bites). Alongside: the **assumption band now moves three factors instead of one** (equity risk premium ±1 standard error, mortality level ±1σ, care incidence ±20%, one at a time; the band is the min and max of the six results). Measured, the width is **dominated by the risk premium** — v46's band was not too narrow, but we can now say which assumption creates it and why. `tax.progressive` and `tax.hi_property_points` were rewritten around `searchsorted`, taking one 10,000-path run from **0.94s to 0.78s** with bit-identical results. The economy and mortality chapters of `docs/MODEL.md` were consolidated to the current specification with 20 references, and this English edition was brought in line. Tests 121 → 122.

> **v45 (2026-10-20):** Filled the model's **largest remaining gap** — the **property component of regional health-insurance premiums**. It is worth about -7pp and the default was zero, so unless the user typed in a figure from the insurer's own calculator it silently passed as nothing. The statutory **60-grade score table** from Annex 4 of the National Health Insurance Act Enforcement Decree (grade 1 = 22 points, grade 60 = 2,341 points), the **KRW 211.5 per point** and the **KRW 100M basic deduction** now come straight from the legal text. Measured as a paired comparison: a single-home household with a KRW 500M assessed value goes **45.3% -> 38.5% (-6.78pp, 95% CI [-7.27, -6.29])**, and at KRW 900M **31.4% -> 24.0% (-7.44pp)** - because **twenty years of premiums rise from KRW 15M to KRW 41-56M**, nearly tripling. None of the thresholds are inflation-indexed (as in tax law), so nominal house-price growth pushes a household up the grades by itself: the same quiet tax drift that property taxes have. Seven hand-calculated point tests plus table-shape, non-indexation and engine-integration tests lock it in. The report's **money-flow waterfall also moved into the results tab**, with a test that the app and the report both read from one `reporting.money_flow()`. The reference household owns no home, so its 69.8% is unchanged. Tests 108 -> 116.

Earlier versions (v1–v49) and their measurements are in [CHANGELOG.md](CHANGELOG.md) (Korean).

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
- **Verification ledger**: 26 tax, health-premium, retirement-income-tax and pension items checked against hand calculations from the statutes and published figures; the app and the tests read the same list (`retire_sim/verify.py`)
- 163 tests run on every commit (headline ranges, directional checks, theoretical values, life table, care model,
  reverse-mortgage table, pure helper functions, brand fields reaching the report, notebook/module sync, app smoke tests)
- Data sources and their as-of dates live in one place: **[docs/SOURCES.md](docs/SOURCES.md)** (generated from `retire_sim/sources.py`)

## Structure

```
streamlit_app.py     web app (step-by-step input → goal results → what-if)
retire_sim/          model package
  config.py          input schema (three goals, accumulation phase, care model)
  economy_v2.py      rate, inflation, equity and bond scenarios (Fisher link, integrated rates, jumps, shocks)
  mortality.py       mortality and multi-state care Markov model
  engine_tax.py      tax, accounts, housing, business income and goal guardrail engine
  tax.py             Korean tax and health-insurance rules (as of 2026-09, incl. the 60-grade property score table)
  pension.py         National Pension, Basic Pension and survivor pension formulas
  product.py         annuity products (crediting rate, loadings, surrender charge, annuitization)
  protection.py      protection planning (needs analysis vs human life value)
  optimize.py        option search (equity glide path, National Pension start age, withdrawal order)
  calibrate.py       OU maximum likelihood and ECOS data helpers
  market_data.py     ticker → beta, correlation, rate sensitivity
  metrics.py         depletion probability, confidence intervals, paired comparisons
  sources.py         the single list of external data sources (generates docs/SOURCES.md)
  inputs.py          input → engine config conversion and validation (shared by app and API)
  advice.py          option ranking, assumption ranges
  reporting.py       consultation report assembly, money flow
  report.py          report HTML and SVG charts
  session.py         saving/restoring client inputs, reproducible calculation ID
  branding.py        white-label (institution name, colors, logo)
  api.py             HTTP API (FastAPI) — /simulate, /compare, /report, /product/compare, /usage
  usage.py           API call log (metering and consultation log; client inputs are never stored)
notebooks/           Colab analysis notebook
scripts/             validation and generation scripts (check.py, benchmark_analytic.py, ablate.py, error_budget.py,
                     readme_numbers.py, make_readme_images.py, make_sources_doc.py, sync_notebook.py, product_*.py,
                     measure_hi_property.py, calibrate_*.py, run_example.py)
tests/               regression tests (pytest, run automatically by GitHub Actions)
docs/                equations and design notes (MODEL.md / MODEL.en.md), data sources, figures
CHANGELOG.md         version history (what / how / what changed) — Korean
```

## Usage

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py      # web app
python -m pytest -q                 # full suite, incl. theory checks
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
| Add-on products v15: deposits, life annuities, long-term-care insurance, term life, pension contributions (single-stock-by-ticker removed from the screen in v50.6) | Done |
| ECOS rate/inflation estimation (v14) | Done |
| Institutional precision v28·v32·v45: survivor pension, statutory Basic Pension offset, 60-grade property score for health premiums | Done |
| Longevity risk and option search v29·v30: stochastic mortality, widowhood effect, equity glide path, pension start age, withdrawal order | Done |
| Verification and simplification v31·v37·v44: closed-form benchmarks, component ablation, hand-computed tax tests | Done |
| Consultation tooling v33–v35: sessions, reproducible ID, two-page report, HTTP API, white-label | Done |
| Annuity product comparison v34·v39: numbers straight from product disclosures, product spread vs start age | Done |
| Care-cost wage indexation, double care, behavioral assumptions v36 | Done |
| Single source list v40: refresh order generated automatically | Done |
| Protection planning v42·v43: needs analysis vs human life value, conditional (death/survival) comparison | Done |
| Account/budget-app import, periodic re-measurement | Planned |

See [CHANGELOG.md](CHANGELOG.md) and [docs/MODEL.en.md](docs/MODEL.en.md) for details.

## Limitations

- Data sources and their as-of dates live in one place: **[docs/SOURCES.md](docs/SOURCES.md)**. No placeholder values remain. Mortality uses the Statistics Korea 2024 complete life table **with cohort improvement applied by default** (turning it off raises the headline probability by 7–9pp). The largest uncertainty is the equity risk premium (1σ span 13.7pp) — read ranges, not point estimates, and compare options rather than absolute levels.
- **Institutional values are fixed to current law.** The risk of legal change (amendments, enforcement decrees) is not inside the probability model. Updates are managed semi-manually through the source list (`retire_sim/sources.py`); automatic amendment detection does not exist yet.
- The property component of health premiums does not yet include vehicles or the monthly-rent conversion (lease deposits are included from v50).
- Tax rules are simplified (Korea, as of 2026-09). **Not investment or tax advice.**
- Web-app inputs are used only for calculation and are not stored.

## Contact

- Email: doongss1@naver.com

## Copyright

© 2026 Minyoung Kang. All rights reserved. For usage inquiries, please contact me by email. See [COPYRIGHT](COPYRIGHT).
