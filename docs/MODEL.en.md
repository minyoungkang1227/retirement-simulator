# Model Documentation — Probabilistic Retirement Simulator

[한국어](MODEL.md) | **English**

> As of 2026-10 (model v47) · Currency unit: KRW 10,000 (만원) · Time step: 1 year

## 1. Purpose and scope

The model computes the probability that a retired household's assets last for as long as its members live, and answers two questions:

- **① Choice comparison**: how does the depletion probability change with the National Pension claiming age, withdrawal order, asset allocation, and similar choices?
- **③ Insurance value**: how much do existing or new insurance products reduce depletion probability and shortfall, and at what cost?

Results are presented as **comparisons under identical scenarios**, not as recommendations.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $W_t$ | Liquid assets at the start of year t (nominal) |
| $r_t$ | Short rate |
| $\pi_t$ | Inflation rate, $CPI_t=\prod_{s<t}(1+\pi_s)$ |
| $q_x$ | One-year death probability at age x |
| ${}_tp_x$ | Probability that a person aged x survives t more years $=\prod_{s=0}^{t-1}(1-q_{x+s})$ |
| $q_t$ | Real short rate, $r_t=q_t+\pi_t$ (Fisher) |
| $H_t$ | Real house price index (nominal price $=H_t\times CPI_t$) |
| $\text{ERP}_p$ | Equity risk premium of path $p$ (drawn per path) |
| $N$ | Number of simulated paths (default 10,000) |

## 3. Economic scenarios (current spec: `EconomyV2.enhanced()`, calibrated 2026-08)

> Consolidates what used to be spread over the v14 (rates and inflation), v18 (equities), v26 (house prices) and v11 (term premium) entries, which are kept in Appendix A as history. `retire_sim/sources.py` is the single registry of every calibrated value, its source, as-of date and update procedure.

### 3.0 Structure

Four state variables: inflation $\pi_t$, the real short rate $q_t$, the equity log return, and the real house price index $H_t$. The nominal rate is the Fisher sum $r_t=q_t+\pi_t$, so nominal rates follow inflation. Four parameters are drawn **once per path** and held fixed along it: the equity risk premium $\text{ERP}_p$, long-run inflation $\theta_{\pi,p}$, the long-run real rate $\theta_{q,p}$ and real house price growth $g_p$. This puts **estimation uncertainty** inside the 10,000 paths and creates a variance component that does not shrink with more paths. Paths use antithetic draws ($+Z$, $-Z$ pairs) to reduce sampling error, and choice comparisons use common random numbers.

### 3.1 Inflation and the real rate: Ornstein–Uhlenbeck

$$dX_t=\kappa(\theta-X_t)\,dt+\sigma\,dW_t,\qquad X\in\{\pi,q\}$$

Exact annual discretisation:

$$X_{t+1}=\theta+(X_t-\theta)e^{-\kappa}+\varepsilon_{t+1},\qquad \varepsilon\sim N\!\Big(0,\ \sigma^2\tfrac{1-e^{-2\kappa}}{2\kappa}\Big)$$

**Estimation.** The exact discretisation is an AR(1) $X_{t+1}=a+bX_t+\varepsilon$, so the MLE has a closed form in the regression coefficients: $b=e^{-\kappa\Delta}$, $\theta=a/(1-b)$, $\sigma^2=s^2\cdot 2\kappa/(1-b^2)$. The speed of mean reversion is corrected for AR(1) small-sample bias, $b+(1+3b)/n$ (Kendall 1954; Marriott & Pope 1954). The standard error of the long-run mean comes from the delta method on the asymptotic covariance of $(a,b,s^2)$ and is used as per-path parameter uncertainty $\theta_p\sim N(\theta,\text{se}(\theta)^2)$. Code: `retire_sim/calibrate.py::ou_mle`.

| Variable | $\theta$ | $\kappa$ | $\sigma$ | $\text{se}(\theta)$ | Initial |
|---|---|---|---|---|---|
| Inflation $\pi$ | 2.47% | 0.35 | 1.33% | 0.52pp | 3.09% |
| Real rate $q$ | 0.73% | 0.56 | 1.37% | 0.38pp | 0.70% |

Data: Bank of Korea ECOS, monthly 2000-01 to 2026-08. Inflation = CPI (901Y009) year-on-year; real rate = 3-year KTB yield (721Y001) minus realised inflation.

### 3.2 Nominal rate and bonds

$r_t=q_t+\pi_t$. Bonds are a fund that rolls $D=5$-year zero-coupon bonds annually, priced with the Gaussian closed form for the sum of two correlated OU factors:

$$P(\tau)=\exp\!\Big(-\mathbb{E}\!\Big[\int_0^\tau r_s\,ds\Big]+\tfrac12\,\mathrm{Var}\!\Big[\int_0^\tau r_s\,ds\Big]\Big)$$

The pricing-measure long-run mean is $\theta+\lambda_{\text{term}}$ with $\lambda_{\text{term}}=0.5\%$, giving a positive term premium (v11). The one-year return is $R^B_t=P(D-1;\,r_{t+1})/P(D;\,r_t)-1$. A 0.5% annual fee is deducted from both equity and bond returns. Code: `economy_v2.py::gaussian2_price`.

### 3.3 Equities: diffusion plus Merton jumps

$$\ln(1+R^S_t)=r_t+\text{ERP}_p-\tfrac12\sigma_S^2+\sigma_S Z_t+\sum_{i=1}^{N_t}J_i-\lambda\big(e^{\mu_J+\sigma_J^2/2}-1\big),\qquad N_t\sim\text{Poisson}(\lambda),\ J_i\sim N(\mu_J,\sigma_J^2)$$

The last term is the compensator that keeps the expected return unchanged when jumps are added (Merton 1976).

**Calibration.** KOSPI, 319 monthly observations 2000-02 to 2026-08 (FinanceDataReader `KS11`). Targets: annualised monthly volatility 23.9%, excess kurtosis 2.43, and the frequency of 12-month returns below −20% (9.4%) and below −30% (2.6%). These are matched by diffusion $\sigma_S=21\%$ plus jumps $(\lambda,\mu_J,\sigma_J)=(0.05,\,-20\%,\,10\%)$, which reproduce an annual log-return standard deviation of 21.6% and tail frequencies of 9.6% and 2.9%.

**Bayesian ERP.** The sample estimate alone is too wide to use, so it is combined with a prior.

| | Mean | S.D. |
|---|---|---|
| Data (price excess return 3.92% ± 9.09pp + dividends 1.8% + $\tfrac12\sigma^2$) | 8.6% | 4.6pp |
| Prior (Damodaran 2026 mature-market 4.2% + Korea country risk ≈ 0.7pp) | 4.9% | 1.5pp |
| **Posterior (precision-weighted)** | **5.2%** | **1.4pp** |

Each path draws $\text{ERP}_p\sim N(5.2\%,1.4\%^2)$. The "assumption band" in reports (e.g. 50–65%) is the min and max over six runs that move **one assumption at a time**: ERP $\pm1$ s.e. (1.4pp), the mortality level $\pm1\sigma$ ($\sigma_L=0.05$, §4.3) and the long-term-care incidence $\pm20\%$ (a judgement width) — `advice.py::assumption_band_detail` (v47). In practice the ERP dominates the width (default household: ERP 70–86%, longevity 78–80%, care 77–81%). This band does not shrink with more paths and is a different kind of uncertainty from the sampling error of 10,000 paths (about ±1pp around 57%, which does shrink).

### 3.4 Correlation

The shocks $(\varepsilon^q,\varepsilon^\pi,Z^S,Z^H)$ are generated from a correlation matrix via Cholesky. In Fisher mode $q$ and $\pi$ are linked structurally, so their shock correlation $\rho_{q\pi}$ is set to 0. Monthly data are smoothed and bias correlations toward zero, so correlations were re-estimated on annual data.

| $\rho_{rS}$ | $\rho_{\pi S}$ | $\rho_{HS}$ |
|---|---|---|
| +0.09 | 0 | +0.10 |

### 3.5 House prices

The real index follows geometric Brownian motion; the nominal price is the real index times CPI:

$$\ln H_{t+1}=\ln H_t+g_p-\tfrac12\sigma_H^2+\sigma_H\big(\rho_{HS}Z^S_t+\sqrt{1-\rho_{HS}^2}\,Z^H_t\big),\qquad g_p\sim N(1.25\%,\ 1.18\%^2)$$

**Calibration.** Korea Real Estate Board national house price survey, monthly from 2013-01 (the series provider changed in 2013). The sample real growth 3.25% ± 1.89pp is tilted by the 2013–2026 upswing, so it is combined with the prior $N(0\%,1.5\%^2)$ — real house prices cannot outgrow income indefinitely — giving 1.25% ± 1.18pp. Monthly index volatility is smoothed (lag-1 autocorrelation 0.82) and understates at 2.5%; the annual re-estimate 6.6% plus 7% idiosyncratic single-home risk gives $\sigma_H=10\%$.

The nominal price feeds the property component of health-insurance premiums (Enforcement Decree Table 4 grade schedule, not indexed to inflation), reverse-mortgage payments and downsizing cash flows. Because the grade schedule is nominal, rising house prices raise premiums automatically.

### 3.6 Limiting-case check

With volatility, jumps, taxes, mortality and care all switched off, the wealth path must follow the closed form $W_{t+1}=(W_t-\text{spending})e^{r}$ (real = nominal ÷ CPI when inflation is on). The model matches to a relative error of $10^{-7}$ (`tests/test_engine.py::test_analytic_benchmark_*`).

## 4. People: mortality, longevity risk, widowhood and long-term care

> Consolidates the v13 (life table), v27 (cohort improvement), v30 (longevity risk), v32 (widowhood) and v17 (two-level care) entries.

### 4.1 Base life table

Statistics Korea 2024 complete life table (KOSIS DT_1B42), one-year death probabilities $q_x$ by sex, ages 0–99. Ages 100–109 are extrapolated by fitting a straight line to $\ln\mu_x$ over 90–99 (Gompertz 1825), and $q_{110}=1$. File `retire_sim/data/life_table_2024.csv`, loader `mortality.py::default_qx`.

### 4.2 Cohort mortality improvement

A period table assumes this year's death rates for a whole lifetime and understates longevity. Age-specific annual improvement rates $k_x$ project it forward:

$$q(x,\ y)=q_{2024}(x)\,(1-k_x)^{\,y-2024}$$

| Age | <65 | 65–79 | 80–89 | 90–99 | 100+ |
|---|---|---|---|---|---|
| $k_x$ | 2.5% | 2.0% | 1.3% | 0.7% | 0.3% |

These approximate the 2013→2024 trend in the Statistics Korea tables (life expectancy +2.3 years for men, +1.5 for women; +0.21/+0.14 per year) and play the role of the SOA Mortality Improvement Scale MP. Switching improvement off raises the representative household's probability by 7–9pp (an optimistic bias). Re-estimation from multi-year tables: `scripts/calibrate_mortality.py`.

### 4.3 Systematic longevity risk (v30)

The improvement rate itself is uncertain, so each path carries a mortality multiplier, shared by both spouses so that it is **undiversifiable, systematic** risk:

$$m_{p,t}=\exp\!\big(\eta_p-\varepsilon_p t-\tfrac12(\sigma_L^2+\sigma_T^2t^2)\big),\qquad \eta_p\sim N(0,\sigma_L^2),\ \varepsilon_p\sim N(0,\sigma_T^2)$$

$\sigma_L=0.05$ (level), $\sigma_T=0.005$ (improvement speed, per year); the last term keeps $\mathbb{E}[m]=1$. This compresses the stochastic period effect of Lee–Carter (1992) and Cairns–Blake–Dowd (2006) into two parameters. It uses its own random stream, so turning it on or off leaves the economic and care draws unchanged (common random numbers preserved). Code: `mortality.py::longevity_multiplier`.

### 4.4 Widowhood effect (v32)

Mortality multiplier for a survivor $s$ years after the spouse's death:

$$1+e_{\text{sex}}\big(1+\phi\,e^{-s/\tau}\big),\qquad e=(0.22_{\text{men}},\ 0.03_{\text{women}}),\ \phi=2.0,\ \tau=0.5$$

From the meta-analysis of Moon, Kondo, Glymour & Subramanian (2011) (2.26 million people, 15 prospective cohorts, age- and sex-adjusted): relative risk 1.22 for men and 1.03 for women (the latter not significant), 1.41 within six months of bereavement and 1.14 afterwards — matched here by an initial spike factor of 2.0 decaying over 0.5 years.

### 4.5 Long-term care: four-state Markov model (v17)

States: healthy (H) → mild care (M: LTC grades 3–5 and cognitive support) → severe care (S: grades 1–2) → dead (D). Annual transitions:

- $H\to M$: $(1-q_H)\,i(x)\,(1-e)$, $\quad H\to S$: $(1-q_H)\,i(x)\,e$
- $M\to S$: progression $0.0209$, $\quad M,S\to H$: recovery $0.02$
- Death: $q_H$ (healthy), $m_M q_H$ (mild, $m_M=1.8$), $m_S q_H$ (severe, $m_S=4.0$)

The healthy-state death rate is adjusted so that the life-table $q_x$ is preserved given the target prevalence $p(x)$ and severe share $s$:

$$q_H=\frac{q_x}{1+(m_M-1)\,p(x)(1-s)+(m_S-1)\,p(x)\,s}$$

| Age | 65 | 70 | 75 | 80 | 85 | 90 | 90+ |
|---|---|---|---|---|---|---|---|
| Incidence $i(x)$ | 0.15% | 0.43% | 0.75% | 2.68% | 6.93% | 9.98% | 12.60% |
| Prevalence $p(x)$ | 0.3% | 1.5% | 3.9% | 9.9% | 24.0% | 42.3% | 55.4% |

Severe share $s=12.7\%$ (grades 1–2 among recipients, 2025: 4.5% + 8.2%); share entering directly as severe $e=10\%$. Costs are lognormal: mild KRW 7.2 million a year at the median (15% co-payment for home care plus non-covered and domestic help, $\sigma=0.6$), severe KRW 18 million (a mix of nursing homes at KRW 0.6–0.75 million a month and hospital care with a private carer at KRW 3 million+). Care costs are assumed to outpace general inflation because they are labour costs; the gap between LTC fee increases (2022–2026 average 4.05%) and CPI (3.02%) is applied (v36).

Data: NHIS long-term care statistics (1.235 million recipients in 2025, grade mix), the 2022 LTC survey age distribution, and 2025 statistics on the elderly. Calibration script `scripts/calibrate_care.py`. The framework is the multi-state actuarial model of Fong, Shao & Sherris (2015) with transition rates fitted to Korean NHIS statistics.

### 4.6 Couples

The two lives are simulated as conditionally independent, linked through the shared longevity multiplier (§4.3) and the widowhood effect (§4.4). Last-survivor probability ${}_tp_{\overline{xy}}={}_tp_x+{}_tp_y-{}_tp_x\,{}_tp_y$. Spending after the first death is scaled by `survivor_spending_ratio`.

## 5. Institutional rules (as of 2026-09, kept in a separate configuration)

- National Pension: 6% reduction per year of early claiming (up to 5 years), 7.2% increase per year of deferral (up to 5 years), CPI-indexed
- Private pensions: 3.3–5.5% separate taxation up to KRW 15M per year; above that, full comprehensive taxation or 16.5% separate taxation (Section 12)
- Health-insurance dependent status: annual income including public pensions up to KRW 20M; private pensions excluded (Section 12)

## 6. Cash flow: the annual sequence and the depletion test (current spec, `engine_tax.run`)

> The one-line v1 recursion $W_{t+1}=\max(W_t+I_t-C_t-K_t,0)(1+R_t)$ is kept only as an illustration. The engine performs the following steps for every path, every year.

1. **Survival and care state**: update both members with the life table, longevity multiplier, widowhood effect and care Markov chain of §4; reassign a deceased member's ownership shares to the survivor.
2. **Income** $I_t$: National Pension (basic, survivor, early/deferred adjustments, ×CPI), private pensions, Basic Pension (income-recognition formula), earned and business income, rent, reverse-mortgage payments.
3. **Taxes and health insurance**: comprehensive income tax (pension income deduction, comparative taxation), capital gains tax, property holding taxes, health-insurance income and property components (Table 4 grade schedule, nominal), dependant eligibility. All computed in nominal terms and recorded in real terms.
4. **Spending**: essential spending $E_t$ (×CPI) and discretionary spending $L_t$ (travel, hobbies). Discretionary spending is allowed only to the extent assets exceed the **floor** $F_t$ (Appendix A, v16: survival-weighted present value of essential spending), and **spending rigidity** (`lifestyle_floor`=0.30, default from v47) prevents cutting it by more than 30% from the previous year — what cannot be cut is still spent, so the essential cushion erodes faster. Care cost $K_t$ is the state-specific lognormal cost times care inflation.
5. **Withdrawal order**: shortfalls are met from the taxable account → ISA → pension accounts (taxed on withdrawal) → safety buffer. Pre-retirement shortfalls are treated as borrowing, not depletion.
6. **Depletion test**: after retirement, the **first year** in which an essential-spending shortfall remains after all accounts are exhausted is recorded in `depleted_at`. The probability of sustaining essential living = the share of the 10,000 paths with no such year. Real shortfalls accumulate for the CTE measures.
7. **Asset update**: apply equity and bond returns (§3) to remaining assets; adjust the equity weight via the glide path and, optionally, panic selling.

## 7. Metrics

| Metric | Definition |
|---|---|
| Depletion probability | $P(\text{depletion time}<\text{last death})$ |
| Depletion-age distribution | Quantiles of depletion time among depleted paths |
| Expected shortfall | Mean present value of the shortfall on depleted paths *(planned)* |
| Fan chart | 5/25/50/75/95th percentiles of real wealth $W_t/CPI_t$ among surviving households |

Choice comparisons use **common random numbers**, so differences reflect the choice rather than sampling noise.

## 8. Insurance module (theory fixed, implementation planned)

- Life-annuity present value: $\ddot a_x=\sum_t v^t\,{}_tp_x$; annuity payment = single premium / ($\ddot a_x\times$(1 + loading))
- Mortality credit: effective return to survivors $(1+i)/(1-q_x)-1$
- Long-term-care net premium (equivalence principle): $P\,\ddot a_{x:\overline{n}|}=B\sum_t v^{t+1}{}_tp_x\,i_{x+t}$
- Endowment: $A_{x:\overline{n}|}=A^1_{x:\overline{n}|}+{}_nE_x$
- Pricing uses insurer experience tables while the simulation uses population tables, which exposes the insurer's margin
- Benefits are tied to the same death and care events generated by the engine
- Value metrics: Δ depletion probability, Δ expected shortfall, expected present value of loadings, reduction per KRW 1M of cost

## 9. Validation

**Current (automated tests, `tests/test_engine.py`).**

| Item | Benchmark | Result |
|---|---|---|
| Wealth path in the deterministic limit (volatility, taxes, mortality, care all off) | Closed form $W_{t+1}=(W_t-\text{spending})e^{r}$; real = nominal ÷ CPI with inflation | Matches to relative error $10^{-7}$ over 15 years |
| Pension inflow | An inflow of KRW 12 million a year reduces withdrawals by exactly that amount | Matches |
| Mortality preservation in the care chain | The state-mixture death rate must equal the life-table $q_x$ | Guaranteed by the §4.5 adjustment, pinned by a test |
| Choice comparison | Paired differences and 95% CI on common random numbers | `metrics.compare` |
| Out-of-sample check | The best candidate is re-run on fresh draws not used for selection (avoids the winner's curse) | `optimize.py` |

**At v2 (for reference).** Long-run rate mean/s.d. theory 3.00%/1.83% vs simulation 2.99%/1.84%; inflation 2.00%/1.12% vs 1.98%/1.13%; 5-year zero price 0.87706 closed form vs 0.87723 MC. Parameters have since changed, but the exact discretisation of the generator is unchanged.

**Not yet done.** Reconciliation against official calculators (NPS expected-pension lookup, the integrated pension portal), review by actuarial and tax professionals, and a KOSPI historical backtest. Verifying that the code computes what was intended and verifying that the assumptions match reality are different tasks; the second is still open.

## 10. Sensitivity analysis (example household, baseline depletion probability 39.9%)

![sensitivity](images/sensitivity.png)

| Assumption (low / high) | Depletion probability | Range |
|---|---|---|
| Spending −10% / +10% | 19.7% / 62.4% | 42.7pp |
| Long-run rate 2% / 4% | 52.9% / 27.7% | 25.2pp |
| Equity weight 20% / 60% | 53.3% / 35.3% | 18.0pp |
| Long-run inflation 1.5% / 2.5% | 31.4% / 49.2% | 17.8pp |
| Equity risk premium 3% / 5% | 48.0% / 32.6% | 15.4pp |
| Care incidence 1.5% / 6% | 36.3% / 46.4% | 10.1pp |
| Mortality ×1.2 / ×0.8 | 35.5% / 45.2% | 9.7pp |
| Equity volatility 15% / 21% | 37.2% / 42.6% | 5.4pp |

Interpretation: the result is driven most by **spending, which the household controls**, rather than by market assumptions. The long-run rate matters a lot because expected equity returns are defined as rate plus risk premium, so the rate feeds into both equities and bonds.

## 11. Limitations

- Parameter uncertainty: expected future returns carry large estimation error → shown as ranges and scenarios
- Interpreting probabilities: depletion probability is a statistic over 10,000 scenarios and does not predict an individual outcome
- Vasicek allows negative rates (about 4.7% of path-years) → comparison with CIR planned
- Zero bond risk premium; independent spousal mortality
- Events outside the model (divorce, supporting children, property crashes) and future rule changes are not captured
- **Differences between choices** are more reliable than absolute levels, because shared assumption errors largely cancel

## 12. Tax, health-insurance and property module (engine_tax, approximate)

The engine tracks three accounts (taxable, ISA, pension) and housing (illiquid) per path. Order within each year:
pension income → gifts → account transfers → spending → pension-account withdrawal → taxes, premiums, holding taxes → taxable/ISA withdrawals → returns → inheritance.

| Item | Treatment (as of 2026-09, simplified) |
|---|---|
| Comprehensive income tax | Public pension (taxable share, pension-income deduction) + business income + rental income, progressive rates, 10% local tax |
| Financial-income taxation | Comparative taxation when interest and dividends exceed KRW 20M per year |
| Private pensions | 3.3–5.5% up to KRW 15M per year, 16.5% above |
| Foreign-equity capital gains | Cost basis tracked, 22% above a KRW 2.5M annual deduction (paid the following year) |
| Health-insurance premiums | Regional: 50% of public pensions + financial income (fully counted above KRW 10M) + rental/business income + **property component** (statutory 60-grade score table in Enforcement Decree Annex 4 x KRW 211.5 per point, KRW 100M basic deduction - computed automatically) / Workplace (business owner): business income + other income above KRW 20M / dependent income and property tests |
| Property and comprehensive real-estate taxes | Fair-market-value ratios, progressive rates, comprehensive tax assessed per person (KRW 900M deduction each), senior and long-holding credits for single-home owners (80% cap) |
| Inheritance and gift taxes | At the second death; lump-sum and financial-asset deductions; gifts within 10 years added back; KRW 50M gift deduction per 10 years |
| Tax strategies | ISA (KRW 20M per year, 100M total), pension accounts (KRW 18M per year transferred, withdrawals up to 15M per year after 5 years), ownership splitting, 10-year gifting cycles |

Not included: dividend gross-up credit, high-dividend separate-taxation special rule, home sale and capital-gains tax, reverse mortgages, pension-account tax credits, first-death inheritance tax, earnings-related National Pension reduction.
All tax figures live in `retire_sim/tax.py` and need annual updates. **This is not tax or investment advice.**

## References

The literature each component rests on, with a note on where it is used. Official statistics and legal sources are listed separately in `docs/SOURCES.md`.

**Rates and inflation (§3.1–3.2)**

1. Vasicek, O. (1977). An equilibrium characterization of the term structure. *Journal of Financial Economics*, 5(2), 177–188. — Mean-reverting short rate and zero-coupon closed form.
2. Kendall, M. G. (1954). Note on bias in the estimation of autocorrelation. *Biometrika*, 41(3/4), 403–404; Marriott, F. H. C., & Pope, J. A. (1954). Bias in the estimation of autocorrelations. *Biometrika*, 41(3/4), 390–402. — Small-sample bias correction $b+(1+3b)/n$ for the AR(1) coefficient.

**Equities (§3.3)**

3. Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144. — Diffusion plus Poisson jumps and the mean compensator.
4. Damodaran, A. (2026). *Equity Risk Premiums (ERP): Determinants, Estimation and Implications*. NYU Stern, annual update. — Prior for the mature-market ERP (4.2%) and the country-risk add-on.

**Mortality and longevity risk (§4.1–4.3)**

5. Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. *Philosophical Transactions of the Royal Society*, 115, 513–583. — Log-linear extrapolation of old-age mortality.
6. Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *Journal of the American Statistical Association*, 87(419), 659–671. — Stochastic period effect in mortality improvement.
7. Cairns, A. J. G., Blake, D., & Dowd, K. (2006). A two-factor model for stochastic mortality with parameter uncertainty. *Journal of Risk and Insurance*, 73(4), 687–718. — Two-factor (level and slope) longevity risk; basis for the $(\eta,\varepsilon)$ compression in §4.3.
8. Society of Actuaries (2021). *Mortality Improvement Scale MP-2021*. — Role model for the age-graded improvement scale; Korean values fitted to the 2013→2024 Statistics Korea trend.

**Widowhood (§4.4)**

9. Moon, J. R., Kondo, N., Glymour, M. M., & Subramanian, S. V. (2011). Widowhood and mortality: A meta-analysis. *PLOS ONE*, 6(8), e23465. — Relative risks by sex (1.22 men, 1.03 women) and time pattern (1.41 within six months, 1.14 after).

**Long-term care (§4.5)**

10. Fong, J. H., Shao, A. W., & Sherris, M. (2015). Multistate actuarial models of functional disability. *North American Actuarial Journal*, 19(1), 41–59. — Healthy → disabled → dead multi-state actuarial framework.
11. "Financial sustainability and intergenerational equity of Korea's long-term care insurance" (Korean). *Health and Social Welfare Review*, 40(4), 2020. — Long-run projection of Korean LTC demand and finances. (Authors to be confirmed.)

**Retirement spending, withdrawals and the floor (§6; Appendix A, v16)**

12. Bengen, W. P. (1994). Determining withdrawal rates using historical data. *Journal of Financial Planning*, 7(4), 171–180. — Origin of the withdrawal-rate literature (the 4% rule).
13. Cooley, P. L., Hubbard, C. M., & Walz, D. T. (1998). Retirement savings: Choosing a withdrawal rate that is sustainable. *AAII Journal*, 20(2), 16–21. — The success-probability concept (Trinity Study).
14. Milevsky, M. A., & Robinson, C. (2005). A sustainable spending rate without simulation. *Financial Analysts Journal*, 61(6), 89–100. — Closed-form ruin probability when both lifetime and returns are random; a benchmark for the §9 limiting-case check and the floor.
15. Waring, M. B., & Siegel, L. B. (2015). The only spending rule article you will ever need. *Financial Analysts Journal*, 71(1), 91–107. — Spending rule based on the annuitised value (ARVA). Partly the same idea as the floor (Appendix A, v16): ARVA annuitises all spending, the floor discounts essential spending only.
16. Ko, H., Lee, S., & Lee, J. (2024). Sequence and longevity risks of South Korean retirees: Insights and potential remedies. *Pacific-Basin Finance Journal*, 83, 102263. — Empirical evidence on sequence and longevity risk for Korean retirees and dynamic withdrawal strategies.
17. Yuh, Y., & Yang, J. (2011). Optimal annuity planning and longevity risk: Evidence from Korea. *Applied Economics*, 43(11), 1423–1433. — Annuitisation timing and longevity risk for Korean households.

**Risk measures and simulation (§7, §3.0)**

18. Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228. — CTE as a coherent risk measure.
19. Hardy, M. R. (2003). *Investment Guarantees: Modeling and Risk Management for Equity-Linked Life Insurance*. Wiley. — CTE in actuarial practice; regime-switching equity models.
20. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. — Antithetic and common random numbers and other variance-reduction techniques.

## Appendix A. Version history (v10 → v27 translated; later entries in the Korean file and `CHANGELOG.md`)

> A record of what changed in each version and why. Sections 3, 4, 6 and 9 above are the current specification; the entries below describe the state at that time.

### v10 model enhancements (economy_v2 options, `EconomyV2.enhanced()`)

| Item | Description | Effect (example household, baseline 39.9%) |
|---|---|---|
| Fisher link | Real rate $q$ and inflation $\pi$ follow separate OU processes; nominal rate $r=q+\pi$. Bonds priced with the two-factor Gaussian closed form $P=\exp(-E[\int r]+\tfrac12 Var[\int r])$ | 39.3%. The high-inflation scenario (long-run 3.5%) moves from 69.2% to 50.4%, removing the inconsistency of rates not following inflation |
| Management fees | 0.5% per year deducted from equity and bond returns | 49.2% (+9.9pp) — the largest change; earlier results were optimistic |
| Parameter uncertainty | Per-path ERP ~ N(4%, 1%), long-run inflation ~ N(2%, 0.5%) | Mean 49.6%; paths in the lowest ERP third 57.4% vs highest third 41.1% |
| Crash jumps (Merton) | 0.1 jumps per year, log jump N(−15%, 10%), compensator $-\lambda k$ keeps the mean return | Mean return preserved (7.17% → 7.16%); worst 1% annual return −30.8% → −33.7% |
| Antithetic variates | Half of the paths use $-Z$; standard errors computed from pair means | 95% CI ±0.98pp → ±0.82pp |

Validation (`tests/test_engine.py`): the two-factor formula reduces to Vasicek in the one-factor case (0.877060); two-factor 5-year bond price closed form 0.87553 vs Monte Carlo 0.87518; long-run nominal rate 3.00% (theory 3.00%).
With the v10 settings the overall depletion probability is 49.1% (74.6% with the tax engine). The default `EconomyV2()` reproduces the original v2 results.

### v11 mathematical refinements

- **① Integrated rate.** Equity returns use the one-year integral $\int r\,ds$ instead of the start-of-year rate, matching how bonds are priced. The end-of-year factor values, their integrals and the equity shock are drawn jointly from a 5×5 Gaussian whose covariance follows from the Itô isometry (formulas in the Korean section 14). Max correlation error vs a fine-step simulation: 0.005.
- **② Paired differences.** Choice comparisons use $d_i=\mathbb 1[A_i]-\mathbb 1[B_i]$; with common random numbers the CI shrinks from ±1.38pp to ±0.44pp.
- **③ Multi-state care model.** Markov chain {Healthy, Care, Dead} with care mortality multiplier $m=2.5$ and recovery 0.1; healthy mortality is scaled so the life-table mortality is preserved (life expectancy at 60: 23.5 → 23.6 years).
- **④ Market price of risk.** Paths under P, bond prices under Q with $\theta^Q=\theta+0.5\%$, producing a term premium (bond mean return 2.46% → 2.77%).
- **⑤ OU maximum likelihood.** Exact MLE via the AR(1) closed form with delta-method standard errors, plus ECOS fetch and Fisher-mode calibration helpers. Synthetic tests recover θ and σ; κ shows the known small-sample upward bias.

Before/after numbers are in [CHANGELOG.md](../CHANGELOG.md).


### v12 goal-based structure

- Spending split into Essential $E_t$ (protected), Lifestyle $L_t$ (flexible) and a Legacy target $G$.
- Guardrail: after retirement, the floor $F_t=\max(E_t-I_t,0)\cdot\ddot a_{\overline{n}|2\%}$ with $n=\max(5,95-x_t)$; lifestyle paid $=\min(L_t,\max(W_t-F_t,0))$.
- Metrics: essential success probability, average lifestyle funding ratio, probability that after-tax estate ≥ $G$.
- Accumulation phase: before retirement no spending, annual savings added.
- Required assets: bisection on initial assets for a 90% essential success probability, using common random numbers.


### v14 rate and inflation estimation

ECOS monthly data 2000-01 to 2026-08: inflation = CPI year-on-year; ex-post real rate = 3-year treasury yield − inflation. Exact OU MLE with AR(1) small-sample bias correction $\hat b\to\hat b+(1+3\hat b)/n$.

| Factor | Long-run mean θ (±95%) | Mean reversion κ (before→after correction) | Volatility σ | Start |
|---|---|---|---|---|
| Inflation | 2.47% (±1.02pp) | 0.51 → 0.35 | 1.33% | 3.09% |
| Real rate | 0.73% (±0.74pp) | 0.72 → 0.56 | 1.37% | 0.70% |

Standard errors of the long-run means are used as per-path parameter uncertainty. Nominal long-run mean 3.20%.


### v15 add-on products

Individual stocks (single-index model with a rate-change term; β, idiosyncratic volatility and rate sensitivity estimated from 10 years of monthly prices), deposits, single-premium life annuities (priced with 0.8× life-table mortality, 3% discount, 5% loading; single or joint-last), long-term-care insurance (premium by the equivalence principle on the engine's own care paths, 30% loading) and pension-account contributions (13.2% tax credit). New metric: average real shortfall when depleted. Add-on noise uses a separate random stream so paired comparisons remain valid.


### v16 actuarial floor

The essential floor becomes the survival-weighted present value of future essential shortfalls, computed separately for three household states (both alive, only you, only spouse), using life-table survival × 0.8 mortality, a 2% real discount rate, and projected guaranteed income (National Pension from its start age, fixed-nominal private pensions deflated by expected inflation, rent). Lifestyle spending is paid only above this floor. Previously (v12–v15) the current-year shortfall was treated as lasting to age 95 with certainty.


### v17 two-level long-term-care model

States healthy / mild (grades 3–5, cognitive support) / severe (grades 1–2) / dead. Incidence by age is calibrated so that simulated prevalence matches national figures (1.235M recognised in 2025 × age mix ÷ population: 1.5% at 65–69 up to 55.4% at 90+), with the severe share matched to 12.7%. Mortality multipliers 1.8 (mild) and 4.0 (severe), healthy mortality rescaled to preserve the life table. Annual costs (today's KRW): mild median 7.2M, severe median 18M (lognormal), based on 2026 out-of-pocket rules and caregiver market prices.


### v18 equity calibration

KOSPI monthly data 2000-02 to 2026-08: diffusion volatility 21% plus Merton jumps (0.05 per year, −20% ± 10%) match the observed tail frequencies of 12-month returns (≤ −20%: 9.4%, ≤ −30%: 2.6%); correlation with 3-year yield changes +0.09. The equity risk premium is a normal-normal Bayesian posterior combining the sample estimate (8.6% ± 4.6pp s.e., including an assumed 1.8% dividend yield) with a prior of 4.9% ± 1.5pp (Damodaran mature-market ERP 4.2% plus a Korea country premium), giving 5.2% ± 1.4pp, used as per-path parameter uncertainty.


### v19 housing strategies

Reverse mortgage: fixed nominal monthly payment from the Korea Housing Finance Corporation 2026 table (interpolated by the younger spouse's age and home value, capped at KRW 1.2B), paid while either spouse lives; the loan balance accrues at the model short rate + 1.1pp plus a 0.75% annual guarantee fee (1% upfront) and is repaid from the house at death without recourse; 25% property-tax relief. Downsizing: sale at a chosen age with 0.6% costs, one-home capital-gains tax above KRW 1.2B, and acquisition tax on the new home; released cash joins financial assets.


### v20 accumulation phase

Pre-retirement savings = household gross salary growing at a real wage rate × saving rate (or a fixed amount); workplace pension adds one-twelfth of salary per year to the tax-deferred pension account, withdrawable after 55 and retirement. National Pension is estimated with the 2026 reform formula (43% replacement, constant 1.29, A ≈ KRW 3.19M, income cap/floor KRW 6.59M/0.41M). A bisection solver finds the saving (amount or rate) that keeps essential success at 90%.


### v21 big expenses

Expense events (education, wedding support, other) are added to required spending over their years. A home purchase pays the down payment and acquisition tax from assets (a returned jeonse deposit counts as equity) and takes a fixed-rate amortizing mortgage at the prevailing rate + 1.5pp; payments before retirement come from earned income, remaining payments after retirement from savings. Pre-retirement shortfalls become debt at rate + 1.5pp, repaid first from later savings and settled from assets at retirement.


### v26 house price model

Real house price index follows $\ln(H_{t+1}/H_t)=\ln(1+g)-\sigma_H^2/2+\sigma_H(\rho Z^S+\sqrt{1-\rho^2}Z^H)$ with $g\sim N(1.25\%,1.18\%^2)$, $\sigma_H=10\%$, $\rho=0.1$; nominal price = real index × CPI. Estimated from the Korea Real Estate Board national house price index via ECOS, 2013 onwards only (the series provider changed in 2013). Monthly volatility is smoothed (lag-1 autocorrelation 0.82), so volatility is re-estimated from annual returns (6.6%) and combined with idiosyncratic single-home risk (7%). The real growth rate is a Bayesian posterior of the sample estimate (3.25% ± 1.89pp) and a prior of N(0%, 1.5%).


### v27 actuarial refinements

**Mortality improvement (cohort):** $q(x,t)=q_{2024}(x)(1-k_x)^t$ with age-graded $k_x$ (2.5% below 65 down to 0.3% above 100), calibrated so that period life expectancy at birth rises +0.19/+0.14 years per calendar year, matching the 2013–2024 Statistics Korea trend. Life expectancy at 60 rises from 23.7 to 25.7 years (male).

**CTE:** conditional tail expectation of the real lifetime shortfall at the 70% and 90% levels, the same definition used in US statutory capital requirements.

**Certainty-equivalent consumption:** CRRA utility ($\gamma=3$, $\beta=1/1.02$) over per-person real consumption, reported as the constant annual consumption of equal utility.

**Annuity variants:** single/joint life with a survivor percentage, guarantee periods, escalating payments and deferred (longevity-insurance) start ages, all priced by the equivalence principle with a 0.8 mortality multiplier and 5% loading.
