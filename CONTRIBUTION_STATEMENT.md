# Contribution Statement

This project was prepared by:

- Tushar Singh (22322032)
- Pranav Pipariya (22322022)
- Tanishq Gupta (22322031)
- Kavish Jain (22322017)
- Yash Kumar (22322034)

## Individual Contributions

Each member led one labeled empirical investigation in the Phase 2 notebook, in line with the project requirement of at least one question per team member. The statements below follow the team roster order; the notebook itself labels each investigation with its lead.

The division of work was organized around distinct empirical questions rather than only around mechanical tasks. Each lead was responsible for defining the relevant analytical cohort, producing the main tables and figures for that question, and writing the interpretation and limitations for that section. The final notebook was then reviewed as a single integrated report so that the five investigations support the same central research question.

**Tushar Singh** led the pending-burden and COVID-period investigation, examining how pending case counts and pending durations vary across courts, bail types, and filing years. He produced the court-level pending-rate summary, the bail-type pending-rate summary, the yearly filings and pending-rate trend, the period-by-bail comparison of pre-COVID and COVID-window filings, the monthly filings time-series from 2018 through 2021 with the 25 March 2020 lockdown marked, and the Mann-Whitney U test on disposal-day distributions across the two periods with a rank-biserial effect size, plus the interpretation of right-censoring and registry-behavior caveats that prevents the comparison from being read as a clean causal estimate.

**Pranav Pipariya** led the long-delay risk modeling investigation, examining whether filing-stage metadata can predict when a disposed case crosses a long-delay threshold. He produced the temporal hold-out logistic model implemented in dependency-free NumPy, the p90 robustness check, the leakage-control design with an explicit feature-set assertion, the lift table, the calibration-by-decile analysis, the coefficient interpretation, the court+bail historical-rate baseline, and the bootstrap 95% AUC confidence intervals (400 resamples each) showing that the logistic and baseline AUCs are statistically indistinguishable, which is what motivates framing the model as evidence of structural delay signal rather than as a deterministic decision tool.

**Tanishq Gupta** led the bail-type landscape investigation, examining how regular bail, anticipatory bail, and cancellation cases are distributed across High Courts, filing years, and case-type conventions. He produced the overall bail-type counts and shares, the cross-court bail-mix percentage table, the top-case-types-by-bail breakdown showing which case-type labels dominate within each bail category, the status-mix-by-bail table that connects later cohort definitions, the supporting figures on bail-type composition, filed cases by year and bail type, and the largest-courts bail mix, and the interpretation of registry-convention heterogeneity that motivates the cohort separation used throughout the rest of the notebook.

**Kavish Jain** led the disposal-delay and court-inequality investigation, examining how disposal time varies across bail types and High Courts. He produced the median, p75, p90, and p99 delay summaries by bail type and by court, the supporting figures comparing disposal delay across bail types and across courts, the design of the adjusted court delay index using a bail-by-year-by-case-type stratum with a bail-by-year fallback gated by sample-size and court-coverage thresholds, the bootstrap 95% confidence intervals for the adjusted index (500 resamples per court), and the interpretation explaining why raw court rankings need composition-aware context paired with stability checks before they can support cross-court comparison.

**Yash Kumar** led the observed-outcome investigation, examining how disposal-outcome labels - granted, rejected, withdrawn, and other - distribute across bail types and courts within the labeled subset. He produced the regex-based outcome-grouping logic that maps raw label strings into the four interpretable categories, the court-level outcome-label coverage table reporting where labels are populated, the outcome mix by bail type and by court, the chi-square test of independence between bail type and outcome group with Cramer's V as an effect size, the standardized residuals identifying the cells that drive the dependence, and the caveats explaining why restricted-sample outcome patterns should not be generalized to the full bail population.

## Shared Contributions

All members contributed to dataset selection, Phase 1 exploration, framing of the five empirical questions, validation checks, and review of the final write-up. The cross-cutting regional analysis (mapping each High Court to a broad geographic region and comparing volume, pending rate, and median disposal days across regions) was developed jointly during the final integration stage. Each empirical investigation in the Phase 2 notebook is labeled with its lead so the per-member contribution is reproducible from the notebook itself.

The shared review focused on consistency across sections: using the same cleaned bail-type categories, keeping full-cohort and restricted-cohort claims separate, checking that figures and tables were regenerated by the notebook, ensuring inferential statistics (bootstrap CIs, Mann-Whitney, chi-square) were paired with effect sizes, and making sure limitations were stated wherever the dataset does not support stronger conclusions.
