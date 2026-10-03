# Presenting this portfolio

## A five-minute walkthrough

1. **Problem coverage (30 seconds).** Show the seven-project gallery. Explain the distinction
   between regression, binary classification, multiclass classification, and text classification.
2. **One case study (90 seconds).** Open a notebook. State the target, who could use the prediction,
   and the cost of a false positive or large regression error.
3. **Evaluation credibility (90 seconds).** Show the audit, training-only pipeline, CV search,
   dummy baseline, and diagnostics. Explain why GPA and repeated heart records were excluded.
4. **Engineering (60 seconds).** Run `python -m ml_portfolio run sms-spam`. Show reports, tests,
   and CI. Trace one loader → pipeline → report path in the source.
5. **Next step (30 seconds).** Explain the biggest limitation and the independent experiment
   needed to address it. Avoid presenting educational benchmarks as production systems.

## Questions worth preparing for

| Question | Evidence to show |
| --- | --- |
| How do you prevent leakage? | Pipeline fitting inside CV; GPA exclusion; duplicate-disjoint tests |
| Why not report accuracy alone? | Class distribution, macro F1, minority-class recall |
| Does the model add value? | Same-holdout dummy baseline and error diagnostics |
| Can another person reproduce this? | Setup commands, fixed seed, numerical dependency pins, SHA-256 |
| Where does this model fail? | Confusion matrix, residuals, training/test gap, limitations |
| What would deployment require? | Verified data rights, inference contracts, external validation, monitoring |

## Suggested portfolio description

“A collection of seven supervised-learning case studies built with Python and scikit-learn.
The repository includes reusable preprocessing and training pipelines, training-only model
selection, held-out evaluation against dummy baselines, executable notebooks, and automated
quality checks. Each case study documents its assumptions and limitations.”

Only claim work and understanding you can explain. Use the actual report values when discussing
performance; do not invent business savings, client deployments, or personal contributions.

## Client handoff criteria

Agree on the business decision and error costs; verify data permissions; define available inputs
and validation populations; assess subgroup errors; establish an inference interface; and document
ownership, monitoring, and retraining responsibilities. This repository supplies a reproducible
experimentation foundation, not those deployment guarantees.
