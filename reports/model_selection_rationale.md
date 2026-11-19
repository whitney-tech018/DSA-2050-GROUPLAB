# Week 9 — Model-selection rationale and written reflection

See the README for the full results and the notebook for the code.

## Answers — Group written reflection (notebook version)

**1. Which model was selected and why was it defensible?** Logistic regression with `C = 1`, selected on F1 (agreed before comparing). Its CV mean F1 was **0.470 with fold SD 0.053**, against **0.398 (SD 0.049)** for the best decision tree (`max_depth = 3`). The 0.072 advantage is larger than the fold spread; the model is clearly ahead in three folds and within 0.003 in the other two; its variability is not worse than the tree's; and it has higher precision and recall. It is also simpler and more explainable. The choice was frozen in `reports/frozen_selection.json` before the test file was read.

**2. What did false negatives mean for this provider? Which subgroup struggled most?** A false negative is a customer who was predicted to stay but churned: no retention offer is made and the customer is lost unexpectedly. Out-of-fold, the model missed 566 of 910 churners (62 %). **Established customers struggled most**: recall 0.22 (509 false negatives among 649 churners) versus 0.78 for New customers (57 among 261). Both groups are large, so this is a real pattern rather than a small-sample effect.

**3. Highest permutation importance, and why this is not causation.** **`tenure_months`** (mean F1 drop ≈ 0.079), then `support_tickets_90d` (≈ 0.035). Importance shows what the model *relies on to predict*, not what *causes* customers to leave: tenure and tickets may be symptoms or markers of other things (introductory deals, dissatisfaction, competitor offers). It is also shared among correlated features (tenure and customer segment) and measured on a small, noisy slice.

**4. Final-test F1 vs mean CV F1.** Test F1 = 0.449 vs CV 0.470, a drop of 0.021, which is under half of one CV SD. The model generalised reasonably; the gap is not concerning, though the absolute level (recall 0.36) is modest.

**5. Two limitations and one pre-deployment step.** *Limitations:* (a) the data are **synthetic**, so the relationships, base rate and segment patterns need not resemble real customers, and conclusions cannot be transferred to a real provider; (b) it is a **static snapshot** with only eight features, so hidden dynamics (competitor offers, network outages, price changes, seasonality) and **temporal drift** are not captured, and a model trained on past customers can decay. *Before deploying a real churn model:* run a monitored pilot with **human review** — track recall and precision by segment, choose the decision threshold using the real costs of a missed churner versus an unnecessary offer, and confirm that handling customer data meets privacy rules.

## Answers — Handout reflection prompts
1. **Why should models share folds?** So the models are compared on identical validation data; differences in score then reflect the models, not a lucky or unlucky split. It also makes the comparison paired (fold by fold) and fair.
2. **What does F1 measure?** The harmonic mean of precision and recall, F1 = 2·P·R / (P + R). It rewards a model only if it both catches churners (recall) and is right when it flags them (precision), and it ignores the large number of easy true negatives, so it is suited to an imbalanced target.
3. **Where do errors concentrate?** In missed churners: false negatives (566) outnumber false positives (205), and 509 of the 566 missed churners are Established customers.
4. **What is the highest-reliance feature?** `tenure_months` (≈ 0.079 mean F1 drop when shuffled).
5. **Why is the final-test score not a tuning tool?** Once a test score guides any change, the test set becomes part of development; the score is then optimistic and there is no unbiased estimate left. It is used once, after freezing, and weaknesses are reported as limitations.

## Answer — Final test (run once; `reports/final_metrics.json`)

| Metric | Final test (n = 680) | Development CV (mean) |
|---|---|---|
| **F1** | **0.449** | 0.470 (SD 0.053) |
| **Precision** | **0.585** | 0.624 |
| **Recall** | **0.364** | 0.378 |
| Accuracy | 0.700 | — (OOF 0.72) |

Test confusion matrix: 393 true negatives, **59** false positives, **145** false negatives, 83 true positives (228 actual churners in the test set).

**CV vs test.** Test F1 (0.449) is 0.021 below the CV mean (0.470). That drop is less than half of one fold-to-fold SD (0.053), so it is well within the normal variation we already saw between validation folds; precision and recall moved by similar small amounts (−0.04 and −0.01). A small drop is expected, because the CV score comes from data used to choose the model, whereas the test set was untouched. **This is not concerning**: there is no sign of overfitting or leakage, and the model generalised reasonably. The weakness is not the CV-to-test gap but the modest level of performance: the model still misses about 64 % of churners on the test set (145 of 228), confirming the false-negative problem found in development.

The final-test result was **not** used to change the model, grid, metric or threshold. Any weakness it shows is recorded as a limitation, not fixed and re-scored.
