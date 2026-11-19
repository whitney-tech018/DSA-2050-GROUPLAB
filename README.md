# DSA 2050A — Week 9: Model selection and defensible evaluation

**Group name/number:** [TO FILL — e.g. Group XX]
**Members and student IDs:** [TO FILL — list all group members with student IDs]
**Submission:** One shared GitHub repository link per group
**Course:** DSA 2050A Data Science Methodology
**Week:** 9

## Group responsibilities
Completed in the existing group on one notebook and one repository. Each member's concrete contribution is recorded below (edit the names, IDs, work split and usernames to match what your group actually did; every member should have a visible Git commit).

| Group member | Student ID | Work completed | GitHub username or commit reference |
|---|---|---|---|
| Victor Apamo | [ID] | [e.g. data inspection, leakage exclusions, pipeline] | [username] |
| [Name] | [ID] | [e.g. model comparison, experiment log, selection rationale] | [username] |
| [Name] | [ID] | [e.g. error and subgroup analysis, permutation importance] | [username] |
| [Name] | [ID] | [e.g. final test run, README, repository set-up] | [username] |

## Study objective
A fictional Kenyan internet provider wants to identify customers likely to cancel (churn) so it can offer retention help. The goal is not the highest score but to **choose, document and defend one model using development data only**, then evaluate it **once** on an untouched final test set. Why selection must use development data only: the test set stands in for future customers; if it influences any choice, the final score becomes optimistic and there is no unbiased estimate left.

## Data statement
Synthetic fictional customer data created for teaching. No real customer records.
- `data/churn_development.csv` — 2,720 customers, 33.5 % churn (910 churners, 1,810 stayers). Missing values: `monthly_charges_kes` 156, `support_tickets_90d` 120, `region` 93.
- `data/churn_final_test.csv` — 680 customers, read **only** in the last notebook cell.

## Setup and reproduction
1. Open this folder in VS Code.
2. Create/select a Python environment (3.10–3.13 recommended): `python -m venv .venv` (Windows: `py -m venv .venv`), then Ctrl+Shift+P → *Python: Select Interpreter* → `.venv`.
3. In **Terminal → New Terminal**, run `python -m pip install -r requirements.txt`.
4. Open `notebooks/Week9_Model_Selection_Lab.ipynb` and **Select Kernel** (the same `.venv`).
5. Run each cell in order. The rationale is completed before the final-test cell. The final-test cell is protected: if `reports/final_metrics.json` exists it prints the saved result instead of re-testing (delete that file only to reproduce from scratch, and never to improve the score).

## Prediction-time features and excluded fields
- Unit of analysis: one customer account. Target: `churn` (1 = left; 0 = stayed).
- Excluded: `customer_id` (arbitrary identifier, no generalisable signal), `cancellation_confirmed` (post-outcome leakage: "Cancelled" for all 910 churners and "Active" for all 1,810 stayers, a perfect copy of the target recorded only after the event).
- Allowed features: `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, `customer_segment` — all known before a customer cancels. Numeric: median imputation + standard scaling; categorical: most-frequent imputation + one-hot encoding, all inside a `Pipeline` so it is re-fitted within each fold.
- Baseline: a majority-class dummy has F1 = 0.000 in every fold (it never flags a churner) even though its accuracy would be about 66.5 %.

## Experiments and model selection
- Primary metric chosen before comparison: **F1**.
- CV: **5-fold stratified, shuffle=True, seed=42**, the same `StratifiedKFold` object for every model.
- Full log: `reports/experiment_log.csv`. Best per family:

| Model | Setting | Mean F1 | SD F1 | Mean precision | Mean recall | Fold F1 scores (1–5) |
|---|---|---|---|---|---|---|
| **Logistic regression** | **C = 1** | **0.470** | **0.053** | 0.624 | 0.378 | 0.391, 0.534, 0.472, 0.434, 0.519 |
| Decision tree | max_depth = 3 | 0.398 | 0.049 | 0.566 | 0.318 | 0.392, 0.457, 0.316, 0.437, 0.391 |

Other grid points: LR C = 0.1 → 0.465 (SD 0.054), C = 10 → 0.470 (identical to C = 1); tree depth 5 → 0.382 (SD 0.023), depth 8 → 0.372 (SD 0.051).

- **Chosen model:** Logistic regression
- **Tuned hyperparameters:** `C = 1` (C = 1 and C = 10 tied; the more regularised value was kept)
- **Mean/SD CV F1:** 0.470 / 0.053
- **Why this choice (approx. 140 words):** We selected logistic regression (C=1) using F1, the metric agreed before comparing models. Its 5-fold stratified CV mean F1 was 0.470 (SD 0.053) against 0.398 (SD 0.049) for the best decision tree (max_depth=3). The 0.072 gap is larger than the fold spread, the model is ahead or level in all five folds, and its variability is no worse than the tree's, so the advantage is not luck. It also had higher precision (0.62 vs 0.57) and recall (0.38 vs 0.32). Out-of-fold, it missed 566 of 910 churners (62%) and recall was uneven: 0.22 for Established customers (509 false negatives) versus 0.78 for New customers; we accept this as a documented limitation and a reason for monitoring and threshold review. Logistic regression is also simple, fast and easy to explain to the business. C=1 and C=10 tied, so we kept the more regularised C=1. (A 0.005 gain would not have been convincing given a fold SD of about 0.05; a 0.072 gain is.)

## Error and subgroup analysis
Out-of-fold on development data (chart: `reports/development_confusion_matrix.png`):

| | Predicted stay | Predicted churn |
|---|---|---|
| Actually stayed | 1,605 | **205** (false positives) |
| Actually churned | **566** (false negatives) | 344 |

- False negatives = churners predicted to stay (no retention offer, customer lost unexpectedly); false positives = loyal customers flagged as high risk (a wasted offer). The model misses 62 % of churners; accuracy (0.72) hides this.
- Subgroups (`reports/subgroup_metrics.csv`):

| Segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---|---|---|---|---|
| Established | 2,268 | 649 | **0.216** | 0.593 | 509 |
| New | 452 | 261 | **0.782** | 0.652 | 57 |

- Recall is **worse for Established customers, not New ones**; both groups are large, so this is not a small-sample artefact. A likely (unproven) reason: new customers churn much more (58 % vs 29 %) and short tenure is the model's main signal, so established churners look "loyal" and are missed (509 of the 566 false negatives).
- Business implication: retention effort guided by this model would reach most at-risk new customers but miss about four in five at-risk established customers; consider segment-specific thresholds/models and track recall by segment.

## Permutation importance
Held-out 25 % slice of development data, F1 scoring, 5 repeats (`reports/permutation_importance.csv`, `reports/permutation_importance.png`):
- Top two features: **`tenure_months` ≈ 0.079** and **`support_tickets_90d` ≈ 0.035** mean F1 drop (next: `customer_segment` ≈ 0.018).
- Association is not causation: importance shows what the model relies on to predict, not what makes customers leave (tenure and tickets could be markers of promotions, dissatisfaction or competitor offers). Correlated features (e.g. tenure and customer segment) share importance, and the estimate is noisy (small slice, 5 repeats).

## Final test (run once only)
- F1: **0.449** ; precision: **0.585** ; recall: **0.364** (accuracy 0.700; n = 680; confusion matrix [[393, 59], [145, 83]]) — `reports/final_metrics.json`.
- CV vs test: CV mean F1 0.470 vs test 0.449, a drop of 0.021, less than half a CV standard deviation (0.053). This is **not concerning**: no sign of overfitting, and the model generalised reasonably. The concern is the modest level — the model still misses about 64 % of churners (145 of 228). The test result was not used to change anything.

## Limitations and ethics
- **Synthetic data:** relationships, base rates and segment effects may not resemble a real provider; results do not transfer.
- **Static snapshot, few features:** hidden customer dynamics (competitor offers, outages, price changes) and **temporal drift** are not captured; a model trained on past customers can decay.
- **Unequal performance across segments** (recall 0.22 vs 0.78) means retention effort could be unevenly distributed.
- **Before real use:** pilot with monitoring and **human review**, track precision/recall by segment and over time, set the decision threshold from the real costs of a missed churner vs an unnecessary offer, and check that customer-data handling meets privacy law.

## Evidence checklist
- [x] Executed notebook with explanations
- [x] `reports/experiment_log.csv`
- [x] `reports/subgroup_metrics.csv`
- [x] `reports/permutation_importance.csv`
- [x] `reports/frozen_selection.json` created before test
- [x] `reports/final_metrics.json` generated once
- [x] Two generated PNG charts
- [ ] Meaningful commits or co-authorship evidence for all group members
- [ ] Completed group contribution table with names and student IDs

## Final group review
- [ ] One team member verified the notebook runs from start to finish
- [ ] Group discussed and approved the written model-selection rationale
- [ ] No one used final-test scores to revise the model
- [ ] One representative will submit the shared URL to the course LMS

## GitHub repository URL
[Paste URL here after publishing]
