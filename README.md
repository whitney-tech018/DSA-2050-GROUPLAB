# DSA 2050A — Week 9: Model Selection and Defensible Evaluation

**Group name/number:**  
**Members and student IDs:**
Victor Apamo
Whitney Jebor
Enzo Sylvain

**Submission:** One shared GitHub repository link per group  
**Course:** DSA 2050A Data Science Methodology  
**Week:** 9

## Group responsibilities

The project was completed collaboratively using one shared notebook and one GitHub repository. The following table outlines the main responsibilities undertaken by each group member.

| Group member | Student ID | Work completed | GitHub username or commit reference |
|---|---|---|---|
| Victor Apamo | [ID] | Data inspection, preprocessing and leakage prevention |
| Enzo Sylvain| [] | Model development, cross-validation and model selection |
| [Whitney Jebor] | [673308] | Error analysis, subgroup evaluation, permutation importance and documentation | 


## Study objective

A fictional Kenyan internet provider wants to identify customers likely to cancel (churn) so it can offer retention help. The goal is not simply to achieve the highest score, but to **choose, document and defend one model using development data only**, then evaluate it **once** on an untouched final test set.

The final test set represents future, unseen customers. If it influences model selection or tuning, the final performance estimate becomes biased and is no longer an independent evaluation.

## Data statement

Synthetic fictional customer data created for teaching. No real customer records.

- `data/churn_development.csv` — 2,720 customers, 33.5% churn (910 churners, 1,810 stayers). Missing values: `monthly_charges_kes` 156, `support_tickets_90d` 120, `region` 93.
- `data/churn_final_test.csv` — 680 customers, read **only** in the last notebook cell.

## Setup and reproduction

1. Open this folder in VS Code.
2. Create/select a Python environment (3.10–3.13 recommended):

```bash
python -m venv .venv
```

On Windows:

```bash
py -m venv .venv
```

3. Select the `.venv` interpreter using **Ctrl+Shift+P → Python: Select Interpreter**.
4. In **Terminal → New Terminal**, run:

```bash
python -m pip install -r requirements.txt
```

5. Open `notebooks/Week9_Model_Selection_Lab.ipynb` and select the same `.venv` as the notebook kernel.
6. Run each cell in order. Complete the rationale before the final-test cell.

The final-test cell is protected: if `reports/final_metrics.json` exists, it prints the saved result instead of re-testing. Do not delete the file to improve the score or repeat evaluation for model selection.

## Prediction-time features and excluded fields

- **Unit of analysis:** One customer account. Each row represents an individual customer.
- **Target:** `churn` (1 = left; 0 = stayed).
- **Excluded:** `customer_id` (arbitrary identifier with no generalisable predictive signal) and `cancellation_confirmed` (post-outcome leakage: it perfectly reveals the cancellation outcome).
- **Allowed features:** `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, `customer_segment`. These are treated as information available before cancellation.
- **Preprocessing:** Numeric features use median imputation and standard scaling; categorical features use most-frequent imputation and one-hot encoding. All transformations are inside a `Pipeline` so they are fitted within each training fold.
- **Baseline:** A majority-class dummy classifier achieved F1 = 0.000 in every fold because it never predicted a churner, despite the majority class representing 66.5% of customers.

## Experiments and model selection

- **Primary metric:** F1-score, selected before model comparison.
- **Cross-validation:** 5-fold stratified, `shuffle=True`, `seed=42`, using the same `StratifiedKFold` strategy for every model.
- **Full experiment log:** `reports/experiment_log.csv`.

Best configuration per model family:

| Model | Setting | Mean F1 | SD F1 | Mean precision | Mean recall | Fold F1 scores (1–5) |
|---|---|---:|---:|---:|---:|---|
| **Logistic Regression** | **C = 1** | **0.470** | **0.053** | **0.624** | **0.378** | 0.391, 0.534, 0.472, 0.434, 0.519 |
| Decision Tree | max_depth = 3 | 0.398 | 0.049 | 0.566 | 0.318 | 0.392, 0.457, 0.316, 0.437, 0.391 |

Other grid points:

- Logistic Regression C = 0.1: mean F1 = 0.465 (SD = 0.054).
- Logistic Regression C = 10: mean F1 = 0.470 (same as C = 1).
- Decision Tree depth = 5: mean F1 = 0.382 (SD = 0.023).
- Decision Tree depth = 8: mean F1 = 0.372 (SD = 0.051).

**Chosen model:** Logistic Regression  
**Tuned hyperparameter:** `C = 1`  
**Mean CV F1:** 0.470  
**SD CV F1:** 0.053

**Why this choice:**

Logistic Regression with C = 1 was selected because it achieved a higher mean F1-score (0.470) than the best Decision Tree configuration (0.398). It also achieved higher precision (0.624 compared to 0.566) and recall (0.378 compared to 0.318). Its performance was reasonably consistent across the five folds. Although C = 1 and C = 10 tied, C = 1 was retained because it provides stronger regularisation. Logistic Regression was also preferred for its practical simplicity and interpretability. However, the model missed 566 of 910 churners in the out-of-fold predictions, and recall was considerably lower for Established customers (0.216) than New customers (0.782). These limitations were documented before final testing.

## Error and subgroup analysis

Out-of-fold predictions on development data:

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actually stayed | 1,605 | **205** (false positives) |
| Actually churned | **566** (false negatives) | 344 |

- **False negatives:** Churners predicted to stay. The model misses approximately 62.2% of actual churners.
- **False positives:** Customers who stayed but were incorrectly flagged as high risk. This could lead to unnecessary retention offers or discounts.
- The model achieved approximately 71.7% accuracy, but accuracy alone hides its difficulty identifying churners.

Generated confusion matrix: `reports/development_confusion_matrix.png`.

**Subgroup results** (`reports/subgroup_metrics.csv`):

| Segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---:|---:|---:|---:|---:|
| Established | 2,268 | 649 | **0.216** | 0.593 | 509 |
| New | 452 | 261 | **0.782** | 0.652 | 57 |

- Recall is considerably worse for Established customers, not New customers.
- Established customers account for approximately 90% of all false negatives.
- A possible explanation is that tenure and customer segment contain overlapping information, and the model identifies New customer churn more readily. This is a hypothesis, not a proven cause.
- **Business implication:** Retention efforts guided by this model could miss many at-risk Established customers. The provider should investigate the performance gap, monitor recall by segment and consider evaluating segment-specific thresholds.

## Permutation importance

Held-out 25% slice of development data, F1 scoring, five repeats (`reports/permutation_importance.csv`, `reports/permutation_importance.png`).

- **Top feature:** `tenure_months` — approximately 0.079 mean F1 reduction.
- **Second feature:** `support_tickets_90d` — approximately 0.035 mean F1 reduction.
- Next was `customer_segment`, at approximately 0.018.

**Association is not causation:** Permutation importance shows which features the model relies on for prediction, not what directly causes customers to leave. For example, support tickets may indicate dissatisfaction without necessarily causing cancellation.

**Correlated features can share importance:** Features such as `tenure_months` and `customer_segment` may contain overlapping information. Shuffling one can have a limited effect when the model can still rely on the other.

## Final test (run once only)

The selected Logistic Regression model was evaluated once on the untouched final test dataset (680 customers).

| Metric | Final test | Development CV (mean) |
|---|---:|---:|
| **F1** | **0.449** | 0.470 (SD 0.053) |
| **Precision** | 0.585 | 0.624 |
| **Recall** | 0.364 | 0.378 |
| Accuracy | 0.700 | 0.720 (OOF) |

Final test confusion matrix: 393 true negatives, 59 false positives, 145 false negatives and 83 true positives (228 actual churners).

**CV versus test:** The final test F1-score of 0.449 was 0.021 below the development CV mean of 0.470. This difference is approximately 0.40 standard deviations, within the variation observed across the five folds. The model therefore demonstrated reasonably consistent performance on unseen data.

However, it still missed approximately 63.6% of actual churners (145 out of 228), confirming the false-negative problem.

The final test result was not used to change the model, grid, metric or classification threshold. The results are saved in `reports/final_metrics.json`.

## Limitations and ethics

- **Synthetic data:** The relationships, base rates and segment effects may not resemble those of a real provider. Results should not automatically be generalised to real customers.
- **Limited features:** Important factors such as competitor offers, network quality and price changes are not captured.
- **Unequal subgroup performance:** Recall differs substantially between Established and New customers (0.216 versus 0.782), meaning retention efforts could be unevenly distributed.
- **Changing customer behaviour:** Customer patterns may change over time, potentially reducing model performance.

**Before real use:** The provider should conduct further validation, monitor precision and recall across customer segments and over time, consider the costs of missed churners versus unnecessary offers, and include human review in retention decisions. Customer information must also be handled in accordance with applicable privacy requirements.

## Reflection prompts

**1. Why should models share folds?**

All models should use the same cross-validation folds to ensure a fair comparison. This means each model is evaluated on the same training and validation splits, reducing differences caused by how the data is divided.

**2. What does F1 measure?**

F1-score is the harmonic mean of precision and recall. It balances the proportion of predicted churners who actually churned (precision) with the proportion of actual churners correctly identified (recall). It is particularly useful when dealing with imbalanced classes.

**3. Where do errors concentrate?**

Errors are concentrated among Established customers. The model produced 509 false negatives in this group, compared to 57 among New customers. Established customers account for approximately 90% of all false negatives.

**4. What is the highest-reliance feature?**

`tenure_months` was the feature with the highest permutation importance, producing an approximate mean F1 reduction of 0.079 when shuffled. This indicates that the model relies heavily on this feature for its predictions, although it does not establish causation.

**5. Why is the final-test score not a tuning tool?**

The final test set is intended to provide an independent evaluation of the selected model on unseen data. Using its results to adjust hyperparameters, features or classification thresholds would introduce bias and compromise the reliability of the final performance estimate.

## GitHub repository URL

[Paste shared repository URL here after publishing]
