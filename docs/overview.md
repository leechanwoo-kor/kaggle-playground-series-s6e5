# Overview

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict whether a Formula 1 driver will pit on the next lap.

## Evaluation
Submissions are evaluated on [area under the ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) between the predicted probability and the observed target.

### Submission File
For each `id` in the test set, you must predict a probability for the `PitNextLap` target. The file should contain a header and have the following format:

```
id,PitNextLap
439140,0.2
439141,0.3
439142,0.9
etc.
```

## Timeline
- **Start Date** — May 1, 2026
- **Entry Deadline** — Same as the Final Submission Deadline
- **Team Merger Deadline** — Same as the Final Submission Deadline
- **Final Submission Deadline** — May 31, 2026

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted.

## Prizes
- 1st / 2nd / 3rd Place — Choice of Kaggle merchandise (한 번 수상 시 동일 시리즈에서 재수상 불가)

## Citation
Yao Yan, Walter Reade, Elizabeth Park. Predicting F1 Pit Stops. https://kaggle.com/competitions/playground-series-s6e5, 2026. Kaggle.
