# 실험 기록 - Playground Series S6E5

**대회**: Predicting F1 Pit Stops
**평가지표**: ROC AUC
**기간**: 2026-05-01 ~ 2026-05-31
**타깃**: `PitNextLap` (이진, 양성 비율 19.9%)

---

## 스코어 추이

| 실험 | OOF AUC | LB | CV-LB 격차 | 비고 |
|------|---------|-----|-----------|------|
| EXP-001 XGB Baseline | 0.94906 | — | — | std 0.00076 — fold 일관 |

---

## 실험 로그

### EXP-001: XGBoost Baseline
- **노트북**: `xgb_baseline.ipynb`
- **설정**: 5-Fold StratifiedKFold, lr=0.05, max_depth=8, n_estimators=1000, subsample/colsample 0.8
- **전처리**: LabelEncoder (Driver/Compound/Race) — Driver test에 unseen 없음
- **Fold 점수**: 0.95031 / 0.94819 / 0.94905 / 0.94840 / 0.94937 → **Mean 0.94907 ± 0.00076**
- **Submission**: mean prob 0.1973 (실제 양성률 0.1990과 거의 일치 — 캘리브레이션 양호)
- **OOF AUC**: **0.94906** / **LB**: —

---

## 학습된 교훈

---

## TODO
- [x] 데이터 다운로드 (`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`)
- [x] EDA 노트북 (`eda.ipynb`)
- [x] Baseline 학습/제출 (`xgb_baseline.ipynb`) — OOF 0.94906
- [ ] `PitStop` 컬럼 정의 재확인 (PitStop=1 → PitNextLap=1 비율 24.8% 의외)
- [ ] LB 점수 기록 (Kaggle 제출 후)
- [ ] EXP-002: 학습 설정 개선 (early stopping, 10-Fold, lr 튜닝)
