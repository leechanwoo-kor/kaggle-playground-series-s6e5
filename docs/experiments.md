# 실험 기록 - Playground Series S6E5

**대회**: Predicting F1 Pit Stops
**평가지표**: ROC AUC
**기간**: 2026-05-01 ~ 2026-05-31
**타깃**: `PitNextLap` (이진, 양성 비율 19.9%)

---

## 스코어 추이

| 실험 | OOF | LB | CV-LB 격차 | 비고 |
|------|-----|-----|-----------|------|
| — | — | — | — | — |

---

## 실험 로그

### EXP-001: (Baseline)
- **노트북**: `xgb_baseline.ipynb`
- **설정**: TODO
- **OOF**: — / **LB**: —

---

## 학습된 교훈

---

## TODO
- [x] 데이터 다운로드 (`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`)
- [x] EDA 노트북 (`eda.ipynb`)
- [ ] `PitStop` 컬럼 정의 재확인 (PitStop=1 → PitNextLap=1 비율 24.8% 의외)
- [ ] Baseline 학습/제출 (`xgb_baseline.ipynb`)
