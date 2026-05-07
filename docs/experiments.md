# 실험 기록 - Playground Series S6E5

**대회**: Predicting F1 Pit Stops
**평가지표**: ROC AUC
**기간**: 2026-05-01 ~ 2026-05-31
**타깃**: `PitNextLap` (이진, 양성 비율 19.9%)

---

## 스코어 추이

| 실험 | OOF AUC | LB | CV-LB 격차 | 비고 |
|------|---------|-----|-----------|------|
| EXP-001 XGB Baseline | 0.94906 | 0.94874 | 0.00032 | std 0.00076 — fold 일관, 격차 매우 작음 |

---

## 실험 로그

### EXP-001: XGBoost Baseline
- **노트북**: `xgb_baseline.ipynb`
- **설정**: 5-Fold StratifiedKFold, lr=0.05, max_depth=8, n_estimators=1000, subsample/colsample 0.8
- **전처리**: LabelEncoder (Driver/Compound/Race) — Driver test에 unseen 없음
- **Fold 점수**: 0.95031 / 0.94819 / 0.94905 / 0.94840 / 0.94937 → **Mean 0.94907 ± 0.00076**
- **Submission**: mean prob 0.1973 (실제 양성률 0.1990과 거의 일치 — 캘리브레이션 양호)
- **OOF AUC**: **0.94906** / **LB**: **0.94874** (격차 0.00032)

---

## 검증 전략 점검 (EXP-001 직후)

### Train/Test split 구조
| 단위 | overlap |
|------|---------|
| `(Year, Race)` 104개 | **train, test 양쪽 모두 100% 공유** |
| `(Year, Race, Driver)` | shared 35,674 / train-only 5,195 / test-only 1,364 |
| `(Year, Race, Driver, Stint)` | **shared 72,859** / train-only 40,708 / test-only 10,953 |
| `id` 순서 | random shuffle (시계열 정보 없음) |

→ **사실상 row-level random split**. 같은 stint의 lap들이 train/test에 흩어져 있음 — TyreLife/LapNumber 등 stint 내부 패턴이 양쪽에 거의 보존됨. 이것이 baseline 0.94874의 근본 원인.

### Random 5-Fold vs GroupKFold(Year, Race) 비교

| 검증 방식 | OOF AUC | Fold std | LB와 격차 |
|---|---|---|---|
| **Random 5-Fold** (현재) | 0.94906 | 0.00076 | **0.00032** |
| GroupKFold by (Year, Race) | 0.93263 | 0.00766 | ~0.016 |

→ **결론: Random KFold가 LB 정합. 그대로 유지.**
→ GroupKFold OOF(0.93263)는 "본 적 없는 레이스에서의 일반화 성능"이라는 다른 의미 — 모델 과적합 점검용 보조 지표로만 활용.

---

## 학습된 교훈

- **이 대회는 row-level random split** — random KFold가 LB와 일관됨 (CV-LB 격차 0.00032).
- 같은 stint의 lap들이 train/test에 분산돼 있어 baseline이 자연히 강함. 이는 누수가 아니라 데이터 분리 방식의 결과.
- GroupKFold는 본 대회 LB와 매칭되지 않음 — 의도적 보수적 검증으로만 의미.

---

## TODO
- [x] 데이터 다운로드 (`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`)
- [x] EDA 노트북 (`eda.ipynb`)
- [x] Baseline 학습/제출 (`xgb_baseline.ipynb`) — OOF 0.94906 / LB 0.94874
- [x] 검증 전략 점검 — Random KFold 유지 결정
- [ ] `PitStop` 컬럼 정의 재확인 (PitStop=1 → PitNextLap=1 비율 24.8% 의외)
- [ ] EXP-002: 학습 설정 개선 (early stopping, 10-Fold, lr 튜닝)
