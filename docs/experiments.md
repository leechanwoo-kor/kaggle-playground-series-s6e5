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
| EXP-002 XGB+LGB+CAT equal blend | 0.95010 | 0.94942 | 0.00068 | 단일 최고 0.94913 → 블렌딩 OOF +0.001 / LB +0.00068 |
| EXP-002 단일: XGB | 0.94913 | — | — | n_est=2000+ES200, best_iter mean 865 |
| EXP-002 단일: LGB | 0.94906 | — | — | best_iter mean 1573 |
| EXP-002 단일: CAT | 0.94880 | — | — | best_iter mean 1993 — **수렴 전 천장 도달** (n_est↑ 여지) |
| EXP-003 GPU + CAT iter↑ blend | 0.95030 | 0.94964 | 0.00066 | EXP-002 + GPU. OOF Δ +0.00020 → LB Δ +0.00022 (거의 1:1 전이) |
| EXP-003 단일: CAT | 0.94933 | — | — | best_iter mean 3485 (max 10000+ES200) |
| EXP-004 lr=0.02 blend equal | 0.95042 | 0.94967 | 0.00075 | OOF Δ +0.00012 → LB Δ +0.00003 (25% 전이, 다양성↓) |
| **EXP-004 blend opt (0.25/0.25/0.50)** | **0.95048** | **0.94977** | 0.00071 | LB 신기록. 처음으로 opt가 equal보다 LB 우위 (+0.00010) — CAT 비중 강조가 LB 전이 |
| EXP-004 단일: XGB | 0.94946 | — | — | best_iter mean 2288 |
| EXP-004 단일: LGB | 0.94933 | — | — | best_iter mean 3927 |
| EXP-004 단일: CAT | 0.94961 | — | — | best_iter mean 9187 (max 20000+ES200) |

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

## EXP-002: Multi-model + Blending

- **노트북**: `exp002_multimodel.ipynb`
- **설정**: 5-Fold StratifiedKFold (seed=42, EXP-001과 동일 split). 모두 lr=0.05, n_est/iterations=2000, early_stopping=200.
  - XGB / LGB: LabelEncoder 처리
  - CAT: native categorical (`cat_features`)
- **개별 OOF**: XGB 0.94913 / LGB 0.94906 / CAT 0.94880
- **블렌딩**:
  - Equal (1/3, 1/3, 1/3) → **0.95010**
  - OOF 가중치 격자 탐색 (0.30, 0.30, 0.40) → 0.95011 (≈ equal, 과적합 미미)
- **CatBoost는 best_iter=1993~1999** (`iterations=2000` 한계) → **수렴 전 천장 도달**, n_est 늘리면 추가 개선 여지
- **소요시간**: XGB 62s / LGB 53s / **CAT 1504s (25분)** — CPU에서 CAT 비용이 압도적

→ 모델 다양성 가설 검증: 단일 0.94913 → 블렌딩 0.95010 (**+0.00097**)
→ Submission 5종 생성: xgb / lgb / cat / blend_equal / blend_opt

---

## EXP-003: GPU + CatBoost iter 확대

- **노트북**: `exp003_gpu_catextend.ipynb`
- **변경점 vs EXP-002**:
  - XGB: `device='cuda'` (재현성 OK, OOF 0.94913 동일)
  - LGB: CPU 유지 (pip wheel GPU 미지원)
  - CAT: `task_type='GPU'` + `iterations=10000` + ES200
- **결과**:
  - CAT OOF 0.94880 → **0.94933** (best_iter mean 1993 → 3485, max 10000+ES200 안에서 자연 수렴)
  - Blend equal 0.95010 → **0.95030** (+0.00020)
  - Blend opt (0.25, 0.30, 0.45) → 0.95035 — CAT 비중 자연 증가
- **소요시간**: EXP-002 26분 → **17분** (CAT 25분 → 14분, iter 2배 늘었음에도 GPU로 단축)
- **교훈**: 단일 모델 천장 풀어주는 것(CAT iter 확대) 만으로도 블렌딩에 +0.00020 전이됨

---

## EXP-004: lr 미세 튜닝 (lr=0.02)

- **노트북**: `exp004_lr_lower.ipynb`
- **변경점 vs EXP-003**: lr 0.05 → **0.02**, XGB n_est 2000 → 5000, LGB n_est 2000 → 5000, CAT iterations 10000 → 20000 (모두 ES200)
- **결과**:
  - XGB 0.94913 → **0.94946** (+0.00033)
  - LGB 0.94906 → **0.94933** (+0.00027)
  - CAT 0.94933 → **0.94961** (+0.00028)
  - Blend equal 0.95030 → **0.95042** (+0.00012)
  - Blend opt (0.25, 0.25, 0.50) → 0.95048 — CAT 비중 0.45 → 0.50 자연 증가
- **소요**: XGB 70s / LGB 115s / **CAT 2715s (45분)** — lr 낮춰서 iter 확대된 부담
- **교훈**: EXP-001 직후 ES 진단(`lr=0.05 천장 ≈ 0.94915`)은 **부정확**했음. ES만 적용한 빠른 진단으로는 lr 변경 효과 측정 못함. 다른 lr + 충분한 트리로 직접 학습해야 정확.
- **Blend 개선 +0.00012는 단일 개선 평균 +0.00029보다 작음** — 세 모델이 더 일관되게 짜질수록 다양성이 줄어 블렌딩 이득 일부 상쇄.

---

## 진단 - Early stopping 수렴점 (EXP-001 직후, 별도 노트북 없음)

EXP-001 동일 설정에서 `n_estimators=5000`+`early_stopping_rounds=200`만 추가하여 자연 수렴점 확인:

| 항목 | EXP-001 | ES diagnostic |
|---|---|---|
| OOF AUC | 0.94906 | **0.94913** (+0.00007) |
| best_iter mean | — (1000 fixed) | **865** (819–932) |
| Fold std | 0.00076 | 0.00073 |

→ **lr=0.05 상한 ≈ 0.94915 부근**. EXP-001은 이미 거의 천장. lr 더 낮춰서 짜려면 별도 시도 필요. 모델 다양성(LGB/CAT)이 더 큰 레버일 가능성.

⚠️ **이 진단은 EXP-004에서 부정확함이 확인됨** — lr=0.02로 XGB OOF 0.94946까지 나옴 (+0.00040). ES 빠른 진단의 한계: lr 변경 효과는 직접 학습으로만 측정 가능.

---

## 학습된 교훈

- **이 대회는 row-level random split** — random KFold가 LB와 일관됨 (CV-LB 격차 0.00032).
- 같은 stint의 lap들이 train/test에 분산돼 있어 baseline이 자연히 강함. 이는 누수가 아니라 데이터 분리 방식의 결과.
- GroupKFold는 본 대회 LB와 매칭되지 않음 — 의도적 보수적 검증으로만 의미.
- **데이터는 row-level synthetic** — `(Year, Race, Driver)` 그룹 내에서 LapNumber 따라 정렬해도 Stint가 70.8% 그룹에서 감소(역행). 즉 `(LapNumber, Stint)` 쌍은 시간 순서를 따르지 않음. `(Stint, TyreLife)` 내부 일관성만 유지됨. **시퀀스/lag 기반 FE 시도는 무의미**.
- `PitStop`은 시간 의미 없는 단순 feature: PitStop=1 vs 0의 LapTime 거의 동일(90.96 vs 90.98), 핏인/아웃 가설 기각. PitStop=1 → 다음 lap Stint 증가 12%뿐. EDA에서 발견한 "PitStop=1 ∧ PitNextLap=1 비율 24.8%"는 합성 데이터의 marginal noise.
- **블렌딩 LB 전이율**: EXP-002 OOF Δ+0.00104 → LB Δ+0.00068 (65%); EXP-003 OOF Δ+0.00020 → LB Δ+0.00022 (1:1); EXP-004 equal OOF Δ+0.00012 → LB Δ+0.00003 (25%); EXP-004 opt → +0.00013 (72%). **단일 모델이 모두 같은 방향으로 개선되면 다양성이 줄어 블렌딩 이득 일부만 LB로 전이**.
- **CV-LB 격차 추세**: 0.00032 → 0.00068 → 0.00066 → 0.00075. 블렌딩이 OOF에 점점 친화적. 추가 OOF 짜기는 위험.
- **Blend opt vs equal**: 모델 OOF가 비등할 땐 둘 다 의미 없는 차이. 모델 격차가 커진 EXP-004부터 opt(CAT 0.50)가 equal 대비 +0.00010 LB 우위. 가중치 강조가 LB로도 전이됨.

---

## TODO
- [x] 데이터 다운로드 (`data/train.csv`, `data/test.csv`, `data/sample_submission.csv`)
- [x] EDA 노트북 (`eda.ipynb`)
- [x] Baseline 학습/제출 (`xgb_baseline.ipynb`) — OOF 0.94906 / LB 0.94874
- [x] 검증 전략 점검 — Random KFold 유지 결정
- [x] `PitStop` / 시퀀스 점검 — 데이터 row-level synthetic, 시퀀스 FE 무의미 결론
- [x] EXP-002: 멀티모델 (XGB+LGB+CAT) 블렌딩 — OOF 0.95010
- [x] EXP-002 LB 점수 기록 — 0.94942 (gap 0.00068)
- [x] GPU 활성화: XGB `device='cuda'`, CAT `task_type='GPU'`. LGB는 pip 빌드 GPU 미지원이라 CPU 유지
- [x] EXP-003: GPU + CAT iter 확대 — Blend OOF 0.95030
- [x] EXP-003 LB 점수 기록 — **0.94964** (gap 0.00066, 신기록)
- [x] EXP-004: lr=0.02 미세 튜닝 — Blend OOF **0.95042** / OOF opt 0.95048
- [x] EXP-004 LB — equal **0.94967**, opt **0.94977** (LB 신기록)
- [ ] EXP-005 후보: FE (Driver 빈도/타깃 인코딩) — 새 시그널로 다양성 회복 가능성. 마무리에 multi-seed
