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
| EXP-005 + Driver FE blend equal | 0.95025 | 0.94964 | 0.00061 | **실패**: 모든 모델 -0.00005~-0.00019, blend -0.00017 |
| EXP-005 + Driver FE blend opt (0.30/0.25/0.45) | 0.95030 | — | — | EXP-004 opt 0.95048 대비 -0.00018 |
| EXP-006 multi-seed×3 blend equal | 0.95048 | 0.94968 | 0.00080 | OOF Δ +0.00006 → LB Δ +0.00001 (17%) |
| **EXP-006 multi-seed×3 blend opt** | **0.95053** | **0.94978** | 0.00075 | LB 신기록(+0.00001) — noise 수준. multi-seed 한계 효용 진입 |
| EXP-007 +MLP blend equal (1/4) | 0.95020 | — | — | MLP 25%는 너무 큼 → -0.00022 |
| **EXP-007 +MLP blend opt (0.25/0.25/0.45/**0.05**)** | **0.95053** | **0.94979** | 0.00074 | LB 신기록 (vs EXP-006 +0.00001). OOF 동점이지만 LB 우위 — MLP 다양성이 진짜 시그널 추가 |
| EXP-008 MLP solo | 0.94095 | — | — | epochs 80 → +0.00051 vs EXP-007 (학습 미수렴 가설 입증) |
| EXP-008 +강화MLP blend opt (0.25/0.25/0.40/**0.10**) | 0.95053 | 0.94973 | 0.00080 | LB **-0.00006** vs EXP-007. 가중치 변화가 OOF에는 중립, LB엔 negative — OOF-친화 |
| **EXP-009 PLR-MLP solo** | **0.94486** | — | — | RealMLP의 PLR 직접 구현. MLP solo **+0.00391** vs EXP-008. 학습도 절반 시간에 일찍 수렴 |
| EXP-010 4-way blend equal | 0.95045 | — | — | -0.00008 vs 천장 — equal 가중치는 MLP 흡수 부담 |
| **EXP-010 4-way blend opt (0.25/0.20/0.40/**0.15**)** | **0.95061** | **0.94987** | 0.00074 | **LB 신기록 (+0.00008 vs EXP-007)**. OOF Δ→LB Δ 100% 전이 — PLR이 진짜 새 시그널 |
| EXP-011 PLR+BinCat MLP solo | 0.94444 | — | — | bin-cat 부정 (-0.00042 vs EXP-009 PLR-MLP). PLR과 redundant + broad 카디널리티 embedding sparse |
| **EXP-012 PLR-MLP + inner ens=3 solo** | **0.94763** | — | — | **+0.00277 vs EXP-009 PLR-MLP**. 모든 fold에서 +0.0025 일관 향상. multi-seed의 ~50배 효과 |
| EXP-013 4-way blend equal | 0.95057 | — | — | +0.00012 vs EXP-010 equal. MLP가 더 균형 기여 |
| **EXP-013 4-way blend opt (0.20/0.20/0.40/**0.20**)** | **0.95063** | — | — | +0.00002 vs EXP-010 opt. **OOF 천장 미세 돌파**. MLP solo +0.00277은 blend에 거의 흡수 |
| EXP-007 단일: MLP | 0.94044 | — | — | 트리 대비 -0.009. 50 epoch 다 사용 — 추가 학습 여지 |
| EXP-006 단일: XGB | 0.94969 | — | — | seed range 0.00006 |
| EXP-006 단일: LGB | 0.94959 | — | — | seed range 0.00002 |
| EXP-006 단일: CAT | 0.94978 | — | — | seed range 0.00010 |
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

## EXP-013: 4-way blend with PLR-MLP + Inner Ensemble (n_ens=3)

- **노트북**: `exp013_plr_inner_4way.ipynb`
- **구성**: XGB/LGB/CAT (EXP-010 동일) + PLR-MLP w/ inner ens=3 (EXP-012 패턴)
- **결과**:
  - PLR-MLP+ens OOF **0.94763** (EXP-012 재현, +0.00277 vs PLR-MLP solo)
  - Blend equal 4-way: 0.95057 (+0.00012 vs EXP-010)
  - **Blend opt 4-way `(0.20, 0.20, 0.40, 0.20)`**: **0.95063** (+0.00002 vs EXP-010 0.95061)
  - PLR-MLP 한계 효과 +0.00014 → **+0.00018** (소폭 증가)
  - **w_mlp 0.15 → 0.20** (PLR-MLP+ens 단독 강해진 만큼 비중 ↑)
- **소요시간**: ~67분 (CAT 33분 + PLR-MLP+ens 29분)
- **교훈 — 한계 효용 진입**:
  - MLP solo +0.00277은 blend opt에 거의 흡수 (+0.00002만 LB 가능 영역)
  - 트리들이 정교해서 MLP 향상 흡수, blend 천장 사실상 0.95063 근처
  - Equal blend +0.00012로 의미 있음 — MLP가 더 균형 기여
  - LB 신기록 가능성은 미세 (+0~+0.00002)

---

## EXP-012: PLR-MLP + Inner Ensemble (n_ens=3, MLP only)

- **노트북**: `exp012_plr_inner_ens.ipynb`
- **구현**: 각 fold 안에서 3개 모델 학습 (seed = SEED + ens_i * 1000), val/test prediction 평균
- **결과**: MLP solo OOF **0.94763 (+0.00277 vs EXP-009 0.94486)** — Inner ens 효과 매우 큼
- **Fold별 일관 향상**: 모든 fold에서 solo → ens-avg +0.0024~0.0029 (매우 일관)
- **소요시간**: 31분 (5 fold × 3 ens × ~80 epoch max)
- **Inner ens vs Multi-seed 비교**:
  - EXP-006 multi-seed×3: blend OOF +0.00006, LB +0.00001 — 한계 효용
  - EXP-012 inner ens×3: MLP solo **+0.00277** — multi-seed의 ~50배
  - 이유: PLR-MLP의 학습 stochasticity (옵티마이저 noise, batch order, PLR random freqs init)가 크기 때문. fold-local 평균이 가장 효과적
- **시사점**: 4-way blend (EXP-013)에서 PLR-MLP 단독 강화가 큰 효과 가능. w_mlp 비중 추가 증가 + LB 도약 기대

---

## EXP-011: PLR-MLP + Bin-cat (실패)

- **노트북**: `exp011_plr_bincat_mlp.ipynb`
- **목적**: RealMLP의 `feature_engineering` (numeric → np.floor → factorize → embedding) 자체 구현 검증
- **구현**:
  - 11개 NUM_COL → np.floor → train으로 factorize (라벨 무관, 누수 없음)
  - test에 mapping 적용, unseen → unknown slot 0
  - bin-cat 카디널리티: Year 5, PitStop 3, LapNumber 79, Stint 9, TyreLife 78, Position 21, **LapTime 133**, **LapTime_Delta 219**, **Cumulative_Degradation 446**, RaceProgress 3, Position_Change 38
  - 각 bin-cat embedding dim=4 → 44차 추가
  - 입력 dim 127 → **171**
- **결과**: MLP solo OOF **0.94444 (-0.00042 vs EXP-009 0.94486)** — 부정 효과
- **원인 추정**:
  1. **PLR과 redundant** — periodic encoding이 이미 fine-grained 수치 표현 잡음
  2. **Broad cardinality embedding sparse** — Cumulative_Degradation(446), LapTime_Delta(219) 등 embedding 학습 부담
  3. PLR-MLP가 우리 dataset의 numeric 표현을 이미 충분히 잘 함
- **교훈**: **RealMLP의 모든 기법이 우리에게 효과 있는 건 아님**. 한 번씩 검증 필수. PLR은 가치 있었지만 bin-cat dual은 redundant.

---

## EXP-010: 4-way blend with PLR-MLP

- **노트북**: `exp010_plr_4way.ipynb`
- **구성**: XGB/LGB/CAT (EXP-008 동일) + PLR-MLP (EXP-009 동일)
- **결과**:
  - XGB 0.94946 / LGB 0.94933 / CAT 0.94958 / PLR-MLP 0.94486 (재현 완벽)
  - Blend equal 4-way: **0.95045** (-0.00008 vs 천장 — equal에 MLP 25% 부담)
  - **Blend opt 4-way `(0.25, 0.20, 0.40, 0.15)`: 0.95061** (**자체 학습 천장 돌파 +0.00008**)
  - Blend opt 3-way (no MLP): 0.95047 → **PLR-MLP 한계 효과 +0.00014** (EXP-008의 +0.00011보다 강함)
- **가중치 변화 추세**:
  - EXP-007 (일반 MLP): `(0.25, 0.25, 0.45, 0.05)` — w_mlp 5%
  - EXP-008 (강화 MLP): `(0.25, 0.25, 0.40, 0.10)` — w_mlp 10%
  - **EXP-010 (PLR-MLP)**: **`(0.25, 0.20, 0.40, 0.15)` — w_mlp 15%** (3배 증가). LGB가 일부 줄고 PLR-MLP에 자리
- **소요시간**: 총 ~50분 (XGB 70s, LGB 116s, CAT 36분, PLR-MLP 10분)
- **교훈**:
  - PLR-MLP가 진짜 다양성 + 표현력 추가 — 자체 학습 OOF 천장 0.95053 → 0.95061로 돌파
  - 가중치도 단계적으로 ↑ (5%→10%→15%) — MLP 단독 강도와 비례
  - LB 측정 필요 — OOF +0.00008이 진짜 LB로 전이되는지

---

## EXP-009: PLR-MLP only (Periodic Linear Representation 직접 구현)

- **노트북**: `exp009_plr_mlp.ipynb`
- **목적**: RealMLP의 핵심 기법 PLR이 우리 baseline에 단독 효과 있는지 빠르게 검증 (MLP만 학습, 트리 생략)
- **PLR 구현 (직접)**:
  - 학습 가능 frequencies `Parameter(n_numeric, k=16) * σ=2.33`
  - `sin/cos(2π·c·x)` periodic features
  - shared Linear `2k → hidden=8` + GELU
  - 11 numeric → 88 PLR features (input 39 → 127로 확장)
- **결과**:
  - **PLR-MLP solo OOF: 0.94486** (vs EXP-008 MLP 0.94095, **+0.00391**)
  - 모든 fold가 일찍 early stop (best_epoch 14-19, used 34-39 / max=80, patience=20)
  - 학습 시간 649s (EXP-008 MLP 1238s의 절반)
  - test pred mean 0.2010 (실제 양성률 0.199와 일치)
- **교훈**: PLR이 우리 baseline에 강한 단독 효과. 학습 가속화도 동시. 4-way blend에서 비중 크게 차지할 가능성 → EXP-010 풀 학습 진행 가치 있음

---

## EXP-008: MLP 강화 (epochs 50→80, patience 15→20)

- **노트북**: `exp008_mlp_strong.ipynb`
- **변경점 vs EXP-007**: MLP만 — MAX_EPOCHS 50→80, patience 15→20, T_max 50→80. 그 외 동일.
- **결과**:
  - **MLP solo OOF 0.94044 → 0.94095 (+0.00051)** ← 학습 미수렴 가설 입증
  - MLP best_epoch: 65, 70, 79, 79, 76 (모든 fold 80에 닿음 — 더 늘려도 효과 작을 듯)
  - Blend equal 4-way 0.95020 → 0.95029 (+0.00009)
  - **Blend opt 4-way 0.95053 = 0.95053 (천장 재확인)**
  - **w_mlp 0.05 → 0.10** (2배), CAT 0.45 → 0.40
  - MLP 한계 효과 +0.00006 → +0.00011 (4-way − 3-way opt)
- **소요시간**: XGB 70s / LGB 113s / CAT 1890s (32분) / MLP 1238s (20.6분, EXP-007 12.5분의 1.6배 ≈ epoch 1.6배)
- **교훈**:
  - **학습 미수렴 가설 옳았음** — solo OOF +0.00051은 진짜 효과
  - 그러나 **blend opt 천장은 0.95053으로 변하지 않음** — 단일 강화가 blend에 흡수
  - **opt 가중치 변화는 진짜 다양성 시그널** (w_mlp 2배) — LB 전이 측정 가치
  - 다음: epoch 더 늘리거나 (120) MLP multi-seed 고려. 단 OOF 동점이라 LB 측정 우선

---

## EXP-007: MLP 추가 (4-way blend)

- **노트북**: `exp007_mlp.ipynb`
- **상세 설계 문서**: `docs/mlp_design.md`
- **변경점 vs EXP-004**: PyTorch MLP를 base 모델에 추가 (cat embedding + numeric BN, 256→128→64, GELU, AdamW+CosineLR, 5-fold same split)
- **결과**:
  - MLP 단독 OOF: **0.94044** (트리 대비 -0.009)
  - Blend equal 4-way: 0.95020 (-0.00022 vs EXP-004 3-way) — MLP 25% 너무 큼
  - **Blend opt 4-way (0.25, 0.25, 0.45, 0.05)**: **0.95053** (= EXP-006 OOF best)
  - Blend opt 3-way (no MLP): 0.95047 → MLP 한계 효과 **+0.00006**
- **소요시간**: XGB 70s / LGB 121s / CAT 2154s / **MLP 752s (12.5분)**
- **교훈**:
  - MLP가 진짜 다른 모델 family로 작동 (opt blend가 0이 아닌 5% 비중 자연 선택)
  - 단독 정확도 약하지만 다른 오류 패턴 → blend 다양성 기여
  - 모든 fold에서 50 epoch 다 사용 → MLP 추가 학습 여지 존재 (max_epochs↑)
  - 한계 효과 +0.00006 — multi-seed +0.00005와 비슷한 수준

---

## EXP-006: Multi-seed (마무리 병기)

- **노트북**: `exp006_multiseed.ipynb`
- **변경점 vs EXP-004**: KFold split seed=42 고정, 모델 random_state ∈ {42, 123, 2026} 3 seed 평균
- **결과 (단일 모델)**:
  - XGB 0.94946 → **0.94969** (+0.00023, seed range 0.00006)
  - LGB 0.94933 → **0.94959** (+0.00026, seed range 0.00002)
  - CAT 0.94961 → **0.94978** (+0.00017, seed range 0.00010)
- **Blending**:
  - Equal 0.95042 → **0.95048** (+0.00006)
  - Opt (0.25, 0.25, 0.50) 0.95048 → **0.95053** (+0.00005)
- **소요시간**: XGB 219s / LGB 370s / CAT 6027s (100분) — 총 ~110분 (예상 135분보다 짧음)
- **교훈 — multi-seed의 양면성**:
  - 단일 모델 분산↓ → OOF +0.0002 일관 개선 (예상대로)
  - 각 모델이 안정화되면서 모델 간 다양성도 줄어 **블렌드 이득은 +0.00005만** (단일 평균의 1/4)
  - LB 전이는 단일 개선이 분산 감소 진짜 효과라 1:1에 가까울 가능성 — 단일 LB가 blend LB와 비슷할 수도

---

## EXP-005: Driver Feature Engineering (실패)

- **노트북**: `exp005_driver_fe.ipynb`
- **추가**:
  - `Driver_freq`: train 전체 등장 횟수 (라벨 무관)
  - `Driver_te`: fold 내 train만으로 smoothed target encoding (`prior_w=10`, `prior=0.199`)
- **결과**: 모든 모델/블렌딩 negative
  - XGB 0.94946 → 0.94941 (-0.00005)
  - LGB 0.94933 → 0.94914 (-0.00019)
  - CAT 0.94961 → 0.94950 (-0.00011)
  - Blend equal 0.95042 → 0.95025 (-0.00017)
  - LB equal 0.94967 → **0.94964 (-0.00003)** — OOF 하락이 LB로 정직하게 전이
- **원인 추정**:
  - CatBoost의 native categorical (ordered TS encoding)이 이미 더 정교 — 단순 smoothed TE는 redundant noise
  - XGB/LGB도 LE만으로 split을 잘 학습 — 추가 numeric은 거친 신호
  - 합성 데이터라 Driver_freq의 등장 횟수에 실제 드라이버 정체성 정보가 적을 가능성
  - 누수는 없음 (TE sanity sample에서 PitNextLap=0/1 모두 0.18~0.30 자연 분산 확인)
- **다양성 회복도 실패**: blend(opt - equal) = 0.00006 → 0.00005 거의 동일. 모델 간 다양성 못 만듦
- **흥미로운 부산물**: CV-LB 격차 0.00075 → 0.00061로 줄어듦 — EXP-004의 OOF에 OOF-친화 과적합이 약간 있었다는 시사

→ **결론**: 본 데이터에 단순 categorical encoding FE는 효과 없음. 다음은 인터랙션 FE / multi-seed 다양성 / 다른 모델 알고리즘 추가로 방향 전환.

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
- **합성 데이터에 단순 cat encoding FE 효과 없음** (EXP-005). 트리 모델은 LE/native cat으로 이미 잘 학습. 추가 freq/smoothed TE는 redundant noise.
- **multi-seed 한계 효용 단계 진입 (EXP-006)**: 단일 OOF +0.0002는 진짜 분산 감소 효과지만 동일 모델 안에서의 안정화 ↑ ↔ 모델 간 다양성 ↓ trade-off로 블렌드 OOF는 +0.00006만, LB는 +0.00001 (17% 전이). 다음 도약은 같은 알고리즘 분산 줄이기가 아니라 **진짜 다른 알고리즘** 추가에서 와야 함.
- **MLP 다양성은 LB로 미세하게 전이됨 (EXP-007)**: OOF는 EXP-006과 동점(0.95053)이지만 LB +0.00001 (0.94978 → 0.94979). multi-seed의 OOF→LB와는 **다른 종류의 기여** (분산 감소 vs 진짜 다른 패턴). 효과 미세하지만 본질은 다름. **OOF 0.95053이 자체 학습 천장**일 가능성 — 더 큰 도약은 외부 모델(RealMLP 등)이 필요.
- **MLP 강화는 OOF 향상이 LB로 전이 안 됨 (EXP-008)**: solo +0.00051이지만 blend opt OOF 동점, LB는 -0.00006. **EXP-007 (LB 0.94979)이 자체 학습 진짜 best**. CV-LB 격차 0.00080으로 확대 — 추가 강화는 OOF-친화 영역. 자체 학습 천장 도달 확정.
- **PLR-MLP가 자체 학습 천장 돌파 (EXP-010)**: RealMLP 핵심 기법인 Periodic Linear Representation을 외부 라이브러리 없이 직접 구현. MLP solo OOF 0.94095 → 0.94486 (+0.00391), blend opt OOF 0.95053 → 0.95061 (+0.00008), **LB 0.94979 → 0.94987 (+0.00008, 100% 전이)**. multi-seed/MLP epoch↑와 달리 진짜 새 시그널 추가. CV-LB 격차 0.00074 유지 (overfit 신호 없음).
- **RealMLP의 모든 기법이 우리에게 효과 있는 건 아님 (EXP-011)**: bin-cat dual representation (np.floor + factorize + embedding)은 PLR-MLP에 추가했을 때 -0.00042 부정 효과. PLR이 이미 fine-grained numeric 표현 충분히 잡고 있어 redundant. 외부 노트북 기법은 한 번씩 따로 검증 필수.
- **Inner ensemble은 multi-seed보다 훨씬 효과적 (EXP-012)**: 같은 fold split에서 다른 starting seed로 학습 후 평균이 OOF에 +0.00277 (multi-seed의 ~50배). PLR-MLP의 학습 stochasticity (옵티마이저 noise + batch order + PLR random freqs init)가 크기 때문. **외부 multi-seed보다 inner ens가 우리 NN에 정합**.
- **MLP 단독 강화는 blend에 흡수됨 (EXP-013)**: MLP solo +0.00277 (EXP-012) → blend opt +0.00002 (EXP-013). 트리들이 이미 정교해서 MLP 향상을 흡수, blend 천장이 거의 닫힘. w_mlp는 0.15→0.20으로 비중 ↑하지만 blend OOF는 +0.00002만. **블렌딩 본질적 한계** — 단일 모델 강화가 blend로 1:1 전이 X.

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
- [x] EXP-005: Driver FE — **실패** (OOF -0.00017, LB -0.00003). 합성 데이터에 단순 cat encoding FE 효과 없음
- [x] EXP-006: Multi-seed × 3 — Blend OOF **0.95053** (단일 +0.0002 안정 개선)
- [x] EXP-006 LB — equal 0.94968 / opt **0.94978** (LB +0.00001 신기록, 한계 효용 진입)
- [x] EXP-007: MLP 추가 — Blend opt 0.95053 (MLP 한계 효과 +0.00006). 상세: `mlp_design.md`
- [x] EXP-007 LB — opt **0.94979** (LB +0.00001 신기록). OOF 동점 vs EXP-006이지만 LB 우위 — MLP 다양성 진짜 시그널
- [x] EXP-008: MLP 강화 (epochs 50→80) — MLP solo +0.00051, blend opt OOF 천장(0.95053), w_mlp 0.05→0.10
- [x] EXP-008 LB — opt 0.94973 (-0.00006 vs EXP-007). **자체 학습 천장 0.94979 도달**(나중에 갱신됨)
- [x] EXP-009: PLR-MLP only — MLP solo **0.94486 (+0.00391)**. RealMLP의 PLR 기법 직접 구현으로 자체 학습 천장 돌파 가능성
- [x] EXP-010: 4-way blend w/ PLR-MLP — **OOF 0.95061 (자체 학습 천장 돌파 +0.00008)**. w_mlp 0.05→0.15 단계 증가
- [x] EXP-010 LB — opt **0.94987** (LB +0.00008 신기록). OOF Δ→LB Δ **100% 전이** — PLR이 진짜 새 시그널
- [x] EXP-011: PLR + bin-cat — **부정 효과** (-0.00042). PLR과 redundant + broad 카디널리티 embedding sparse
- [x] EXP-012: PLR-MLP + inner ens=3 — **MLP solo +0.00277**. multi-seed의 ~50배 효과
- [x] EXP-013: 4-way + PLR-MLP+inner ens — **Blend opt OOF 0.95063** (+0.00002, 한계 효용). w_mlp 0.15→0.20
- [ ] EXP-013 LB 제출 결정 — 신기록 가능성 미세지만 확정 측정 가치
