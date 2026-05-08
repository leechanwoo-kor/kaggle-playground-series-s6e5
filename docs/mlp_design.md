# MLP 모델 설계 + EXP-007 결과 (Playground Series S6E5)

**노트북**: `notebooks/exp007_mlp.ipynb`
**작성일**: 2026-05-08
**역할**: XGB/LGB/CAT 트리 3개에 진짜 다른 알고리즘 family를 추가하여 모델 다양성 회복

---

## 1. 도입 배경

EXP-005 Driver FE / EXP-006 multi-seed 모두 다양성 회복에 실패:

| 시도 | OOF Δ | LB Δ | 결과 |
|---|---|---|---|
| EXP-005 Driver FE | -0.00017 | -0.00003 | 합성 데이터에 단순 cat encoding 효과 없음 |
| EXP-006 multi-seed×3 | +0.00005 (blend) | +0.00001 (blend) | 단일 분산↓이지만 모델 간 다양성도↓ — 한계 효용 |

→ 트리 3개로는 한계. **진짜 알고리즘 family를 바꾸는 것**이 다음 직접 경로.

---

## 2. 모델 위치와 역할

| 항목 | 결정 |
|---|---|
| 위치 | XGB/LGB/CAT와 **동급 base 모델** (4-way blend, stacker가 아님) |
| 입력 | **raw features 직접** (다른 모델의 OOF 기반이 아님) |
| 목적 | 진짜 다른 학습 패러다임 — 트리들과 다른 오류 패턴 → blend 다양성 |

s6e5의 외부 stacking 노트북 (`s6e5-stacking-vibe-coding`)의 MLP는 base OOF를 결합하는 작은 stacker(`(32,16)` 또는 `(64,32)`). 우리 MLP는 raw features에서 직접 학습하는 표현 모델이라 더 큼.

---

## 3. 입력 처리

### Categorical embedding
| 컬럼 | unique | Embedding(rows, dim) |
|---|---|---|
| `Driver` | 887 | `Embedding(888, 16)` |
| `Compound` | 5 | `Embedding(6, 4)` |
| `Race` | 26 | `Embedding(27, 8)` |

- LE 결과 `[0..N-1]` → **+1 offset** → `[1..N]`. test의 unseen `-1`은 `0`으로 매핑 (unknown slot)
- **dim 결정 근거**: `min(50, ceil(sqrt(N)))` 휴리스틱에서 dim 16/4/8 선정. Driver는 887에 비해 16으로 보수적 — 과적합 통제

### Numeric features (11개)
`Year, PitStop, LapNumber, Stint, TyreLife, Position, LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change`

- **Per-fold StandardScaler** (train fold만으로 fit → val/test에 transform). 누수 방지.
- 모델 입력 직전 `BatchNorm1d` 한 번 더 — fold별 distribution shift 보정.

### 입력 결합
```
[emb_Driver(16) | emb_Compound(4) | emb_Race(8) | bn(num_11)]  →  39-dim
```

---

## 4. 모델 아키텍처

```
Input (39)
   │
   ├─ Linear(39 → 256)  +  BatchNorm1d  +  GELU  +  Dropout(0.2)
   │
   ├─ Linear(256 → 128) +  BatchNorm1d  +  GELU  +  Dropout(0.2)
   │
   ├─ Linear(128 → 64)  +  BatchNorm1d  +  GELU  +  Dropout(0.2)
   │
   └─ Linear(64 → 1)  →  raw logit
```

**파라미터 수**: ~50K (cat embedding 14K + Linear 36K)

**설계 결정 근거:**
| 요소 | 선택 | 근거 |
|---|---|---|
| Width 256 → 128 → 64 | 점진 축소 | 14 features에 너무 큰 width는 noise 학습 |
| GELU | vs ReLU | 매끈한 gradient, BCE 분류에 약간 우위 |
| BatchNorm1d | 매 layer 후 | covariate shift 안정화, lr robust |
| Dropout 0.2 | 일관 | tabular 0.1~0.3 범위 안전, 0.2 표준값 |
| 출력 raw logit | sigmoid는 loss에 흡수 | `BCEWithLogitsLoss` numerically stable |

---

## 5. 학습 설정

| 요소 | 값 |
|---|---|
| Loss | `BCEWithLogitsLoss` |
| Optimizer | `AdamW(lr=1e-3, weight_decay=1e-4)` |
| Scheduler | `CosineAnnealingLR(T_max=50)` |
| Batch size | 4096 — 큰 batch (BatchNorm 안정 + GPU throughput) |
| Max epochs | 50 |
| Early stopping | val AUC patience=15 (best state restore) |
| `torch.manual_seed` | 42 (fold 간 동일) |
| `cudnn.benchmark` | True (입력 크기 고정) |

---

## 6. Validation 호환성

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — XGB/LGB/CAT와 **동일 split**. OOF 인덱스 정확히 일치 → 4-way blending 직접 가능.

---

## 7. 결과

### 단일 모델 OOF
| 모델 | OOF AUC |
|---|---|
| XGB | 0.94946 |
| LGB | 0.94933 |
| CAT | 0.94958 |
| **MLP** | **0.94044** |

→ MLP 단독이 트리 대비 **-0.009** 낮음. 합성 tabular에선 트리가 더 강한 표현력. 예상한 결과.

### Fold별 MLP 점수
| Fold | AUC | epochs_used |
|---|---|---|
| 0 | 0.94209 | 50 |
| 1 | 0.93940 | 50 |
| 2 | 0.94023 | 50 |
| 3 | 0.94005 | 50 |
| 4 | 0.94047 | 50 |

**관찰**: 모든 fold에서 50 epoch 다 사용 (patience=15에 안 걸림) — **MLP 추가 학습 여지 존재**. CosineAnnealingLR이 `T_max=50`이라 후반 lr 작아져 있음. MAX_EPOCHS↑ 또는 lr 조정 시 추가 향상 가능.

### Blending 결과
| Blend | OOF AUC | vs EXP-004 |
|---|---|---|
| Equal 4-way (1/4 each) | 0.95020 | -0.00022 ❌ |
| **Opt 4-way `(0.25, 0.25, 0.45, 0.05)`** | **0.95053** | +0.00005 |
| Opt 3-way (no MLP) | 0.95047 | -0.00001 |
| **MLP 한계 효과** (4-way opt − 3-way opt) | **+0.00006** | — |

**핵심:**
- **Equal blend 악화** — MLP에 25% 가중치는 너무 큼 (단독 0.94044가 평균에 끌림)
- **Opt에서는 `w_mlp=0.05` (5%)** 자연 선택 → MLP가 진짜 다른 정보 갖고 있다는 시그널
- **+0.00006 한계 효과** — multi-seed의 +0.00005와 비슷한 수준. 다양성 기여 있지만 한계

### 소요 시간
| 모델 | 시간 |
|---|---|
| XGB GPU | 70초 |
| LGB CPU | 121초 |
| CAT GPU iter=20000 | 2154초 (36분) |
| **MLP GPU 5-fold × 50 epoch** | **752초 (12.5분)** |

MLP 학습은 빠른 편. 추가 epoch이나 architecture 키워도 비용 부담 적음.

---

## 8. 인사이트와 한계

### 시그널
- **MLP는 진짜 다른 모델 family로 작동** — opt blend가 0이 아닌 비중 (0.05) 부여한 것이 증거
- **OOF 한계 효과 +0.00006** — 작지만 진짜. 다른 오류 패턴 학습

### 한계
- **단독 정확도 트리 대비 매우 약함** (-0.009) — tabular 합성 데이터의 본질적 패턴
- **Equal blend는 사용 불가** — opt blend로만 의미 있음
- **50 epoch 다 사용** — early stop 안 걸려 추가 학습 여지 존재

### LB 결과 (확정)
- **`submission_exp007_blend_opt.csv` LB: 0.94979** (vs EXP-006 0.94978, **+0.00001**)
- OOF는 EXP-006과 동점(0.95053)인데 LB 우위 → MLP가 진짜 새 시그널 추가 (multi-seed의 분산 감소와는 다른 종류 기여)
- CV-LB 격차 0.00074 — EXP-006 0.00075와 동일 수준
- 효과 미세하지만 **OOF 동점 → LB +**라는 점에서 MLP 가치 입증. 다음 단계로 MLP 강화하면 추가 가능성

---

## 9. 향후 개선 후보

| 옵션 | 예상 효과 | 비용 |
|---|---|---|
| MAX_EPOCHS 80~100 + patience 20 | MLP 단독 +0.0005~0.001 | +5~10분 |
| Width 512→256→128 | 표현력 ↑ | +5분 |
| Mixup / label smoothing | 일반화 | 구현 비용 |
| MLP multi-seed × 3 | 분산 ↓ | +25분 |
| `lr=5e-4` + 더 많은 epoch | 미세 학습 | +10분 |

→ 단독 OOF가 0.945 정도까지 올라오면 blend opt에서 비중도 올라가 LB 효과 ↑.

---

## 10. 참고 노트북과 비교

| 항목 | `s6e5-stacking-vibe-coding` MLP | EXP-007 MLP |
|---|---|---|
| 역할 | L1 stacker (base OOF 위) | base model (raw feature) |
| 입력 | base OOF + logit + 통계 (~14차) | cat emb 28 + numeric 11 = 39차 |
| 본체 | `(32,16)` 또는 `(64,32)` 작음 | `(256, 128, 64)` 더 큼 |
| Activation | SiLU | GELU |
| Internal CV | 학습용 inner K-fold (early stop) | 동일 fold val로 patience |
| 의미 | base 결합기 | 진짜 다른 모델 family |

→ 두 MLP는 **목적과 입력이 근본적으로 다름**. stacking MLP는 base 출력의 비선형 결합기. 우리 MLP는 표현력 학습 모델.
