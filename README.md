# Playground Series S6E5 - Predicting F1 Pit Stops

이진 분류 (PitNextLap 0/1) | ROC AUC | 2026-05-01 ~ 2026-05-31

대회 링크: https://www.kaggle.com/competitions/playground-series-s6e5

```
data/               대회 데이터
data/notebooks/     참고용 상위 노트북
docs/               대회 개요, 데이터 설명, 실험 기록
notebooks/          실험 노트북
submissions/        제출 파일
```

## Setup

```bash
conda activate s6e5
pip install -r requirements.txt
```

## Experiments

| # | 설명 | OOF AUC | LB |
|---|------|---------|-----|
| 001 | XGBoost Baseline (5F, lr=0.05, 1000est, 14 features) | 0.94906 | 0.94874 |
| 002 | XGB + LGB + CAT, equal blend (5F 동일 split, ES) | 0.95010 | 0.94942 |
| 003 | EXP-002 + GPU + CAT iter 확대 (best_iter ~3485) | 0.95030 | 0.94964 |
| 004 equal | EXP-003 + lr=0.02 + n_est 확대 (CAT best_iter ~9187) | 0.95042 | 0.94967 |
| **004 opt** | EXP-004 blend opt (0.25, 0.25, 0.50) | **0.95048** | **0.94977** |
| 005 | + Driver freq + fold-wise TE (실패) | 0.95025 / 0.95030 | 0.94964 |
| **006** | EXP-004 + multi-seed × 3 (XGB/LGB/CAT 모두) | **0.95053** | **0.94978** |
| **007** | + MLP base (4-way blend opt 0.25/0.25/0.45/0.05) | **0.95053** | **0.94979** |
| 008 | MLP 강화 (epochs 80, opt 0.25/0.25/0.40/0.10) | 0.95053 | 0.94973 ↓ |
| 009 | PLR-MLP only (RealMLP의 PLR 직접 구현) | MLP solo **0.94486** (+0.00391 vs 008) | — |
| **010** | 4-way blend w/ PLR-MLP (opt 0.25/0.20/0.40/**0.15**) | **0.95061** | **0.94987** |
| 011 | PLR-MLP + bin-cat (실패) | MLP solo 0.94444 (-0.00042 vs 009) | — |
| 012 | PLR-MLP + inner ens=3 (MLP only) | MLP solo **0.94763** (+0.00277 vs 009) | — |
