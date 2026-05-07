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
| 001 | XGBoost Baseline (5F, lr=0.05, 1000est, 14 features) | 0.94906 | — |
