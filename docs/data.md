## Dataset Description

The dataset for this competition (both train and test) was inspired by an F1 strategy dataset. Feature distributions are close to, but not exactly the same, as the original. The competition organizers **intentionally removed `Normalized_TyreLife`** to prevent the prediction from being trivial. Feel free to use the original dataset as part of this competition.

### Files
- **train.csv** — 학습 데이터 (439,140 rows × 16 cols, target `PitNextLap` 포함)
- **test.csv** — 테스트 데이터 (188,165 rows × 15 cols)
- **sample_submission.csv** — 제출 포맷

License: CC BY 4.0

### Target
| 컬럼 | 의미 | 분포 |
|------|------|------|
| `PitNextLap` | 다음 랩에 핏스톱 여부 (0/1) | 양성 비율 **19.9%** (87,381 / 439,140) |

### Columns (15 features)

| 컬럼 | 타입 | 의미 |
|------|------|------|
| `id` | int | 샘플 ID (train 0–439,139, test 439,140–627,304) |
| `Driver` | str | 드라이버 코드 (예: `D109`, `D086`, `ZON`) |
| `Compound` | str | 타이어 컴파운드 (HARD 등) |
| `Race` | str | 레이스명 (예: `Canadian Grand Prix`) |
| `Year` | int | 연도 |
| `PitStop` | int | 핏스톱 여부 (현재 랩 기준) |
| `LapNumber` | int | 랩 번호 |
| `Stint` | int | 스틴트 번호 |
| `TyreLife` | float | 타이어 수명 (랩 단위) — `Normalized_TyreLife`는 의도적 제거됨 |
| `Position` | int | 현재 순위 |
| `LapTime (s)` | float | 랩 타임 (초) |
| `LapTime_Delta` | float | 랩 타임 델타 |
| `Cumulative_Degradation` | float | 누적 타이어 열화 |
| `RaceProgress` | float | 레이스 진행률 (0~1) |
| `Position_Change` | float | 순위 변동 |

### 데이터 특성
- **결측 없음**
- ID는 train/test 연속 (시계열적 분리 가능성 — 검증 시 주의)
- Position_Change 가 float인데 의미상 정수 — 유의

---

## 원본 데이터 (참고)
대회 측이 "F1 strategy dataset"이라고만 언급, 직접 링크 미제공. 원본을 찾으면 추가.
