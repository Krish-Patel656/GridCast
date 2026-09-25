# F1 finishing-order model

Ranks the field of a Grand Prix from data that exists before the race starts:
car strength, championship position, recent form, circuit history, reliability,
and the qualifying grid once it has been published. Built with PyTorch, FastF1
and the Ergast archive.

```
1. collect_data  →  data/processed/training_data.csv
2. train         →  models/race_predictor.pt + encoders
3. backtest      →  models/backtest.json   (optional but recommended)
4. predict       →  ranked order in the web UI, or from Python
```

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1 — Build the dataset (about a minute)

Race and sprint results come from Ergast, one paged request per season, so a
full rebuild is quick. Standings are accumulated from those results rather than
requested per round, which keeps the run inside the API's rate limit and
guarantees the championship table matches the races the feature history has
already seen.

```bash
python -m ml.collect_data
python -m ml.collect_data --seasons 2022 2023 2024 2025 2026
```

Output:

- `data/processed/training_data.csv` — one row per driver per race
- `data/processed/dataset_meta.json` — row, race, driver and circuit counts

### Features

Everything below is knowable before lights out. Nothing from the race being
predicted is ever read.

| Feature | What it carries |
|---|---|
| `grid_position` | Real grid slot after qualifying, otherwise the pace prior |
| `has_grid` | Whether qualifying had actually run |
| `pace_prior` | Expected starting slot implied by the constructor's rank |
| `grid_percentile`, `grid_vs_teammate` | Starting slot relative to the field and to the team-mate |
| `driver_points_share`, `driver_position_before` | Driver championship strength |
| `constructor_points_share`, `constructor_position_before` | Car strength |
| `avg_finish_last_3`, `avg_finish_last_5` | Driver form |
| `team_avg_finish_last_3` | Team form |
| `circuit_avg_finish`, `team_circuit_avg_finish` | History at this circuit |
| `dnf_rate_last_5`, `team_dnf_rate_last_5` | Driver and car reliability |
| `season_progress` | How far into the season the round falls |
| `driver_code`, `team_id`, `circuit_id` | Categorical embeddings |

Target: `finish_position`.

## Step 2 — Train

```bash
python -m ml.train
python -m ml.train --epochs 150 --lr 5e-4
```

Two details matter:

- **The model predicts a correction, not a result.** Starting position already
  explains most of a race, so the network outputs a shift away from the grid
  slot (or the pace prior before qualifying). The output layer starts at zero,
  so an untrained model reproduces the grid order and training can only improve
  on it.
- **Half the training races are duplicated with the grid hidden.** Without that,
  the model would only ever have seen races where qualifying had happened, and
  predictions for an upcoming Grand Prix would be poorly calibrated.

Batches are whole races, and the loss combines closeness to the real position
with a pairwise penalty for every pair of drivers ordered the wrong way round.

Output:

- `models/race_predictor.pt` — best checkpoint
- `data/processed/encoders.json`, `data/processed/feature_scaler.joblib`
- `models/metadata.json` — metrics, baselines and per-epoch history

## Step 3 — Backtest

A single hold-out split of 40 races is small enough that one lucky weekend moves
the numbers by several points. The backtest retrains from scratch across several
consecutive windows, always training only on earlier races.

```bash
python -m ml.backtest --windows 4 --window-size 20
python -m ml.train        # folds the summary into models/metadata.json
```

Latest run, averaged over 80 unseen races (order accuracy is the share of driver
pairs placed in the correct relative order):

| | Order accuracy | Spearman | Podium names | Winner |
|---|---|---|---|---|
| Model, after qualifying | 78.0% | 0.692 | 67.5% | 65.0% |
| Grid order (baseline) | 76.5% | 0.661 | 67.1% | 58.8% |
| Model, before qualifying | 73.5% | 0.608 | 51.7% | 38.8% |
| Car-strength order (baseline) | 72.2% | 0.583 | 47.9% | 21.2% |

The model beats both baselines in both regimes. Exact position accuracy is
deliberately not reported: safety cars, strategy and retirements make the exact
slot mostly luck, while the order is the part that can be learned.

## Step 4 — Predict

Web UI:

```bash
uvicorn main:app --reload
# then open /predictions
```

Python:

```python
from ml.predict import RacePredictor

result = RacePredictor().predict_race(2026, 14)
print(result["source_label"], result["grid_known"])
for p in result["predictions"][:3]:
    print(p["predicted_position"], p["driver_code"], p["expected_position"])
```

The entry list is resolved from the best pre-race source available: published
qualifying (which also gives the real grid), otherwise the field from the most
recent race, otherwise the championship entry list, otherwise last season's
final field for a season opener. Ergast sometimes omits a driver who set no
qualifying time, so missing cars are added back from that fallback field, two
per constructor at most; they are ranked on car strength and shown without a
grid slot.

## Notes

- A prediction is a ranking, not a forecast of the classification. Read the
  score as an expected finishing position.
- The first `collect_data` run needs network access; results are cached afterwards.
- CUDA is used automatically when available, though the model trains in seconds on a CPU.
