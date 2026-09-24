# F1 Analytics Hub

A FastAPI site for Formula 1 race data with a model that ranks the field before
a Grand Prix starts. Results, championship tables, session calendars and lap
telemetry come from FastF1 and the Ergast archive; the prediction page runs a
PyTorch ranking model over pre-race features only.

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. The first request for a season downloads timing data
into `cache/`, so give it a moment; everything after that is served from disk.

## Pages

| Page | What it shows |
|---|---|
| `/` | Countdown to the next race and a jump-to-a-race form |
| `/results` | Full classification for a race: grid, places gained, points |
| `/standings/drivers`, `/standings/constructors` | Championship tables with points share |
| `/schedule` | Every round of a season with its session times |
| `/telemetry` | Fastest lap and sectors, a lap-by-lap trace, the speed trace when it was archived, and tyre stints |
| `/predictions` | Model-ranked finishing order, plus its measured accuracy |

JSON is available at `/get_races`, `/get_race_drivers` and `/api/predictions`.

## The model

Full detail, including how to rebuild the dataset and retrain, is in
[`ml/README.md`](ml/README.md). The short version:

- It ranks the field rather than predicting exact positions, because the order
  is the learnable part and the exact slot is mostly luck.
- It only ever reads what exists before lights out: car strength, championship
  position, recent form, circuit history, reliability, and the qualifying grid
  once it has been published.
- It predicts a *correction* to the starting order, so it begins from the grid
  and improves on it rather than learning the whole race from nothing.
- Measured over 80 unseen races in a walk-forward backtest, it places 78.0% of
  driver pairs in the correct relative order after qualifying (grid order alone
  gets 76.5%) and 73.5% before qualifying has run (car-strength order gets
  72.2%). Those numbers are shown on the predictions page.

Retraining, if the shipped checkpoint is stale:

```bash
python -m ml.collect_data     # rebuild data/processed/training_data.csv
python -m ml.train            # write models/race_predictor.pt
python -m ml.backtest         # optional: refresh the accuracy figures
```

## Layout

```
main.py             routes and page rendering
f1_data.py          FastF1 / Ergast access, team colours and logos
ml/                 dataset, features, model, training, backtest, prediction
templates/          Jinja2 templates
static/             styles and the motion/parallax script
data/, models/      generated dataset, encoders and checkpoints
cache/              FastF1's own download cache
```

Predictions are a model's opinion, not betting advice.
