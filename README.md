# Demand Forecasting Engine (Time-Series ML)

Predicts **daily demand per outlet** for the next 7 days by combining sales history,
advance reservations, outlet attributes and the holiday calendar. This is the same
planning input a business uses for staffing, inventory and revenue decisions.

Three gradient-boosted / tree models (LightGBM, XGBoost, Random Forest) are validated
with **walk-forward (time-based) validation** and blended into an ensemble that is
more accurate than any single model.

## Results (sample data, 12 outlets, 2 years daily)

Hold-out = the 3 most recent 7-day validation windows. Ensemble weights were learned
on the earlier windows only, so the ensemble score is not tuned on its own test data.

| Model | RMSLE ↓ | MAE (visitors) ↓ | MAPE % ↓ |
|---|---|---|---|
| **Ensemble (weighted)** | **0.1059** | **4.31** | **8.64** |
| XGBoost | 0.1075 | 4.35 | 8.73 |
| LightGBM | 0.1080 | 4.31 | 8.67 |
| Random Forest | 0.1085 | 4.49 | 8.93 |

The most important signals (averaged rank across models): same-weekday average of
the last 4 weeks, last week's demand, reservations relative to the outlet's normal
booking level, and long-run rolling averages.

## How it works

```
sales.csv ─┐
reservations.csv ─┤   build_panel()     add_features()        walk-forward CV        weighted
locations.csv ─┤──► one row per  ──► calendar, lags, ──► LightGBM / XGBoost / ──► ensemble ──► forecast.csv
holidays.csv ─┘    outlet × day      rolling, bookings    Random Forest          (1/RMSE²)
```

**Data model** – `forecast/data.py` generates realistic sample data: weekly and yearly
seasonality, trend, holiday and pre-holiday uplift, auto-correlated noise, outlets
that are occasionally closed, and booking rates that depend on store type.

**Features** – `forecast/features.py`
- Calendar: day of week, weekend, month, cyclical day-of-year
- Holidays: holiday, day before, day after
- Reservations: bookings for the target day, bookings vs the outlet's 28-day average, bookings / capacity
- History: lags 7/14/21/28, same-weekday mean, rolling means (7–56 days), rolling std, EWM
- Outlet: ID, area, store type, capacity (categorical)

**No leakage** – the model forecasts up to 7 days ahead in one shot (direct strategy),
so every lag/rolling feature is shifted by at least 7 days. Only information known on
the forecast date is used. Reservations and holidays are legitimately known in advance.

**Validation** – `walk_forward_folds()` creates 6 folds spaced 28 days apart. Each
trains on all data *before* a 7-day window and predicts that window, exactly like
production use. Random K-fold would leak future information and overstate accuracy.

**Target** – models are trained on `log1p(visitors)`, so RMSE becomes RMSLE: a 10%
miss at a small outlet counts the same as a 10% miss at a large one.

**Ensemble** – weights ∝ 1 / RMSE² of each model's out-of-fold error. The models make
partly different errors (boosting vs bagging), so averaging cancels some of them out.

## Run it

```bash
pip install -r requirements.txt
python -m forecast.data --out data        # generate sample data (or drop in your own CSVs)
python -m forecast.pipeline               # validate, ensemble, forecast
pytest -q                                 # 6 tests incl. a leakage check
```

Outputs in `reports/`:

| File | Contents |
|---|---|
| `forecast.csv` | next 7 days per outlet: each model's prediction + ensemble forecast |
| `holdout_comparison.csv` | ensemble vs individual models |
| `cv_metrics.csv` | RMSLE / MAE / MAPE for every fold and model |
| `feature_importance.csv` | importance rank per model |
| `validation_plot.html` | interactive actual-vs-forecast chart |
| `summary.json` | weights, scores, forecast window |

## Using real data

The input format matches the public *Recruit Restaurant Visitor Forecasting* dataset
(Kaggle): visits, reservations, store info and a holiday calendar. Map its columns to
`sales.csv (date, location_id, visitors)`, `reservations.csv (date, location_id, reserve_visitors)`,
`locations.csv (location_id, area, store_type, capacity)`, `holidays.csv (date, holiday_name)`.

## Project structure

```
forecast/
  data.py        sample data generator
  features.py    panel construction + feature engineering
  models.py      model wrappers, ensemble weights
  pipeline.py    validation, training, forecasting (CLI)
tests/           pytest suite
```

## Tech
Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · Plotly
