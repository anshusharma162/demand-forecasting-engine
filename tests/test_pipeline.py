import json

import numpy as np
import pandas as pd
import pytest

from forecast.data import generate
from forecast.features import add_features, build_panel
from forecast.models import ensemble_weights
from forecast.pipeline import run, walk_forward_folds


@pytest.fixture(scope="module")
def tables():
    return generate(n_locations=3, start="2024-01-01", end="2024-12-31", seed=1)


def test_generated_tables_have_expected_columns(tables):
    assert set(tables) == {"locations", "holidays", "reservations", "sales"}
    assert {"date", "location_id", "visitors"} <= set(tables["sales"].columns)
    assert (tables["sales"].visitors >= 0).all()


def test_panel_extends_into_forecast_horizon(tables):
    panel = build_panel(tables, horizon=7)
    last_sale = pd.to_datetime(tables["sales"].date).max()
    assert panel.date.max() == last_sale + pd.Timedelta(days=7)
    assert panel.loc[~panel.is_history, "visitors"].isna().all()


def test_lag_features_do_not_leak_target(tables):
    """lag_7 on day t must equal the actual log demand on day t-7."""
    df = add_features(build_panel(tables, horizon=7), horizon=7)
    one = df[df.location_id == df.location_id.iloc[0]].set_index("date")
    t = one.index[100]
    expected = one.loc[t - pd.Timedelta(days=7), "log_visitors"]
    got = one.loc[t, "lag_7"]
    assert (np.isnan(expected) and np.isnan(got)) or got == pytest.approx(expected)
    # No feature for a future (unknown) day is built from that day's own demand
    assert "visitors" not in [c for c in df.columns if c.startswith(("lag_", "roll_"))]


def test_walk_forward_folds_never_overlap():
    dates = pd.Series(pd.date_range("2024-01-01", "2024-12-31"))
    folds = list(walk_forward_folds(dates, n_folds=4, horizon=7, spacing=28))
    assert len(folds) == 4
    for tr_end, va_start, va_end in folds:
        assert tr_end < va_start <= va_end
        assert (va_end - va_start).days == 6


def test_ensemble_weights_favor_better_model():
    w = ensemble_weights({"a": 0.1, "b": 0.2})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["a"] > w["b"]


def test_pipeline_end_to_end(tables, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name, df in tables.items():
        df.to_csv(data / f"{name}.csv", index=False)
    summary = run(data, tmp_path / "reports", n_folds=2, verbose=False)
    fc = pd.read_csv(tmp_path / "reports" / "forecast.csv")
    assert len(fc) == 3 * 7
    assert (fc.forecast >= 0).all()
    assert json.loads((tmp_path / "reports" / "summary.json").read_text()) == summary
