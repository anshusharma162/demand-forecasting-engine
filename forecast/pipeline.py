"""End-to-end forecasting pipeline.

    python -m forecast.pipeline --data data --out reports

1. Load sales, reservations, locations and holidays
2. Build the daily panel and features
3. Time-based (walk-forward) validation of LightGBM, XGBoost and Random Forest
4. Combine them with an inverse-error weighted ensemble
5. Retrain on all history and forecast the next ``horizon`` days
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_features, build_panel, feature_columns
from .models import ALL_MODELS, blend, ensemble_weights, rmse

MIN_HISTORY_DAYS = 56  # need this much history before rolling features are reliable


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    return {n: pd.read_csv(d / f"{n}.csv") for n in ("sales", "reservations", "locations", "holidays")}


def walk_forward_folds(dates: pd.Series, n_folds: int, horizon: int, spacing: int):
    """Yield (train_end, valid_start, valid_end). Training always ends before validation starts."""
    last = dates.max()
    for k in range(n_folds - 1, -1, -1):
        valid_end = last - pd.Timedelta(days=k * spacing)
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        train_end = valid_start - pd.Timedelta(days=1)
        yield train_end, valid_start, valid_end


def metrics(y_true_log, y_pred_log) -> dict[str, float]:
    yt, yp = np.expm1(y_true_log), np.clip(np.expm1(y_pred_log), 0, None)
    mask = yt > 0
    return {
        "rmsle": rmse(y_true_log, y_pred_log),
        "mae": float(np.mean(np.abs(yt - yp))),
        "mape": float(np.mean(np.abs(yt[mask] - yp[mask]) / yt[mask]) * 100),
    }


def run(data_dir="data", out_dir="reports", horizon=7, n_folds=6, spacing=28, seed=0, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = load_tables(data_dir)
    panel = build_panel(tables, horizon)
    df = add_features(panel, horizon)
    feats = feature_columns(df)

    start = df.date.min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    hist = df[df.is_history & df.visitors.notna() & (df.date >= start)]
    log(f"Training rows: {len(hist):,}  |  features: {len(feats)}")

    # ---------- walk-forward validation ----------
    fold_rows, oof = [], []
    for i, (tr_end, va_start, va_end) in enumerate(walk_forward_folds(hist.date, n_folds, horizon, spacing)):
        tr = hist[hist.date <= tr_end]
        va = hist[(hist.date >= va_start) & (hist.date <= va_end)]
        preds = {}
        for M in ALL_MODELS:
            m = M(seed).fit(tr[feats], tr.log_visitors)
            preds[m.name] = m.predict(va[feats])
            fold_rows.append({"fold": i, "model": m.name, "valid_start": va_start.date(), **metrics(va.log_visitors, preds[m.name])})
        frame = va[["date", "location_id", "log_visitors"]].copy()
        for k, v in preds.items():
            frame[k] = v
        frame["fold"] = i
        oof.append(frame)
        log(f"fold {i}: train to {tr_end.date()}, validate {va_start.date()}..{va_end.date()}")

    oof = pd.concat(oof, ignore_index=True)
    names = [M.name for M in ALL_MODELS]

    # Ensemble weights are learned on the earlier folds and tested on the later
    # ones, so the reported ensemble score is not tuned on its own test data.
    split = n_folds // 2
    early, late = oof[oof.fold < split], oof[oof.fold >= split]
    w_early = ensemble_weights({n: rmse(early.log_visitors, early[n]) for n in names})
    late_scores = {n: metrics(late.log_visitors, late[n]) for n in names}
    late_scores["ensemble_mean"] = metrics(late.log_visitors, late[names].mean(axis=1))
    late_scores["ensemble_weighted"] = metrics(
        late.log_visitors, blend({n: late[n].values for n in names}, w_early)
    )

    # Final weights use all folds
    weights = ensemble_weights({n: rmse(oof.log_visitors, oof[n]) for n in names})

    cv = pd.DataFrame(fold_rows)
    cv.to_csv(out / "cv_metrics.csv", index=False)
    holdout = pd.DataFrame(late_scores).T.sort_values("rmsle")
    holdout.to_csv(out / "holdout_comparison.csv")
    log("\nHold-out comparison (later folds, lower is better):")
    log(holdout.round(4).to_string())

    # ---------- final fit & forecast ----------
    future = df[~df.is_history]
    final_preds, importances = {}, []
    for M in ALL_MODELS:
        m = M(seed).fit(hist[feats], hist.log_visitors)
        final_preds[m.name] = m.predict(future[feats])
        importances.append(m.importance(feats).rank(pct=True).rename(m.name))
    fc = future[["date", "location_id", "area", "store_type", "reserve_visitors"]].copy()
    for n in names:
        fc[f"pred_{n}"] = np.clip(np.expm1(final_preds[n]), 0, None).round(1)
    fc["forecast"] = np.clip(np.expm1(blend(final_preds, weights)), 0, None).round(0).astype(int)
    fc.to_csv(out / "forecast.csv", index=False)

    imp = pd.concat(importances, axis=1)
    imp["mean_rank"] = imp.mean(axis=1)
    imp.sort_values("mean_rank", ascending=False).to_csv(out / "feature_importance.csv")

    summary = {
        "horizon_days": horizon,
        "n_folds": n_folds,
        "ensemble_weights": {k: round(v, 4) for k, v in weights.items()},
        "holdout": {k: {m: round(x, 4) for m, x in v.items()} for k, v in late_scores.items()},
        "forecast_window": [str(fc.date.min().date()), str(fc.date.max().date())],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(oof, names, weights, out)
    log(f"\nEnsemble weights: {summary['ensemble_weights']}")
    log(f"Forecast for {summary['forecast_window'][0]} .. {summary['forecast_window'][1]} -> {out / 'forecast.csv'}")
    return summary


def _plot(oof: pd.DataFrame, names, weights, out: Path) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    oof = oof.copy()
    oof["ensemble"] = blend({n: oof[n].values for n in names}, weights)
    daily = oof.groupby("date")[["log_visitors", "ensemble", *names]].apply(lambda g: np.expm1(g).sum()).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=daily.date, y=daily.log_visitors, name="Actual", mode="lines+markers")
    fig.add_scatter(x=daily.date, y=daily.ensemble, name="Ensemble", mode="lines+markers")
    for n in names:
        fig.add_scatter(x=daily.date, y=daily[n], name=n, mode="lines", line={"dash": "dot"}, visible="legendonly")
    fig.update_layout(title="Validation windows: total daily demand, actual vs forecast", yaxis_title="Visitors")
    fig.write_html(out / "validation_plot.html", include_plotlyjs="cdn")


def main() -> None:
    p = argparse.ArgumentParser(description="Demand forecasting pipeline")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="reports")
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--folds", type=int, default=6)
    args = p.parse_args()
    run(args.data, args.out, args.horizon, args.folds)


if __name__ == "__main__":
    main()
