from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(r"c:\Users\jorge\OneDrive\Documentos\Data 606\Project")
FEATURE_DIR = PROJECT_ROOT / "feature_datasets"
SECTOR_PATH = PROJECT_ROOT / "stock_sectors.csv"
OUT_BASE = PROJECT_ROOT / "MultinomialLogit"

START_DATE = pd.Timestamp("2024-11-01")
END_DATE = pd.Timestamp("2025-10-31")
TRAIN_START = pd.Timestamp("2024-11-01")
TRAIN_END = pd.Timestamp("2025-08-31")
VAL_START = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-10-31")

RETURN_COL = "Return"
TARGET_COL = "Return_Status"

N_SPLITS_DEFAULT = 5
C_GRID = np.logspace(-2, 1.5, 8)
MAX_ITER = 3000
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Return_Lag10",
    "Return_Lag20",
    "Sentiment_Lag1",
    "Sentiment_Lag2",
    "Sentiment_Lag3",
    "Sentiment_Lag5",
    "Sentiment_Lag10",
    "Sentiment_Lag20",
    "Sentiment_Lag1_squared",
    "Sentiment_Lag1_cubic",
    "VIX_Lag1",
    "VIX_Lag2",
    "VIX_Lag3",
    "VIX_Lag5",
    "VIX_Lag10",
    "VIX_Lag20",
    "roe",
    "roa",
    "op_margin",
    "debt_to_equity",
    "liquidity_ratio",
    "current_ratio",
    "free_cf_margin",
    "revenue_growth",
    "ocf_to_assets",
    "GDP_GDP_SA_PC_QOQ",
    "GDP_GDP_SA_PC_YOY",
    "GDP_GDP_NSA_PC_QOQ",
    "GDP_GDP_NSA_PC_YOY",
    "IPI_IPI",
    "IPI_IPI_YOY",
    "IPI_IPI_QOQ",
    "IPI_IPI_SA",
    "IPI_IPI_SA_YOY",
    "IPI_IPI_SA_QOQ",
    "UNEMP_UNRATE",
    "UNEMP_UNRATE_PC1",
    "UNEMP_UNRATE_PCH",
    "UNEMP_UNRATENSA",
    "UNEMP_UNRATENSA_PC1",
    "UNEMP_UNRATENSA_PCH",
]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_return_status(ret: pd.Series, thr: float) -> pd.Series:
    y = pd.Series(index=ret.index, dtype="object")
    y[ret > thr] = "up"
    y[ret < -thr] = "down"
    y[ret.between(-thr, thr, inclusive="both")] = "same"
    return y


def choose_balanced_threshold(train_ret: pd.Series, percentiles: np.ndarray) -> tuple[float, pd.DataFrame]:
    train_ret = train_ret.dropna()
    abs_ret = train_ret.abs()
    rows: list[dict] = []
    for p in percentiles:
        thr = np.percentile(abs_ret, p)
        y = make_return_status(train_ret, thr)
        share = y.value_counts(normalize=True).reindex(["down", "same", "up"], fill_value=0.0)
        imb = ((share - (1.0 / 3.0)) ** 2).sum()
        rows.append(
            {
                "percentile": float(p),
                "threshold": float(thr),
                "share_down": float(share["down"]),
                "share_same": float(share["same"]),
                "share_up": float(share["up"]),
                "imbalance_score": float(imb),
            }
        )
    grid = pd.DataFrame(rows).sort_values(["imbalance_score", "percentile"]).reset_index(drop=True)
    return float(grid.loc[0, "threshold"]), grid


def ts_splits_by_date(dates: pd.Series, n_splits: int):
    d = pd.to_datetime(dates)
    uniq = np.array(sorted(pd.Series(d.unique())))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_d_idx, te_d_idx) in enumerate(tscv.split(uniq), start=1):
        tr_dates = set(uniq[tr_d_idx])
        te_dates = set(uniq[te_d_idx])
        tr_idx = np.where(d.isin(tr_dates))[0]
        te_idx = np.where(d.isin(te_dates))[0]
        yield fold, tr_idx, te_idx


def metric_row(split: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_samples": len(y_true),
    }


def build_panel_for_all() -> pd.DataFrame:
    sector_map = pd.read_csv(SECTOR_PATH)[["ticker", "sector"]].dropna(subset=["ticker", "sector"]).copy()
    sector_map["ticker"] = sector_map["ticker"].astype(str).str.strip()
    ticker_to_sector = dict(zip(sector_map["ticker"], sector_map["sector"]))

    feature_files = sorted(FEATURE_DIR.glob("*_features.csv"))
    if not feature_files:
        raise FileNotFoundError(f"No *_features.csv files found in {FEATURE_DIR}")

    feature_tickers = [p.name.replace("_features.csv", "") for p in feature_files]
    used_tickers = [t for t in feature_tickers if t in ticker_to_sector]
    if not used_tickers:
        raise ValueError("No overlap between feature files and stock_sectors.csv tickers")

    frames = []
    derived_cols = {"Sentiment_Lag1_squared", "Sentiment_Lag1_cubic"}
    needed = ["Date", RETURN_COL] + [c for c in NUMERIC_FEATURES if c not in derived_cols]
    for t in used_tickers:
        p = FEATURE_DIR / f"{t}_features.csv"
        df = pd.read_csv(p)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} missing columns: {missing}")
        tmp = df[needed].copy()
        tmp["Sentiment_Lag1"] = pd.to_numeric(tmp["Sentiment_Lag1"], errors="coerce")
        tmp["Sentiment_Lag1_squared"] = tmp["Sentiment_Lag1"] ** 2
        tmp["Sentiment_Lag1_cubic"] = tmp["Sentiment_Lag1"] ** 3
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp["ticker"] = t
        tmp["sector"] = ticker_to_sector[t]
        frames.append(tmp)

    panel = pd.concat(frames, ignore_index=True).dropna(subset=["Date"])
    panel = panel[(panel["Date"] >= START_DATE) & (panel["Date"] <= END_DATE)].copy()
    panel = panel.sort_values(["Date", "ticker"]).reset_index(drop=True)
    return panel


def run_sector_model(sector_name: str, sector_df: pd.DataFrame, out_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sector_slug = (
        sector_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("-", "_")
    )
    sec_dir = out_dir / f"sector_{sector_slug}"
    sec_dir.mkdir(parents=True, exist_ok=True)

    train_df = sector_df[(sector_df["Date"] >= TRAIN_START) & (sector_df["Date"] <= TRAIN_END)].copy()
    val_df = sector_df[(sector_df["Date"] >= VAL_START) & (sector_df["Date"] <= VAL_END)].copy()

    if len(train_df) < 120 or len(val_df) < 30:
        raise ValueError(f"insufficient samples (train={len(train_df)}, val={len(val_df)})")

    threshold, thr_grid = choose_balanced_threshold(train_df[RETURN_COL], np.arange(10, 46, 1))
    sector_df = sector_df.copy()
    sector_df[TARGET_COL] = make_return_status(sector_df[RETURN_COL], threshold)
    train_df = sector_df[(sector_df["Date"] >= TRAIN_START) & (sector_df["Date"] <= TRAIN_END)].copy()
    val_df = sector_df[(sector_df["Date"] >= VAL_START) & (sector_df["Date"] <= VAL_END)].copy()

    X_train = train_df[NUMERIC_FEATURES + ["ticker"]].copy()
    y_train = train_df[TARGET_COL].copy()
    X_val = val_df[NUMERIC_FEATURES + ["ticker"]].copy()
    y_val = val_df[TARGET_COL].copy()

    train_class_count = y_train.nunique()
    val_class_count = y_val.nunique()
    if train_class_count < 2 or val_class_count < 2:
        raise ValueError(
            f"not enough target classes (train_classes={train_class_count}, val_classes={val_class_count})"
        )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "ticker",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_ohe()),
                    ]
                ),
                ["ticker"],
            ),
        ],
        remainder="drop",
    )

    dtr = train_df["Date"].reset_index(drop=True)
    Xtr = X_train.reset_index(drop=True)
    ytr = y_train.reset_index(drop=True)

    unique_train_dates = dtr.nunique()
    n_splits = min(N_SPLITS_DEFAULT, max(2, unique_train_dates - 1))

    cv_rows = []
    for C in C_GRID:
        fold_scores = []
        for fold, tr_idx, te_idx in ts_splits_by_date(dtr, n_splits=n_splits):
            y_tr_fold = ytr.iloc[tr_idx]
            y_te_fold = ytr.iloc[te_idx]
            if y_tr_fold.nunique() < 2:
                continue
            clf = Pipeline(
                steps=[
                    ("preprocess", preprocess),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l1",
                            C=float(C),
                            solver="saga",
                            multi_class="multinomial",
                            max_iter=MAX_ITER,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            clf.fit(Xtr.iloc[tr_idx], y_tr_fold)
            pred = clf.predict(Xtr.iloc[te_idx])

            acc = accuracy_score(y_te_fold, pred)
            f1m = f1_score(y_te_fold, pred, average="macro", zero_division=0)
            cv_rows.append(
                {
                    "sector": sector_name,
                    "C": float(C),
                    "fold": fold,
                    "accuracy": acc,
                    "f1_macro": f1m,
                    "n_samples": len(te_idx),
                }
            )
            fold_scores.append(f1m)

        if not fold_scores:
            cv_rows.append(
                {
                    "sector": sector_name,
                    "C": float(C),
                    "fold": -1,
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "n_samples": 0,
                }
            )

    cv_detail = pd.DataFrame(cv_rows)
    cv_summary = (
        cv_detail.groupby("C", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1_macro=("f1_macro", "mean"),
            std_f1_macro=("f1_macro", "std"),
            n_valid_folds=("f1_macro", lambda s: int(s.notna().sum())),
        )
        .sort_values(["mean_f1_macro", "mean_accuracy"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    if cv_summary["mean_f1_macro"].notna().sum() == 0:
        raise ValueError("all CV folds invalid")

    best_C = float(cv_summary.loc[cv_summary["mean_f1_macro"].notna()].iloc[0]["C"])

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    C=best_C,
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=MAX_ITER,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    fold_rows = []
    oof_true = []
    oof_pred = []
    for fold, tr_idx, te_idx in ts_splits_by_date(dtr, n_splits=n_splits):
        y_tr_fold = ytr.iloc[tr_idx]
        y_te_fold = ytr.iloc[te_idx]
        if y_tr_fold.nunique() < 2:
            continue
        m = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        C=best_C,
                        solver="saga",
                        multi_class="multinomial",
                        max_iter=MAX_ITER,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        m.fit(Xtr.iloc[tr_idx], y_tr_fold)
        p = m.predict(Xtr.iloc[te_idx])
        fr = metric_row(f"test_fold_{fold}", y_te_fold, p)
        fr["fold"] = fold
        fold_rows.append(fr)
        oof_true.append(y_te_fold)
        oof_pred.append(pd.Series(p, index=y_te_fold.index))

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        oof_true_all = pd.concat(oof_true).sort_index()
        oof_pred_all = pd.concat(oof_pred).sort_index()
        test_oof = metric_row("testing_tscv_oof", oof_true_all, oof_pred_all)
        test_mean = {
            "split": "testing_tscv_fold_mean",
            "accuracy": fold_df["accuracy"].mean(),
            "balanced_accuracy": fold_df["balanced_accuracy"].mean(),
            "f1_macro": fold_df["f1_macro"].mean(),
            "f1_weighted": fold_df["f1_weighted"].mean(),
            "n_samples": int(fold_df["n_samples"].sum()),
        }
    else:
        test_oof = {
            "split": "testing_tscv_oof",
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1_macro": np.nan,
            "f1_weighted": np.nan,
            "n_samples": 0,
        }
        test_mean = {
            "split": "testing_tscv_fold_mean",
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1_macro": np.nan,
            "f1_weighted": np.nan,
            "n_samples": 0,
        }

    metrics_df = pd.DataFrame(
        [
            metric_row("train", y_train, pred_train),
            test_oof,
            test_mean,
            metric_row("validation_out_of_sample", y_val, pred_val),
        ]
    )
    metrics_df.insert(0, "sector", sector_name)

    val_pred = val_df[["Date", "ticker", "sector", RETURN_COL, TARGET_COL]].copy().reset_index(drop=True)
    val_pred["pred_status"] = pred_val

    proba = model.predict_proba(X_val)
    proba_cols = [f"p_{c}" for c in model.named_steps["model"].classes_]
    val_pred = pd.concat([val_pred, pd.DataFrame(proba, columns=proba_cols)], axis=1)

    labels = sorted(y_val.dropna().unique().tolist())
    cm = confusion_matrix(y_val, pred_val, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])

    pre = model.named_steps["preprocess"]
    clf = model.named_steps["model"]
    feat_names = pre.get_feature_names_out()
    coef_df = pd.DataFrame(
        clf.coef_.T,
        index=feat_names,
        columns=[f"coef_{c}" for c in clf.classes_],
    )
    coef_df["max_abs_coef"] = coef_df.abs().max(axis=1)
    coef_df = coef_df.sort_values("max_abs_coef", ascending=False)

    thr_grid.to_csv(sec_dir / "threshold_grid_train_percentiles.csv", index=False)
    cv_detail.to_csv(sec_dir / "cv_detail_by_fold.csv", index=False)
    cv_summary.to_csv(sec_dir / "cv_summary_by_C.csv", index=False)
    metrics_df.to_csv(sec_dir / "metrics_train_testing_validation.csv", index=False)
    fold_df.to_csv(sec_dir / "testing_tscv_fold_metrics.csv", index=False)
    val_pred.to_csv(sec_dir / "validation_predictions.csv", index=False)
    cm_df.to_csv(sec_dir / "validation_confusion_matrix_counts.csv", index=True)
    coef_df.to_csv(sec_dir / "lasso_coefficients.csv", index=True)
    (sec_dir / "train_classification_report.txt").write_text(
        classification_report(y_train, pred_train, digits=4, zero_division=0), encoding="utf-8"
    )
    (sec_dir / "validation_classification_report.txt").write_text(
        classification_report(y_val, pred_val, digits=4, zero_division=0), encoding="utf-8"
    )

    summary = {
        "sector": sector_name,
        "n_tickers": int(sector_df["ticker"].nunique()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "threshold": float(threshold),
        "best_C": float(best_C),
        "n_splits_used": int(n_splits),
        "train_accuracy": float(metrics_df.loc[metrics_df["split"] == "train", "accuracy"].iloc[0]),
        "train_f1_macro": float(metrics_df.loc[metrics_df["split"] == "train", "f1_macro"].iloc[0]),
        "test_oof_accuracy": float(metrics_df.loc[metrics_df["split"] == "testing_tscv_oof", "accuracy"].iloc[0]),
        "test_oof_f1_macro": float(metrics_df.loc[metrics_df["split"] == "testing_tscv_oof", "f1_macro"].iloc[0]),
        "val_accuracy": float(metrics_df.loc[metrics_df["split"] == "validation_out_of_sample", "accuracy"].iloc[0]),
        "val_f1_macro": float(metrics_df.loc[metrics_df["split"] == "validation_out_of_sample", "f1_macro"].iloc[0]),
        "out_dir": str(sec_dir),
    }
    return summary, metrics_df, fold_df, val_pred


def main():
    panel = build_panel_for_all()
    n_tickers = panel["ticker"].nunique()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"sector_models_{n_tickers}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_metrics = []
    all_folds = []
    all_val_preds = []
    errors = []

    sectors = sorted(panel["sector"].dropna().unique().tolist())
    print(f"Sectors to model: {len(sectors)}")
    for i, sector in enumerate(sectors, start=1):
        sec_df = panel[panel["sector"] == sector].copy()
        print(f"[{i:02d}/{len(sectors)}] {sector} | tickers={sec_df['ticker'].nunique()} rows={len(sec_df)}")
        try:
            summary, mdf, fdf, vdf = run_sector_model(sector, sec_df, out_dir)
            summaries.append(summary)
            all_metrics.append(mdf)
            if not fdf.empty:
                all_folds.append(fdf.assign(sector=sector))
            all_val_preds.append(vdf.assign(model_sector=sector))
            print(
                "   OK | val_acc={:.4f} val_f1={:.4f} best_C={:.5f} thr={:.4%}".format(
                    summary["val_accuracy"], summary["val_f1_macro"], summary["best_C"], summary["threshold"]
                )
            )
        except Exception as e:
            errors.append({"sector": sector, "error": str(e)})
            print(f"   FAILED | {e}")

    summary_df = pd.DataFrame(summaries).sort_values("val_accuracy", ascending=False)
    summary_df.to_csv(out_dir / "sector_model_summary.csv", index=False)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "all_sector_metrics_long.csv", index=False)
    if all_folds:
        pd.concat(all_folds, ignore_index=True).to_csv(out_dir / "all_sector_fold_metrics.csv", index=False)
    if all_val_preds:
        pd.concat(all_val_preds, ignore_index=True).to_csv(out_dir / "all_sector_validation_predictions.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "sector_model_errors.csv", index=False)

    lines = [
        "Sector-wise multinomial logit (LASSO)",
        f"Run directory: {out_dir}",
        f"Tickers used: {n_tickers}",
        f"Sectors attempted: {len(sectors)}",
        f"Sectors succeeded: {len(summary_df)}",
        f"Sectors failed: {len(errors)}",
        "",
    ]
    if not summary_df.empty:
        lines.append("Top sectors by validation accuracy:")
        lines.append(summary_df[["sector", "n_tickers", "val_accuracy", "val_f1_macro"]].head(10).to_string(index=False))
        lines.append("")
        lines.append("Bottom sectors by validation accuracy:")
        lines.append(summary_df[["sector", "n_tickers", "val_accuracy", "val_f1_macro"]].tail(10).to_string(index=False))
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.append(pd.DataFrame(errors).to_string(index=False))
    (out_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE")
    print(f"Output: {out_dir}")
    print(f"Succeeded: {len(summary_df)} | Failed: {len(errors)}")
    if not summary_df.empty:
        print("\nValidation summary:")
        print(summary_df[["sector", "n_tickers", "val_accuracy", "val_f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
