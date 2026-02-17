from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRFRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for this script. Install it with `pip install xgboost`."
    ) from exc


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = PROJECT_ROOT / "ConstructionDataset" / "all_companies_features.csv"
SECTOR_PATH = PROJECT_ROOT / "stock_sectors.csv"
OUT_BASE = PROJECT_ROOT / "OptimizedXGboost" / "runs" / "local"
FEATURE_DIR_PRIMARY = PROJECT_ROOT / "ConstructionDataset" / "feature_datasets_enhanced"
FEATURE_DIR_FALLBACK = PROJECT_ROOT / "feature_datasets"
PANEL_CACHE_DIR = PROJECT_ROOT / "ConstructionDataset" / "cache"
PANEL_CACHE_PATH = PANEL_CACHE_DIR / "all_companies_features_runtime_cache.csv"
USE_TICKER_LEVEL_STORAGE = True
FORCE_REBUILD_PANEL_CACHE = False

START_DATE = pd.Timestamp("2024-11-01")
END_DATE = pd.Timestamp("2025-10-31")
TRAIN_START = pd.Timestamp("2024-11-01")
TRAIN_END = pd.Timestamp("2025-08-31")
VAL_START = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2025-10-31")

RETURN_COL = "Return"
TARGET_COL = "Next_Return"

N_SPLITS_DEFAULT = 5
N_ESTIMATORS_GRID = np.array([200, 400], dtype=int)
MAX_DEPTH_GRID = np.array([4, 6], dtype=int)
MIN_CHILD_WEIGHT_GRID = np.array([1.0, 5.0], dtype=float)
SUBSAMPLE_GRID = np.array([0.7, 0.9], dtype=float)
COLSAMPLE_BYNODE_GRID = np.array([0.7, 0.9], dtype=float)
REG_ALPHA_GRID = np.array([0.0, 0.1], dtype=float)
REG_LAMBDA_GRID = np.array([1.0, 5.0], dtype=float)
TS_MAX_TRAIN_DATES = 126
RANDOM_STATE = 42
SMOKE_SELECTION_SEED = 2026
SMOKE_TEST_N_TICKERS = 50
MIN_TICKERS_PER_SECTOR = 3
VIF_THRESHOLD = 5.0
VIF_EPS = 1e-12


def write_run_snapshot(out_dir: Path) -> None:
    """
    Save an exact copy of the current training script plus run metadata.
    This guarantees each run folder is self-contained and reproducible.
    """
    snapshot_dir = out_dir / "_run_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, snapshot_dir / script_path.name)

    metadata = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script_name": script_path.name,
        "script_path": str(script_path),
        "python_version": sys.version.split()[0],
        "project_root": str(PROJECT_ROOT),
        "target": TARGET_COL,
        "selection_metric": "min_rmse",
        "smoke_test_n_tickers": int(SMOKE_TEST_N_TICKERS),
        "smoke_selection_seed": int(SMOKE_SELECTION_SEED),
        "n_estimators_grid": [int(x) for x in N_ESTIMATORS_GRID],
        "max_depth_grid": [int(x) for x in MAX_DEPTH_GRID],
        "min_child_weight_grid": [float(x) for x in MIN_CHILD_WEIGHT_GRID],
        "subsample_grid": [float(x) for x in SUBSAMPLE_GRID],
        "colsample_bynode_grid": [float(x) for x in COLSAMPLE_BYNODE_GRID],
        "reg_alpha_grid": [float(x) for x in REG_ALPHA_GRID],
        "reg_lambda_grid": [float(x) for x in REG_LAMBDA_GRID],
        "vif_threshold": float(VIF_THRESHOLD),
        "ts_max_train_dates": int(TS_MAX_TRAIN_DATES),
        "panel_source_mode": "ticker-level + cache" if USE_TICKER_LEVEL_STORAGE else "single consolidated csv",
        "panel_cache_path": str(PANEL_CACHE_PATH),
    }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
        metadata["git_commit"] = commit
        metadata["git_branch"] = branch
    except Exception:
        metadata["git_commit"] = "unavailable"
        metadata["git_branch"] = "unavailable"

    (snapshot_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


NUMERIC_FEATURES = [
    "Dividends_Lag1",
    "Dividends_Lag2",
    "Dividends_Lag3",
    "Dividends_Lag5",
    "Dividends_Lag10",
    "Dividends_Lag20",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Return_Lag10",
    "Return_Lag20",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "Volatility_5",
    "ATR_5",
    "Volatility_10",
    "ATR_10",
    "Volatility_20",
    "ATR_20",
    "Sentiment_Lag1",
    "Sentiment_Lag2",
    "Sentiment_Lag3",
    "Sentiment_Lag5",
    "Sentiment_Lag10",
    "Sentiment_Lag20",
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
    "log_mktcap",
    "book_to_market",
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


def rmse_value(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def directional_accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def ts_splits_by_date(dates: pd.Series, n_splits: int, max_train_size: int | None = None):
    d = pd.to_datetime(dates)
    uniq = np.array(sorted(pd.Series(d.unique())))
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    for fold, (tr_d_idx, te_d_idx) in enumerate(tscv.split(uniq), start=1):
        tr_dates = set(uniq[tr_d_idx])
        te_dates = set(uniq[te_d_idx])
        tr_idx = np.where(d.isin(tr_dates))[0]
        te_idx = np.where(d.isin(te_dates))[0]
        yield fold, tr_idx, te_idx


def metric_row(split: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    baseline_pred = np.full(shape=yt.shape, fill_value=float(np.nanmean(yt)), dtype=float)
    rmse = rmse_value(yt, yp)
    mae = float(mean_absolute_error(yt, yp))
    baseline_rmse = rmse_value(yt, baseline_pred)
    baseline_mae = float(mean_absolute_error(yt, baseline_pred))

    if len(yt) > 1 and np.nanstd(yt) > 0:
        r2 = float(r2_score(yt, yp))
        baseline_r2 = float(r2_score(yt, baseline_pred))
    else:
        r2 = np.nan
        baseline_r2 = np.nan

    return {
        "split": split,
        "rmse": rmse,
        "baseline_rmse_mean": baseline_rmse,
        "mae": mae,
        "baseline_mae_mean": baseline_mae,
        "r2": r2,
        "baseline_r2_mean": baseline_r2,
        "directional_accuracy": directional_accuracy(yt, yp),
        "n_samples": len(yt),
    }


def build_param_grid() -> list[dict]:
    return [
        {
            "model__n_estimators": [int(x) for x in N_ESTIMATORS_GRID],
            "model__max_depth": [int(x) for x in MAX_DEPTH_GRID],
            "model__min_child_weight": [float(x) for x in MIN_CHILD_WEIGHT_GRID],
            "model__subsample": [float(x) for x in SUBSAMPLE_GRID],
            "model__colsample_bynode": [float(x) for x in COLSAMPLE_BYNODE_GRID],
            "model__reg_alpha": [float(x) for x in REG_ALPHA_GRID],
            "model__reg_lambda": [float(x) for x in REG_LAMBDA_GRID],
        }
    ]


def make_xgbrf_model(hp: dict) -> XGBRFRegressor:
    return XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        min_child_weight=float(hp["min_child_weight"]),
        subsample=float(hp["subsample"]),
        colsample_bynode=float(hp["colsample_bynode"]),
        reg_alpha=float(hp["reg_alpha"]),
        reg_lambda=float(hp["reg_lambda"]),
        learning_rate=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def refit_by_rmse(cv_results: dict) -> int:
    rmse_neg = np.array(cv_results["mean_test_rmse_neg"], dtype=float)
    rmse_neg = np.where(np.isfinite(rmse_neg), rmse_neg, -np.inf)
    return int(np.argmax(rmse_neg))


def _cross_join_by_key(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_ = left.copy()
    right_ = right.copy()
    left_["_key"] = 1
    right_["_key"] = 1
    out = left_.merge(right_, on="_key", how="inner").drop(columns="_key")
    return out


def run_gridsearch_with_target(
    sector_name: str,
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    preprocess: ColumnTransformer,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[GridSearchCV, pd.DataFrame, pd.DataFrame]:
    base_model = XGBRFRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bynode=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        learning_rate=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    gs_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", base_model),
        ]
    )
    gs = GridSearchCV(
        estimator=gs_pipe,
        param_grid=build_param_grid(),
        scoring={
            "rmse_neg": make_scorer(rmse_value, greater_is_better=False),
            "mae_neg": "neg_mean_absolute_error",
            "r2": "r2",
            "directional_accuracy": make_scorer(directional_accuracy),
        },
        refit=refit_by_rmse,
        cv=cv_splits,
        n_jobs=1,
        return_train_score=False,
        error_score=np.nan,
    )
    gs.fit(Xtr, ytr)

    cv_raw = pd.DataFrame(gs.cv_results_)
    cv_summary = pd.DataFrame(
        {
            "n_estimators": pd.to_numeric(cv_raw["param_model__n_estimators"], errors="coerce").astype("Int64"),
            "max_depth": pd.to_numeric(cv_raw["param_model__max_depth"], errors="coerce").astype("Int64"),
            "min_child_weight": pd.to_numeric(cv_raw["param_model__min_child_weight"], errors="coerce"),
            "subsample": pd.to_numeric(cv_raw["param_model__subsample"], errors="coerce"),
            "colsample_bynode": pd.to_numeric(cv_raw["param_model__colsample_bynode"], errors="coerce"),
            "reg_alpha": pd.to_numeric(cv_raw["param_model__reg_alpha"], errors="coerce"),
            "reg_lambda": pd.to_numeric(cv_raw["param_model__reg_lambda"], errors="coerce"),
            "mean_rmse": -cv_raw["mean_test_rmse_neg"],
            "std_rmse": cv_raw["std_test_rmse_neg"],
            "mean_mae": -cv_raw["mean_test_mae_neg"],
            "std_mae": cv_raw["std_test_mae_neg"],
            "mean_r2": cv_raw["mean_test_r2"],
            "std_r2": cv_raw["std_test_r2"],
            "mean_directional_accuracy": cv_raw["mean_test_directional_accuracy"],
            "std_directional_accuracy": cv_raw["std_test_directional_accuracy"],
        }
    )

    rmse_split_cols = [c for c in cv_raw.columns if c.startswith("split") and c.endswith("_test_rmse_neg")]
    cv_summary["n_valid_folds"] = cv_raw[rmse_split_cols].notna().sum(axis=1).astype(int)
    cv_summary = (
        cv_summary.sort_values(
            ["mean_rmse", "mean_mae", "mean_r2", "mean_directional_accuracy"],
            ascending=[True, True, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    if cv_summary["mean_rmse"].notna().sum() == 0:
        raise ValueError("all CV folds invalid")

    cv_base = pd.DataFrame(
        {
            "candidate_id": np.arange(len(cv_raw), dtype=int),
            "n_estimators": pd.to_numeric(cv_raw["param_model__n_estimators"], errors="coerce").astype("Int64"),
            "max_depth": pd.to_numeric(cv_raw["param_model__max_depth"], errors="coerce").astype("Int64"),
            "min_child_weight": pd.to_numeric(cv_raw["param_model__min_child_weight"], errors="coerce"),
            "subsample": pd.to_numeric(cv_raw["param_model__subsample"], errors="coerce"),
            "colsample_bynode": pd.to_numeric(cv_raw["param_model__colsample_bynode"], errors="coerce"),
            "reg_alpha": pd.to_numeric(cv_raw["param_model__reg_alpha"], errors="coerce"),
            "reg_lambda": pd.to_numeric(cv_raw["param_model__reg_lambda"], errors="coerce"),
        }
    )
    fold_sizes = pd.DataFrame(
        {
            "fold": np.arange(1, len(cv_splits) + 1, dtype=int),
            "n_samples": [int(len(te_idx)) for _, te_idx in cv_splits],
        }
    )

    def to_long(metric_suffix: str, out_col: str, negate: bool = False) -> pd.DataFrame:
        cols = [c for c in cv_raw.columns if c.startswith("split") and c.endswith(metric_suffix)]
        if cols:
            long = (
                cv_raw[cols]
                .copy()
                .assign(candidate_id=np.arange(len(cv_raw), dtype=int))
                .melt(id_vars="candidate_id", var_name="metric_col", value_name=out_col)
            )
            long["fold"] = long["metric_col"].str.extract(r"^split(\d+)_")[0].astype(int) + 1
            if negate:
                long[out_col] = -long[out_col]
            return long[["candidate_id", "fold", out_col]]

        empty = _cross_join_by_key(cv_base[["candidate_id"]], fold_sizes[["fold"]])
        empty[out_col] = np.nan
        return empty

    rmse_long = to_long("_test_rmse_neg", "rmse", negate=True)
    mae_long = to_long("_test_mae_neg", "mae", negate=True)
    r2_long = to_long("_test_r2", "r2")
    dir_long = to_long("_test_directional_accuracy", "directional_accuracy")

    cv_detail = (
        rmse_long.merge(mae_long, on=["candidate_id", "fold"], how="outer")
        .merge(r2_long, on=["candidate_id", "fold"], how="outer")
        .merge(dir_long, on=["candidate_id", "fold"], how="outer")
        .merge(cv_base, on="candidate_id", how="left")
        .merge(fold_sizes, on="fold", how="left")
        .sort_values(["candidate_id", "fold"])
        .reset_index(drop=True)
    )
    cv_detail.insert(0, "sector", sector_name)
    cv_detail = cv_detail[
        [
            "sector",
            "n_estimators",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bynode",
            "reg_alpha",
            "reg_lambda",
            "fold",
            "rmse",
            "mae",
            "r2",
            "directional_accuracy",
            "n_samples",
        ]
    ]
    return gs, cv_summary, cv_detail


def _compute_vif_from_matrix(x: pd.DataFrame) -> pd.Series:
    """
    Compute VIF via inverse correlation matrix.
    Assumes x has no NaN and no near-zero variance columns.
    """
    arr = x.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    arr = arr / std
    corr = np.corrcoef(arr, rowvar=False)
    if np.ndim(corr) == 0:
        return pd.Series([1.0], index=x.columns)
    inv_corr = np.linalg.pinv(corr)
    vif = np.diag(inv_corr)
    vif = np.where(np.isfinite(vif), vif, np.inf)
    vif = np.where(vif < 0, np.inf, vif)
    return pd.Series(vif, index=x.columns)


def iterative_vif_filter(
    train_df: pd.DataFrame,
    features: list[str],
    threshold: float = VIF_THRESHOLD,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Iteratively drop numeric features with VIF > threshold.
    Returns:
      kept_features, vif_initial, vif_final, drop_steps
    """
    x_raw = train_df[features].copy()
    for c in features:
        x_raw[c] = pd.to_numeric(x_raw[c], errors="coerce")

    kept = list(features)
    steps: list[dict] = []
    vif_initial = pd.DataFrame(columns=["feature", "vif"])
    initial_captured = False
    step_n = 0

    while True:
        if len(kept) <= 1:
            break

        x = x_raw[kept].copy()
        med = x.median(numeric_only=True)
        x = x.fillna(med)

        # Drop all-missing / still-NaN columns after median fill
        nan_cols = [c for c in kept if x[c].isna().all()]
        if nan_cols:
            for c in nan_cols:
                step_n += 1
                steps.append(
                    {
                        "step": step_n,
                        "dropped_feature": c,
                        "vif_at_drop": np.nan,
                        "reason": "all_missing",
                    }
                )
                kept.remove(c)
            continue

        # Drop near-zero variance columns (cannot compute stable VIF)
        std = x[kept].std(ddof=0)
        low_var_cols = std[std <= VIF_EPS].index.tolist()
        if low_var_cols:
            for c in low_var_cols:
                step_n += 1
                steps.append(
                    {
                        "step": step_n,
                        "dropped_feature": c,
                        "vif_at_drop": np.nan,
                        "reason": "near_zero_variance",
                    }
                )
                kept.remove(c)
            continue

        vif_s = _compute_vif_from_matrix(x[kept])
        if not initial_captured:
            vif_initial = (
                vif_s.rename("vif")
                .rename_axis("feature")
                .reset_index()
                .sort_values("vif", ascending=False)
                .reset_index(drop=True)
            )
            initial_captured = True

        max_feat = str(vif_s.idxmax())
        max_vif = float(vif_s[max_feat])
        if max_vif > threshold and len(kept) > 1:
            step_n += 1
            steps.append(
                {
                    "step": step_n,
                    "dropped_feature": max_feat,
                    "vif_at_drop": max_vif,
                    "reason": "vif_above_threshold",
                }
            )
            kept.remove(max_feat)
            continue
        break

    if len(kept) == 0:
        raise ValueError("VIF filtering removed all numeric features")

    # Final VIF
    x_final = x_raw[kept].copy()
    med_final = x_final.median(numeric_only=True)
    x_final = x_final.fillna(med_final)
    std_final = x_final.std(ddof=0)
    valid_final = [c for c in kept if std_final[c] > VIF_EPS and not x_final[c].isna().all()]
    if len(valid_final) == 0:
        raise ValueError("No valid features remaining after VIF filtering")
    if len(valid_final) == 1:
        vif_final = pd.DataFrame([{"feature": valid_final[0], "vif": 1.0}])
    else:
        vif_final = (
            _compute_vif_from_matrix(x_final[valid_final])
            .rename("vif")
            .rename_axis("feature")
            .reset_index()
            .sort_values("vif", ascending=False)
            .reset_index(drop=True)
        )

    drop_steps = pd.DataFrame(steps)
    return valid_final, vif_initial, vif_final, drop_steps


def select_smoke_tickers(
    panel: pd.DataFrame,
    n_tickers: int = SMOKE_TEST_N_TICKERS,
    min_per_sector: int = MIN_TICKERS_PER_SECTOR,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], pd.DataFrame]:
    uniq = panel[["ticker", "sector"]].dropna(subset=["ticker", "sector"]).drop_duplicates().copy()
    if uniq.empty:
        raise ValueError("No ticker/sector data available for smoke-test selection")

    sector_counts = uniq.groupby("sector")["ticker"].nunique().sort_values(ascending=False)
    sectors = sector_counts.index.tolist()
    n_sectors = len(sectors)
    if n_tickers < n_sectors:
        raise ValueError(f"n_tickers={n_tickers} is smaller than number of sectors={n_sectors}")

    base_alloc = {s: min(min_per_sector, int(sector_counts[s])) for s in sectors}
    base_total = int(sum(base_alloc.values()))
    if base_total > n_tickers:
        base_alloc = {s: 1 for s in sectors}
        base_total = n_sectors

    remaining = n_tickers - base_total
    extra_cap = {s: int(sector_counts[s] - base_alloc[s]) for s in sectors}
    total_extra_cap = int(sum(extra_cap.values()))
    alloc = base_alloc.copy()

    if remaining > 0 and total_extra_cap > 0:
        raw = {s: (remaining * extra_cap[s] / total_extra_cap) for s in sectors}
        add = {s: min(extra_cap[s], int(np.floor(raw[s]))) for s in sectors}
        for s in sectors:
            alloc[s] += add[s]
        left = remaining - int(sum(add.values()))
        frac_order = sorted(sectors, key=lambda s: raw[s] - np.floor(raw[s]), reverse=True)
        for s in frac_order:
            if left <= 0:
                break
            if alloc[s] < sector_counts[s]:
                alloc[s] += 1
                left -= 1

    rng = np.random.RandomState(random_state)
    chosen: list[str] = []
    for s in sectors:
        pool = sorted(uniq.loc[uniq["sector"] == s, "ticker"].tolist())
        k = int(min(alloc[s], len(pool)))
        if k <= 0:
            continue
        if k == len(pool):
            chosen.extend(pool)
        else:
            idx = rng.choice(len(pool), size=k, replace=False)
            chosen.extend([pool[i] for i in sorted(idx)])

    chosen = sorted(set(chosen))
    if len(chosen) > n_tickers:
        rng = np.random.RandomState(random_state)
        chosen = sorted(rng.choice(chosen, size=n_tickers, replace=False).tolist())

    alloc_df = (
        pd.Series(chosen, name="ticker")
        .to_frame()
        .merge(uniq, on="ticker", how="left")
        .groupby("sector", as_index=False)
        .agg(n_tickers=("ticker", "nunique"))
        .sort_values("n_tickers", ascending=False)
    )
    return chosen, alloc_df


def build_panel_for_all() -> pd.DataFrame:
    def _discover_feature_files() -> tuple[list[Path], Path]:
        primary_files = sorted(FEATURE_DIR_PRIMARY.glob("*_features.csv")) if FEATURE_DIR_PRIMARY.exists() else []
        if primary_files:
            return primary_files, FEATURE_DIR_PRIMARY
        fallback_files = sorted(FEATURE_DIR_FALLBACK.glob("*_features.csv")) if FEATURE_DIR_FALLBACK.exists() else []
        if fallback_files:
            return fallback_files, FEATURE_DIR_FALLBACK
        return [], FEATURE_DIR_PRIMARY

    def _build_panel_from_ticker_files(required_cols: list[str]) -> pd.DataFrame:
        files, source_dir = _discover_feature_files()
        if not files:
            raise FileNotFoundError(
                f"No ticker-level feature files found in '{FEATURE_DIR_PRIMARY}' or '{FEATURE_DIR_FALLBACK}'."
            )
        print(f"Building consolidated panel from ticker-level files in: {source_dir}")
        frames: list[pd.DataFrame] = []
        required_set = set(required_cols)
        for fp in files:
            try:
                cols = pd.read_csv(fp, nrows=0).columns.tolist()
            except Exception:
                continue
            keep_cols = [c for c in cols if c in required_set]
            if "Date" not in keep_cols:
                continue
            f = pd.read_csv(fp, usecols=keep_cols)
            if "ticker" not in f.columns:
                f["ticker"] = fp.name.replace("_features.csv", "")
            for c in required_cols:
                if c not in f.columns:
                    f[c] = np.nan
            f = f[required_cols]
            frames.append(f)
        if not frames:
            raise ValueError("Failed to build panel from ticker-level files (no valid files with required columns).")
        panel_built = pd.concat(frames, ignore_index=True)
        print(f"Built consolidated panel from ticker files: rows={len(panel_built):,} cols={panel_built.shape[1]}")
        return panel_built

    sector_map = pd.read_csv(SECTOR_PATH)[["ticker", "sector", "industry"]].dropna(subset=["ticker", "sector"]).copy()
    sector_map["ticker"] = sector_map["ticker"].astype(str).str.strip()
    needed = ["Date", "ticker", RETURN_COL] + NUMERIC_FEATURES
    panel_raw: pd.DataFrame
    if USE_TICKER_LEVEL_STORAGE:
        if PANEL_CACHE_PATH.exists() and not FORCE_REBUILD_PANEL_CACHE:
            print(f"Loading consolidated panel from cache: {PANEL_CACHE_PATH}")
            panel_raw = pd.read_csv(PANEL_CACHE_PATH)
        else:
            panel_raw = _build_panel_from_ticker_files(needed)
            PANEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            panel_raw.to_csv(PANEL_CACHE_PATH, index=False)
            print(f"Saved consolidated panel cache: {PANEL_CACHE_PATH}")
    else:
        if not PANEL_PATH.exists():
            raise FileNotFoundError(f"Panel dataset not found: {PANEL_PATH}")
        panel_raw = pd.read_csv(PANEL_PATH)

    missing = [c for c in needed if c not in panel_raw.columns]
    if missing:
        raise ValueError(f"Consolidated panel is missing columns: {missing}")

    panel = panel_raw[needed].copy()
    panel["ticker"] = panel["ticker"].astype(str).str.strip()
    panel = panel.merge(sector_map, on="ticker", how="inner")
    if panel.empty:
        raise ValueError("No overlap between panel tickers and stock_sectors.csv tickers")

    for c in NUMERIC_FEATURES + [RETURN_COL]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")

    panel = panel.dropna(subset=["Date"])
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

    sector_df = sector_df.sort_values(["ticker", "Date"]).copy()
    sector_df[TARGET_COL] = sector_df.groupby("ticker", sort=False)[RETURN_COL].shift(-1)

    train_df = sector_df[(sector_df["Date"] >= TRAIN_START) & (sector_df["Date"] <= TRAIN_END)].copy()
    val_df = sector_df[(sector_df["Date"] >= VAL_START) & (sector_df["Date"] <= VAL_END)].copy()
    train_df = train_df.dropna(subset=[TARGET_COL]).copy()
    val_df = val_df.dropna(subset=[TARGET_COL]).copy()

    if len(train_df) < 120 or len(val_df) < 30:
        raise ValueError(f"insufficient samples (train={len(train_df)}, val={len(val_df)})")

    vif_features, vif_initial_df, vif_final_df, vif_steps_df = iterative_vif_filter(
        train_df,
        NUMERIC_FEATURES,
        threshold=VIF_THRESHOLD,
    )

    X_train = train_df[vif_features + ["ticker", "industry"]].copy()
    y_train = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
    X_val = val_df[vif_features + ["ticker", "industry"]].copy()
    y_val = pd.to_numeric(val_df[TARGET_COL], errors="coerce")

    tr_valid = y_train.notna()
    va_valid = y_val.notna()
    X_train = X_train.loc[tr_valid].copy()
    y_train = y_train.loc[tr_valid].copy()
    train_df = train_df.loc[tr_valid].copy()
    X_val = X_val.loc[va_valid].copy()
    y_val = y_val.loc[va_valid].copy()
    val_df = val_df.loc[va_valid].copy()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                vif_features,
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
            (
                "industry",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_ohe()),
                    ]
                ),
                ["industry"],
            ),
        ],
        remainder="drop",
    )

    dtr = train_df["Date"].reset_index(drop=True)
    Xtr = X_train.reset_index(drop=True)
    ytr = y_train.reset_index(drop=True)

    unique_train_dates = int(dtr.nunique())
    if unique_train_dates < 3:
        raise ValueError("insufficient unique train dates for TimeSeriesSplit")
    n_splits = min(N_SPLITS_DEFAULT, max(2, unique_train_dates - 1))

    cv_splits = [
        (tr_idx, te_idx)
        for _, tr_idx, te_idx in ts_splits_by_date(dtr, n_splits=n_splits, max_train_size=TS_MAX_TRAIN_DATES)
    ]
    if len(cv_splits) == 0:
        raise ValueError("no valid time-series CV splits")

    best_search, cv_summary, cv_detail = run_gridsearch_with_target(
        sector_name=sector_name,
        Xtr=Xtr,
        ytr=ytr,
        preprocess=preprocess,
        cv_splits=cv_splits,
    )

    best_params = best_search.best_params_
    best_hp = {
        "n_estimators": int(best_params["model__n_estimators"]),
        "max_depth": int(best_params["model__max_depth"]),
        "min_child_weight": float(best_params["model__min_child_weight"]),
        "subsample": float(best_params["model__subsample"]),
        "colsample_bynode": float(best_params["model__colsample_bynode"]),
        "reg_alpha": float(best_params["model__reg_alpha"]),
        "reg_lambda": float(best_params["model__reg_lambda"]),
    }

    model = best_search.best_estimator_
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    fold_rows = []
    oof_true = []
    oof_pred = []
    for fold, (tr_idx, te_idx) in enumerate(cv_splits, start=1):
        y_te_fold = ytr.iloc[te_idx]
        m = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", make_xgbrf_model(best_hp)),
            ]
        )
        m.fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
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
            "rmse": float(fold_df["rmse"].mean()),
            "baseline_rmse_mean": float(fold_df["baseline_rmse_mean"].mean()),
            "mae": float(fold_df["mae"].mean()),
            "baseline_mae_mean": float(fold_df["baseline_mae_mean"].mean()),
            "r2": float(fold_df["r2"].mean()),
            "baseline_r2_mean": float(fold_df["baseline_r2_mean"].mean()),
            "directional_accuracy": float(fold_df["directional_accuracy"].mean()),
            "n_samples": int(fold_df["n_samples"].sum()),
        }
    else:
        test_oof = {
            "split": "testing_tscv_oof",
            "rmse": np.nan,
            "baseline_rmse_mean": np.nan,
            "mae": np.nan,
            "baseline_mae_mean": np.nan,
            "r2": np.nan,
            "baseline_r2_mean": np.nan,
            "directional_accuracy": np.nan,
            "n_samples": 0,
        }
        test_mean = {
            "split": "testing_tscv_fold_mean",
            "rmse": np.nan,
            "baseline_rmse_mean": np.nan,
            "mae": np.nan,
            "baseline_mae_mean": np.nan,
            "r2": np.nan,
            "baseline_r2_mean": np.nan,
            "directional_accuracy": np.nan,
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

    val_pred = val_df[["Date", "ticker", "sector", "industry", RETURN_COL, TARGET_COL]].copy().reset_index(drop=True)
    val_pred["pred_next_return"] = pred_val
    val_pred["pred_direction"] = np.sign(val_pred["pred_next_return"])
    val_pred["actual_direction"] = np.sign(val_pred[TARGET_COL])
    val_pred["abs_error"] = (val_pred[TARGET_COL] - val_pred["pred_next_return"]).abs()
    val_pred["squared_error"] = (val_pred[TARGET_COL] - val_pred["pred_next_return"]) ** 2

    pre = model.named_steps["preprocess"]
    reg = model.named_steps["model"]
    feat_names = pre.get_feature_names_out()
    imp_df = pd.DataFrame(
        {
            "feature": feat_names,
            "importance": reg.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    cv_detail.to_csv(sec_dir / "cv_detail_by_fold.csv", index=False)
    cv_summary.to_csv(sec_dir / "cv_summary_hyperparams.csv", index=False)
    cv_summary_by_n = (
        cv_detail.groupby("n_estimators", as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            std_directional_accuracy=("directional_accuracy", "std"),
            n_valid_folds=("rmse", lambda s: int(s.notna().sum())),
        )
        .sort_values(["mean_rmse", "mean_mae", "mean_r2"], ascending=[True, True, False], na_position="last")
        .reset_index(drop=True)
    )
    cv_summary_by_n.to_csv(sec_dir / "cv_summary_by_n_estimators.csv", index=False)
    metrics_df.to_csv(sec_dir / "metrics_train_testing_validation.csv", index=False)
    fold_df.to_csv(sec_dir / "testing_tscv_fold_metrics.csv", index=False)
    val_pred.to_csv(sec_dir / "validation_predictions.csv", index=False)
    imp_df.to_csv(sec_dir / "feature_importance.csv", index=False)
    vif_initial_df.to_csv(sec_dir / "vif_initial.csv", index=False)
    vif_final_df.to_csv(sec_dir / "vif_final.csv", index=False)
    vif_steps_df.to_csv(sec_dir / "vif_elimination_steps.csv", index=False)
    pd.Series(vif_features, name="feature").to_csv(sec_dir / "vif_features_kept.txt", index=False, header=False)

    summary = {
        "sector": sector_name,
        "n_tickers": int(sector_df["ticker"].nunique()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_industries": int(train_df["industry"].dropna().nunique()),
        "n_features_before_vif": int(len(NUMERIC_FEATURES)),
        "n_features_after_vif": int(len(vif_features)),
        "n_features_dropped_vif": int(len(NUMERIC_FEATURES) - len(vif_features)),
        "best_n_estimators": int(best_hp["n_estimators"]),
        "best_max_depth": int(best_hp["max_depth"]),
        "best_min_child_weight": float(best_hp["min_child_weight"]),
        "best_subsample": float(best_hp["subsample"]),
        "best_colsample_bynode": float(best_hp["colsample_bynode"]),
        "best_reg_alpha": float(best_hp["reg_alpha"]),
        "best_reg_lambda": float(best_hp["reg_lambda"]),
        "n_splits_used": int(n_splits),
        "max_train_dates": int(TS_MAX_TRAIN_DATES),
        "selection_metric": "min_rmse",
        "train_rmse": float(metrics_df.loc[metrics_df["split"] == "train", "rmse"].iloc[0]),
        "train_mae": float(metrics_df.loc[metrics_df["split"] == "train", "mae"].iloc[0]),
        "train_r2": float(metrics_df.loc[metrics_df["split"] == "train", "r2"].iloc[0]),
        "test_oof_rmse": float(metrics_df.loc[metrics_df["split"] == "testing_tscv_oof", "rmse"].iloc[0]),
        "test_oof_mae": float(metrics_df.loc[metrics_df["split"] == "testing_tscv_oof", "mae"].iloc[0]),
        "test_oof_r2": float(metrics_df.loc[metrics_df["split"] == "testing_tscv_oof", "r2"].iloc[0]),
        "val_rmse": float(metrics_df.loc[metrics_df["split"] == "validation_out_of_sample", "rmse"].iloc[0]),
        "val_mae": float(metrics_df.loc[metrics_df["split"] == "validation_out_of_sample", "mae"].iloc[0]),
        "val_r2": float(metrics_df.loc[metrics_df["split"] == "validation_out_of_sample", "r2"].iloc[0]),
        "val_directional_accuracy": float(
            metrics_df.loc[metrics_df["split"] == "validation_out_of_sample", "directional_accuracy"].iloc[0]
        ),
        "out_dir": str(sec_dir),
    }
    return summary, metrics_df, fold_df, val_pred


def main():
    panel_all = build_panel_for_all()
    selected_tickers, alloc_df = select_smoke_tickers(
        panel_all,
        n_tickers=SMOKE_TEST_N_TICKERS,
        random_state=SMOKE_SELECTION_SEED,
    )
    panel = panel_all[panel_all["ticker"].isin(selected_tickers)].copy()
    n_tickers = panel["ticker"].nunique()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / f"smoke_test_{n_tickers}_sector_models_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_snapshot(out_dir)
    pd.Series(sorted(selected_tickers), name="ticker").to_csv(out_dir / "used_tickers.txt", index=False, header=False)
    alloc_df.to_csv(out_dir / "sector_ticker_allocation.csv", index=False)

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
                (
                    "   OK | val_rmse={:.6f} val_mae={:.6f} val_r2={:.4f} "
                    "val_dir_acc={:.4f} n_estimators={} max_depth={} "
                    "subsample={:.2f} colsample_bynode={:.2f}"
                ).format(
                    summary["val_rmse"],
                    summary["val_mae"],
                    summary["val_r2"],
                    summary["val_directional_accuracy"],
                    summary["best_n_estimators"],
                    summary["best_max_depth"],
                    summary["best_subsample"],
                    summary["best_colsample_bynode"],
                )
            )
        except Exception as exc:
            errors.append({"sector": sector, "error": str(exc)})
            print(f"   FAILED | {exc}")

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("val_rmse", ascending=True).reset_index(drop=True)
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
        "Sector-wise XGBoost random-forest regressor (next-day return target)",
        f"Run directory: {out_dir}",
        f"Panel source mode: {'ticker-level + cache' if USE_TICKER_LEVEL_STORAGE else 'single consolidated csv'}",
        f"Panel cache path: {PANEL_CACHE_PATH}",
        f"Smoke test target tickers: {SMOKE_TEST_N_TICKERS}",
        f"Smoke ticker random seed: {SMOKE_SELECTION_SEED}",
        f"TimeSeriesSplit max_train_dates: {TS_MAX_TRAIN_DATES}",
        f"Selection metric: min_rmse",
        f"Code snapshot: {out_dir / '_run_snapshot' / Path(__file__).name}",
        f"Tickers used: {n_tickers}",
        f"Sectors attempted: {len(sectors)}",
        f"Sectors succeeded: {len(summary_df)}",
        f"Sectors failed: {len(errors)}",
        "",
    ]
    if not summary_df.empty:
        lines.append("Top sectors by validation RMSE (lowest first):")
        lines.append(
            summary_df[["sector", "n_tickers", "val_rmse", "val_mae", "val_r2", "val_directional_accuracy"]]
            .head(10)
            .to_string(index=False)
        )
        lines.append("")
        lines.append("Bottom sectors by validation RMSE (highest first):")
        lines.append(
            summary_df[["sector", "n_tickers", "val_rmse", "val_mae", "val_r2", "val_directional_accuracy"]]
            .tail(10)
            .to_string(index=False)
        )
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.append(pd.DataFrame(errors).to_string(index=False))
    (out_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE")
    print(f"Output: {out_dir}")
    print(f"Succeeded: {len(summary_df)} | Failed: {len(errors)}")
    if not summary_df.empty:
        print("\nValidation summary (best RMSE first):")
        print(
            summary_df[["sector", "n_tickers", "val_rmse", "val_mae", "val_r2", "val_directional_accuracy"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
