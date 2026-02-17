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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = PROJECT_ROOT / "ConstructionDataset" / "all_companies_features.csv"
SECTOR_PATH = PROJECT_ROOT / "stock_sectors.csv"
OUT_BASE = PROJECT_ROOT / "MultinomialLogit" / "runs" / "local"
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
TARGET_COL = "Return_Status"

N_SPLITS_DEFAULT = 5
C_GRID = np.array([0.001, 0.01, 0.1, 1.0], dtype=float)
HYPERPARAM_SEARCH_MODE = "expanded"  # options: "basic", "expanded"
PENALTY_GRID_BASIC = ["l1"]
PENALTY_GRID_EXPANDED = ["l1", "elasticnet"]
L1_RATIO_GRID = [0.2, 0.5, 0.8]  # only used when penalty == "elasticnet"
CLASS_WEIGHT_GRID_BASIC = [None]
CLASS_WEIGHT_GRID_EXPANDED = [None]
# Balanced-target threshold scan (aiming for near-equal down/same/up class shares).
TAU_PERCENTILE_SCAN = np.arange(10, 46, 1)
TAU_TOP_K_BY_BALANCE = 5
VOL_TAU_KAPPA_GRID = np.array([0.4, 0.5, 0.6], dtype=float)
VOL_TAU_STD_WINDOW = 20
VOL_TAU_STD_MIN_PERIODS = 20
COMBO_WEIGHT_ACCURACY = 0.0
COMBO_WEIGHT_F1 = 1.0
MAX_ITER = 3000
RANDOM_STATE = 42
SMOKE_SELECTION_SEED = 2026
SMOKE_TEST_N_TICKERS = 50
MIN_TICKERS_PER_SECTOR = 3
VIF_THRESHOLD = 5.0
VIF_EPS = 1e-12
TARGET_ENGINEERING_MODE = "balanced_threshold"


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
        "selection_metric": f"{COMBO_WEIGHT_ACCURACY:.3g}*accuracy + {COMBO_WEIGHT_F1:.3g}*f1_macro",
        "combo_weight_accuracy": float(COMBO_WEIGHT_ACCURACY),
        "combo_weight_f1": float(COMBO_WEIGHT_F1),
        "smoke_test_n_tickers": int(SMOKE_TEST_N_TICKERS),
        "smoke_selection_seed": int(SMOKE_SELECTION_SEED),
        "c_grid": [float(x) for x in C_GRID],
        "vif_threshold": float(VIF_THRESHOLD),
        "tau_percentile_scan_min": int(TAU_PERCENTILE_SCAN.min()),
        "tau_percentile_scan_max": int(TAU_PERCENTILE_SCAN.max()),
        "tau_kappa_grid": [float(x) for x in VOL_TAU_KAPPA_GRID],
        "ts_train_window_mode": "expanding",
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
    # "RSI_14",
    # "MACD",
    # "MACD_Signal",
    # "MACD_Diff",
    # "Volume_MA_20",
    # "Volume_Ratio",
    # "BB_Middle",
    # "BB_Upper",
    # "BB_Lower",
    # "BB_Width",
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
    # "net_income",
    # "total_revenue",
    # "operating_income",
    # "total_assets",
    # "total_equity",
    # "total_debt",
    # "current_assets",
    # "current_liabilities",
    # "cash_equivalents",
    # "shares_outstanding",
    # "free_cash_flow",
    # "operating_cash_flow",
    "roe",
    "roa",
    "op_margin",
    "debt_to_equity",
    "liquidity_ratio",
    "current_ratio",
    "free_cf_margin",
    "revenue_growth",
    "ocf_to_assets",
    # "prev_revenue",
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


def make_return_status(ret: pd.Series, thr: float) -> pd.Series:
    y = pd.Series(index=ret.index, dtype="object")
    y[ret > thr] = "up"
    y[ret < -thr] = "down"
    y[ret.between(-thr, thr, inclusive="both")] = "same"
    return y


def make_return_status_dynamic(ret: pd.Series, thr_series: pd.Series) -> pd.Series:
    thr = pd.to_numeric(thr_series, errors="coerce").abs()
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
    majority_class_acc = float(y_true.value_counts(normalize=True).max())
    return {
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "baseline_accuracy": majority_class_acc,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_samples": len(y_true),
    }


def build_param_grid() -> list[dict]:
    if HYPERPARAM_SEARCH_MODE == "expanded":
        penalties = PENALTY_GRID_EXPANDED
        class_weights = CLASS_WEIGHT_GRID_EXPANDED
    else:
        penalties = PENALTY_GRID_BASIC
        class_weights = CLASS_WEIGHT_GRID_BASIC

    grid: list[dict] = []
    if "l1" in penalties:
        grid.append(
            {
                "model__C": [float(x) for x in C_GRID],
                "model__penalty": ["l1"],
                "model__class_weight": class_weights,
            }
        )
    if "elasticnet" in penalties:
        grid.append(
            {
                "model__C": [float(x) for x in C_GRID],
                "model__penalty": ["elasticnet"],
                "model__l1_ratio": [float(x) for x in L1_RATIO_GRID],
                "model__class_weight": class_weights,
            }
        )
    return grid


def make_logit_model(hp: dict) -> LogisticRegression:
    kwargs = {
        "penalty": hp["penalty"],
        "C": float(hp["C"]),
        "solver": "saga",
        "multi_class": "multinomial",
        "max_iter": MAX_ITER,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "class_weight": hp["class_weight"],
    }
    if hp["penalty"] == "elasticnet":
        kwargs["l1_ratio"] = float(hp["l1_ratio"])
    return LogisticRegression(**kwargs)


def refit_by_combo(cv_results: dict) -> int:
    acc = np.array(cv_results["mean_test_accuracy"], dtype=float)
    f1 = np.array(cv_results["mean_test_f1_macro"], dtype=float)
    combo = COMBO_WEIGHT_ACCURACY * acc + COMBO_WEIGHT_F1 * f1
    combo = np.where(np.isfinite(combo), combo, -np.inf)
    return int(np.argmax(combo))


def run_gridsearch_with_target(
    sector_name: str,
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    preprocess: ColumnTransformer,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[GridSearchCV, pd.DataFrame, pd.DataFrame]:
    base_model = LogisticRegression(
        solver="saga",
        multi_class="multinomial",
        max_iter=MAX_ITER,
        n_jobs=-1,
        random_state=RANDOM_STATE,
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
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        refit=refit_by_combo,
        cv=cv_splits,
        n_jobs=1,
        return_train_score=False,
        error_score=np.nan,
    )
    gs.fit(Xtr, ytr)

    cv_raw = pd.DataFrame(gs.cv_results_)
    cv_summary = pd.DataFrame(
        {
            "C": pd.to_numeric(cv_raw["param_model__C"], errors="coerce"),
            "penalty": cv_raw["param_model__penalty"].astype(str),
            "l1_ratio": pd.to_numeric(cv_raw.get("param_model__l1_ratio"), errors="coerce"),
            "class_weight_label": cv_raw["param_model__class_weight"].apply(
                lambda x: "balanced" if str(x) == "balanced" else "none"
            ),
            "mean_accuracy": cv_raw["mean_test_accuracy"],
            "std_accuracy": cv_raw["std_test_accuracy"],
            "mean_f1_macro": cv_raw["mean_test_f1_macro"],
            "std_f1_macro": cv_raw["std_test_f1_macro"],
        }
    )
    cv_summary["mean_combo"] = (
        COMBO_WEIGHT_ACCURACY * cv_summary["mean_accuracy"] + COMBO_WEIGHT_F1 * cv_summary["mean_f1_macro"]
    )

    f1_split_cols = [c for c in cv_raw.columns if c.startswith("split") and c.endswith("_test_f1_macro")]
    acc_split_cols = [c for c in cv_raw.columns if c.startswith("split") and c.endswith("_test_accuracy")]
    cv_summary["n_valid_folds"] = cv_raw[f1_split_cols].notna().sum(axis=1).astype(int)
    cv_summary = (
        cv_summary.sort_values(["mean_combo", "mean_accuracy", "mean_f1_macro"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    if cv_summary["mean_combo"].notna().sum() == 0:
        raise ValueError("all CV folds invalid")

    cv_base = pd.DataFrame(
        {
            "candidate_id": np.arange(len(cv_raw), dtype=int),
            "C": pd.to_numeric(cv_raw["param_model__C"], errors="coerce"),
            "penalty": cv_raw["param_model__penalty"].astype(str),
            "l1_ratio": pd.to_numeric(cv_raw.get("param_model__l1_ratio"), errors="coerce"),
            "class_weight_label": cv_raw["param_model__class_weight"].apply(
                lambda x: "balanced" if str(x) == "balanced" else "none"
            ),
        }
    )
    fold_sizes = pd.DataFrame(
        {
            "fold": np.arange(1, len(cv_splits) + 1, dtype=int),
            "n_samples": [int(len(te_idx)) for _, te_idx in cv_splits],
        }
    )

    if acc_split_cols:
        acc_long = (
            cv_raw[acc_split_cols]
            .copy()
            .assign(candidate_id=np.arange(len(cv_raw), dtype=int))
            .melt(id_vars="candidate_id", var_name="metric_col", value_name="accuracy")
        )
        acc_long["fold"] = (
            acc_long["metric_col"].str.extract(r"^split(\d+)_test_accuracy$")[0].astype(int) + 1
        )
        acc_long = acc_long[["candidate_id", "fold", "accuracy"]]
    else:
        acc_long = cv_base[["candidate_id"]].merge(fold_sizes[["fold"]], how="cross")
        acc_long["accuracy"] = np.nan

    if f1_split_cols:
        f1_long = (
            cv_raw[f1_split_cols]
            .copy()
            .assign(candidate_id=np.arange(len(cv_raw), dtype=int))
            .melt(id_vars="candidate_id", var_name="metric_col", value_name="f1_macro")
        )
        f1_long["fold"] = (
            f1_long["metric_col"].str.extract(r"^split(\d+)_test_f1_macro$")[0].astype(int) + 1
        )
        f1_long = f1_long[["candidate_id", "fold", "f1_macro"]]
    else:
        f1_long = cv_base[["candidate_id"]].merge(fold_sizes[["fold"]], how="cross")
        f1_long["f1_macro"] = np.nan

    cv_detail = (
        acc_long.merge(f1_long, on=["candidate_id", "fold"], how="outer")
        .merge(cv_base, on="candidate_id", how="left")
        .merge(fold_sizes, on="fold", how="left")
        .sort_values(["candidate_id", "fold"])
        .reset_index(drop=True)
    )
    cv_detail.insert(0, "sector", sector_name)
    cv_detail = cv_detail[
        ["sector", "C", "penalty", "l1_ratio", "class_weight_label", "fold", "accuracy", "f1_macro", "n_samples"]
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
    train_df = sector_df[(sector_df["Date"] >= TRAIN_START) & (sector_df["Date"] <= TRAIN_END)].copy()
    val_df = sector_df[(sector_df["Date"] >= VAL_START) & (sector_df["Date"] <= VAL_END)].copy()

    if len(train_df) < 120 or len(val_df) < 30:
        raise ValueError(f"insufficient samples (train={len(train_df)}, val={len(val_df)})")

    vif_features, vif_initial_df, vif_final_df, vif_steps_df = iterative_vif_filter(
        train_df,
        NUMERIC_FEATURES,
        threshold=VIF_THRESHOLD,
    )

    X_train = train_df[vif_features + ["ticker", "industry"]].copy()
    y_train = pd.Series(index=train_df.index, dtype="object")
    X_val = val_df[vif_features + ["ticker", "industry"]].copy()
    y_val = pd.Series(index=val_df.index, dtype="object")

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

    unique_train_dates = dtr.nunique()
    n_splits = min(N_SPLITS_DEFAULT, max(2, unique_train_dates - 1))

    cv_splits = [
        (tr_idx, te_idx)
        for _, tr_idx, te_idx in ts_splits_by_date(dtr, n_splits=n_splits)
    ]
    if len(cv_splits) == 0:
        raise ValueError("no valid time-series CV splits")

    # Balanced-target engineering: choose fixed threshold that best balances down/same/up in train.
    threshold, tau_eval_df = choose_balanced_threshold(train_df[RETURN_COL], TAU_PERCENTILE_SCAN)
    best_tau_pct = float(tau_eval_df.loc[0, "percentile"])
    best_kappa = np.nan

    y_train = make_return_status(train_df[RETURN_COL], threshold).reset_index(drop=True)
    y_val = make_return_status(val_df[RETURN_COL], threshold).reset_index(drop=True)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError("target engineering produced fewer than 2 classes")

    best_search, cv_summary, cv_detail = run_gridsearch_with_target(
        sector_name=sector_name,
        Xtr=Xtr,
        ytr=y_train,
        preprocess=preprocess,
        cv_splits=cv_splits,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["tau_dynamic"] = float(threshold)
    val_df["tau_dynamic"] = float(threshold)
    train_df[TARGET_COL] = y_train.to_numpy()
    val_df[TARGET_COL] = y_val.to_numpy()
    ytr = y_train.reset_index(drop=True)

    best_params = best_search.best_params_
    best_hp = {
        "C": float(best_params["model__C"]),
        "penalty": str(best_params["model__penalty"]),
        "l1_ratio": None
        if ("model__l1_ratio" not in best_params or best_params["model__l1_ratio"] is None)
        else float(best_params["model__l1_ratio"]),
        "class_weight": best_params.get("model__class_weight", None),
        "class_weight_label": "balanced"
        if best_params.get("model__class_weight", None) == "balanced"
        else "none",
    }

    # GridSearchCV already refits the best estimator on full training data.
    model = best_search.best_estimator_
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    fold_rows = []
    oof_true = []
    oof_pred = []
    for fold, (tr_idx, te_idx) in enumerate(cv_splits, start=1):
        y_tr_fold = ytr.iloc[tr_idx]
        y_te_fold = ytr.iloc[te_idx]
        if y_tr_fold.nunique() < 2:
            continue
        m = Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    make_logit_model(best_hp),
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

    val_pred = val_df[["Date", "ticker", "sector", "industry", RETURN_COL, TARGET_COL]].copy().reset_index(drop=True)
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

    tau_eval_df.to_csv(sec_dir / "threshold_grid_train_percentiles.csv", index=False)
    cv_detail.to_csv(sec_dir / "cv_detail_by_fold.csv", index=False)
    cv_summary.to_csv(sec_dir / "cv_summary_hyperparams.csv", index=False)
    cv_summary_by_c = (
        cv_detail.groupby("C", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f1_macro=("f1_macro", "mean"),
            std_f1_macro=("f1_macro", "std"),
            n_valid_folds=("f1_macro", lambda s: int(s.notna().sum())),
        )
        .assign(mean_combo=lambda d: COMBO_WEIGHT_ACCURACY * d["mean_accuracy"] + COMBO_WEIGHT_F1 * d["mean_f1_macro"])
        .sort_values(["mean_combo", "mean_accuracy", "mean_f1_macro"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    cv_summary_by_c.to_csv(sec_dir / "cv_summary_by_C.csv", index=False)
    metrics_df.to_csv(sec_dir / "metrics_train_testing_validation.csv", index=False)
    fold_df.to_csv(sec_dir / "testing_tscv_fold_metrics.csv", index=False)
    val_pred.to_csv(sec_dir / "validation_predictions.csv", index=False)
    cm_df.to_csv(sec_dir / "validation_confusion_matrix_counts.csv", index=True)
    coef_df.to_csv(sec_dir / "lasso_coefficients.csv", index=True)
    vif_initial_df.to_csv(sec_dir / "vif_initial.csv", index=False)
    vif_final_df.to_csv(sec_dir / "vif_final.csv", index=False)
    vif_steps_df.to_csv(sec_dir / "vif_elimination_steps.csv", index=False)
    pd.Series(vif_features, name="feature").to_csv(sec_dir / "vif_features_kept.txt", index=False, header=False)
    (sec_dir / "train_classification_report.txt").write_text(
        classification_report(y_train, pred_train, digits=4, zero_division=0), encoding="utf-8"
    )
    (sec_dir / "validation_classification_report.txt").write_text(
        classification_report(y_val, pred_val, digits=4, zero_division=0), encoding="utf-8"
    )

    threshold_effective = float(pd.to_numeric(train_df["tau_dynamic"], errors="coerce").median())
    summary = {
        "sector": sector_name,
        "n_tickers": int(sector_df["ticker"].nunique()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_industries": int(train_df["industry"].dropna().nunique()),
        "n_features_before_vif": int(len(NUMERIC_FEATURES)),
        "n_features_after_vif": int(len(vif_features)),
        "n_features_dropped_vif": int(len(NUMERIC_FEATURES) - len(vif_features)),
        "threshold": threshold_effective,
        "threshold_floor": float(threshold),
        "tau_percentile": float(best_tau_pct) if best_tau_pct is not None else np.nan,
        "tau_kappa": float(best_kappa) if best_kappa is not None else np.nan,
        "best_C": float(best_hp["C"]),
        "best_penalty": str(best_hp["penalty"]),
        "best_l1_ratio": np.nan if best_hp["l1_ratio"] is None else float(best_hp["l1_ratio"]),
        "best_class_weight": str(best_hp["class_weight_label"]),
        "n_splits_used": int(n_splits),
        "max_train_dates": np.nan,
        "train_window_mode": "expanding",
        "selection_metric": f"{COMBO_WEIGHT_ACCURACY:.3g}*accuracy + {COMBO_WEIGHT_F1:.3g}*f1_macro",
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
                "   OK | val_acc={:.4f} val_f1={:.4f} best_C={:.5f} penalty={} cw={} thr={:.4%}".format(
                    summary["val_accuracy"],
                    summary["val_f1_macro"],
                    summary["best_C"],
                    summary["best_penalty"],
                    summary["best_class_weight"],
                    summary["threshold"],
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
        f"Panel source mode: {'ticker-level + cache' if USE_TICKER_LEVEL_STORAGE else 'single consolidated csv'}",
        f"Panel cache path: {PANEL_CACHE_PATH}",
        f"Smoke test target tickers: {SMOKE_TEST_N_TICKERS}",
        f"Smoke ticker random seed: {SMOKE_SELECTION_SEED}",
        "TimeSeriesSplit train window: expanding (no max_train_size)",
        f"Tau percentile scan: {int(TAU_PERCENTILE_SCAN.min())}-{int(TAU_PERCENTILE_SCAN.max())}",
        f"Tau kappa grid: {', '.join(f'{x:.2f}' for x in VOL_TAU_KAPPA_GRID)}",
        f"Selection metric: {COMBO_WEIGHT_ACCURACY:.3g}*accuracy + {COMBO_WEIGHT_F1:.3g}*f1_macro",
        f"Code snapshot: {out_dir / '_run_snapshot' / Path(__file__).name}",
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
