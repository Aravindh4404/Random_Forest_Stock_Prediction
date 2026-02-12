"""
Multinomial Logit with Lasso Regularization for Daily Stock Return Direction
Classifies next-day return into: Down (-1), Same (0), Up (1)

Reuses data loading and feature engineering from StockReturnPredictor,
extending it with classification-specific logic as per DATA 606 Project Proposal.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ── Reuse the full pipeline from your existing file ──────────────────────────
# Assumes StockReturnPredictor is importable from your existing script.
# If running standalone, paste the StockReturnPredictor class here.
from importlib.util import spec_from_file_location, module_from_spec
import sys

spec = spec_from_file_location("module_606", "randomforest.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
StockReturnPredictor = module.StockReturnPredictor


# =============================================================================
# LABEL CONSTRUCTION
# =============================================================================

def make_direction_labels(returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    """
    Convert continuous next-day returns into a 3-class direction label.

    Classes
    -------
    -1  : Down  — return < -threshold
     0  : Same  — |return| <= threshold  (near-zero / flat day)
     1  : Up    — return >  threshold

    Parameters
    ----------
    returns   : Series of next-day returns (already shifted -1 in feature eng.)
    threshold : Boundary for the 'Same' class.
                0.0  → pure sign split (no Same class; flat days go to Down).
                0.001 → ±0.1% band around zero labelled Same.

    Notes
    -----
    The proposal specifies (up, down, same) as the three classes.
    A threshold of 0.0 effectively collapses to binary (up/down) because
    exact-zero returns are very rare on liquid equities. Using a small
    threshold (e.g. 0.001) creates a meaningful Same class and is
    recommended for multinomial logit.
    """
    labels = pd.Series(index=returns.index, dtype=int)
    labels[returns >  threshold] =  1   # Up
    labels[returns < -threshold] = -1   # Down
    labels[returns.between(-threshold, threshold)] = 0  # Same
    return labels


# =============================================================================
# BINARY SENTIMENT FEATURE
# =============================================================================

def binarize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    As specified in Appendix B of the proposal, create a binary
    transformation of the Bloomberg sentiment index.
    Positive (>=0) → 1, Negative (<0) → 0.
    """
    if 'Sentiment' in df.columns:
        df['Sentiment_Binary'] = (df['Sentiment'] >= 0).astype(int)
    return df


# =============================================================================
# MAIN CLASSIFIER CLASS
# =============================================================================

class MultinomialLogitLasso:
    """
    Multinomial Logistic Regression with L1 (Lasso) regularization.

    Workflow mirrors the Random Forest pipeline so the two models are
    directly comparable on the same train/test/validation split.

    Split (per proposal)
    --------------------
    Training   : first 8 months  (~66% of data)
    Testing    : next  2 months  (~17%)
    Validation : last  2 months  (~17%)
    """

    def __init__(self, ticker: str = 'NVDA', direction_threshold: float = 0.001):
        self.ticker = ticker
        self.threshold = direction_threshold
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.coef_df = None

    # ------------------------------------------------------------------ #
    #  STEP 1 — Inherit data + feature engineering from RF pipeline       #
    # ------------------------------------------------------------------ #

    def load_and_engineer(self):
        """
        Re-use the existing StockReturnPredictor to load data and build
        the full feature matrix, then layer on classification-specific
        features (binary sentiment) and the direction label.
        """
        print("=" * 60)
        print(f"MULTINOMIAL LOGIT — LASSO  |  Ticker: {self.ticker}")
        print("=" * 60)

        # ── Reuse RF pipeline for data loading & feature engineering ──
        rf = StockReturnPredictor(ticker=self.ticker)
        rf.load_data()
        rf.engineer_features()

        df = rf.data.copy()

        # ── Add binary sentiment (Appendix B) ─────────────────────────
        df = binarize_sentiment(df)

        # ── Build direction labels ─────────────────────────────────────
        df['Direction'] = make_direction_labels(df['Return'], self.threshold)

        self.raw_data = df
        self.rf_pipeline = rf          # keep reference for feature name lists
        print(f"\nClass distribution:\n{df['Direction'].value_counts().sort_index()}")
        print(f"  -1 (Down): {(df['Direction']==-1).sum()} | "
              f"0 (Same): {(df['Direction']==0).sum()} | "
              f"1 (Up): {(df['Direction']==1).sum()}")

    # ------------------------------------------------------------------ #
    #  STEP 2 — Train / Test / Validation split                           #
    # ------------------------------------------------------------------ #

    def prepare_splits(self):
        """
        Chronological 3-way split: 8 months train, 2 months test,
        2 months validation — matching the proposal's methodology.
        """
        print("\nPreparing train / test / validation split...")

        df = self.raw_data.dropna(subset=['Direction']).copy()

        # ── Feature columns (same exclusion logic as RF, minus leakage) ──
        exclude = [
            'Return', 'Direction',
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Dividends', 'Stock Splits',
            'Close_to_High', 'Close_to_Low', 'Daily_Range',
            'Price_to_SMA_5', 'Price_to_SMA_10',
            'Price_to_SMA_20', 'Price_to_SMA_50',
        ]
        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols].copy()
        y = df['Direction'].copy()

        # Drop columns with >50% NaN (mirrors RF pipeline)
        nan_pct = X.isna().sum() / len(X) * 100
        X = X[nan_pct[nan_pct <= 50].index]

        # Fill remaining NaN
        X = X.ffill().bfill().fillna(X.mean())

        # Final mask
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]

        print(f"Usable samples: {len(X)}")

        # ── Compute split indices by calendar months ───────────────────
        n = len(X)
        total_months = 12          # ~12 months of daily data
        train_months = 8
        test_months  = 2
        # validation = remaining months

        train_end = int(n * train_months / total_months)
        test_end  = int(n * (train_months + test_months) / total_months)

        self.X_train = X.iloc[:train_end]
        self.y_train = y.iloc[:train_end]

        self.X_test  = X.iloc[train_end:test_end]
        self.y_test  = y.iloc[train_end:test_end]

        self.X_val   = X.iloc[test_end:]
        self.y_val   = y.iloc[test_end:]

        self.feature_names = list(X.columns)

        # ── Scale ──────────────────────────────────────────────────────
        self.X_train_sc = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names, index=self.X_train.index
        )
        self.X_test_sc = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names, index=self.X_test.index
        )
        self.X_val_sc = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_names, index=self.X_val.index
        )

        print(f"  Training   : {len(self.X_train)} samples "
              f"({self.X_train.index[0].date()} → {self.X_train.index[-1].date()})")
        print(f"  Testing    : {len(self.X_test)} samples "
              f"({self.X_test.index[0].date()} → {self.X_test.index[-1].date()})")
        print(f"  Validation : {len(self.X_val)} samples "
              f"({self.X_val.index[0].date()} → {self.X_val.index[-1].date()})")

    # ------------------------------------------------------------------ #
    #  STEP 3 — Train with Lasso (L1) via cross-validated C search        #
    # ------------------------------------------------------------------ #

    def train(self, cv_folds: int = 5):
        """
        Fit Multinomial Logistic Regression with L1 penalty.

        Uses LogisticRegressionCV to select the regularization strength C
        (inverse of lambda) via TimeSeriesSplit cross-validation on the
        training set — keeping the time-series structure intact.

        Key parameters
        --------------
        penalty  = 'l1'        : Lasso — drives irrelevant coefficients to zero,
                                 directly identifying key drivers (proposal goal)
        solver   = 'saga'      : The only solver supporting L1 + multinomial
        multi_class = 'multinomial' : True softmax (vs one-vs-rest)
        Cs       = log-spaced grid  : Cross-validated over 20 values of C
        """
        print("\nFitting Multinomial Logit with Lasso (L1) regularization...")
        print(f"  Cross-validation: TimeSeriesSplit with {cv_folds} folds")

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # LogisticRegressionCV searches over Cs automatically
        self.model = LogisticRegressionCV(
            Cs=np.logspace(-1, 4, 10),   # Reduced to 10 C values for efficiency
            penalty='l1',
            solver='saga',
            cv=tscv,
            scoring='accuracy',
            max_iter=5000,               # Reduced max_iter (usually sufficient)
            tol=1e-3,                    # Relaxed tolerance for faster convergence
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(self.X_train_sc, self.y_train)

        best_C = self.model.C_[0]
        print(f"  Best C found: {best_C:.6f}  "
              f"(lambda = {1/best_C:.4f}  — higher lambda = more regularization)")

        # ── Build coefficient table ────────────────────────────────────
        class_labels = self.model.classes_          # e.g. [-1, 0, 1]
        class_names  = {-1: 'Down', 0: 'Same', 1: 'Up'}

        coef_data = {}
        for i, cls in enumerate(class_labels):
            coef_data[class_names[cls]] = self.model.coef_[i]

        self.coef_df = pd.DataFrame(
            coef_data,
            index=self.feature_names
        )

        n_nonzero = (self.coef_df.abs() > 1e-6).any(axis=1).sum()
        print(f"  Non-zero features after Lasso: {n_nonzero} / {len(self.feature_names)}")

    # ------------------------------------------------------------------ #
    #  STEP 4 — Evaluate on test and validation sets                      #
    # ------------------------------------------------------------------ #

    def evaluate(self):
        """
        Report accuracy and F1-score on training, test, and validation
        sets — matching the proposal's evaluation criteria.
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        results = {}
        splits = [
            ('Training',   self.X_train_sc, self.y_train),
            ('Testing',    self.X_test_sc,  self.y_test),
            ('Validation', self.X_val_sc,   self.y_val),
        ]

        for split_name, X_sc, y_true in splits:
            y_pred = self.model.predict(X_sc)

            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro',   zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            results[split_name] = {
                'accuracy':    acc,
                'f1_macro':    f1_macro,
                'f1_weighted': f1_weighted,
            }

            print(f"\n{split_name.upper()} SET:")
            print(f"  Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
            print(f"  F1 (macro)  : {f1_macro:.4f}")
            print(f"  F1 (weighted): {f1_weighted:.4f}")
            print(f"\n  Classification Report:\n")
            print(classification_report(
                y_true, y_pred,
                target_names=['Down (-1)', 'Same (0)', 'Up (1)'],
                zero_division=0
            ))

        self.eval_results = results

        # ── Plots ──────────────────────────────────────────────────────
        self._plot_confusion_matrices()
        self._plot_lasso_coefficients()
        self._plot_metrics_summary(results)

        return results

    # ------------------------------------------------------------------ #
    #  STEP 5 — Interpret Lasso coefficients (driver analysis)            #
    # ------------------------------------------------------------------ #

    def print_drivers(self, top_n: int = 20):
        """
        Print the most important features identified by Lasso.
        Features with coefficient = 0 were eliminated by regularization.
        This answers the proposal's goal: 'identify the drivers of equity
        market price tendency'.
        """
        print("\n" + "=" * 60)
        print(f"LASSO FEATURE SELECTION — TOP {top_n} DRIVERS")
        print("=" * 60)

        # Max absolute coefficient across all classes per feature
        self.coef_df['Max_Abs_Coef'] = self.coef_df[['Down', 'Same', 'Up']].abs().max(axis=1)
        top = self.coef_df.sort_values('Max_Abs_Coef', ascending=False).head(top_n)

        print(f"\n{'Feature':<35} {'Down':>10} {'Same':>10} {'Up':>10} {'MaxAbs':>10}")
        print("-" * 75)
        for feat, row in top.iterrows():
            print(f"{feat:<35} {row['Down']:>10.4f} {row['Same']:>10.4f} "
                  f"{row['Up']:>10.4f} {row['Max_Abs_Coef']:>10.4f}")

        # Features zeroed out
        zeroed = (self.coef_df['Max_Abs_Coef'] < 1e-6).sum()
        print(f"\nFeatures eliminated by Lasso (coef ≈ 0): {zeroed}")

    # ------------------------------------------------------------------ #
    #  PLOTS                                                               #
    # ------------------------------------------------------------------ #

    def _plot_confusion_matrices(self):
        """Side-by-side confusion matrices for test and validation sets."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['Down\n(-1)', 'Same\n(0)', 'Up\n(1)']

        for ax, (name, X_sc, y_true) in zip(
            axes,
            [('Test', self.X_test_sc, self.y_test),
             ('Validation', self.X_val_sc, self.y_val)]
        ):
            y_pred = self.model.predict(X_sc)
            cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=labels)
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(f'Confusion Matrix — {name} Set', fontsize=13, fontweight='bold')

        plt.suptitle(f'{self.ticker} | Multinomial Logit (Lasso)',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        fname = f'{self.ticker}_logit_confusion.png'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"\nConfusion matrix plot saved: '{fname}'")
        plt.show()

    def _plot_lasso_coefficients(self, top_n: int = 20):
        """
        Grouped bar chart of Lasso coefficients for the top features,
        one bar per class (Down / Same / Up).
        """
        if self.coef_df is None:
            return

        plot_df = self.coef_df[['Down', 'Same', 'Up']].copy()
        plot_df['Max_Abs_Coef'] = plot_df.abs().max(axis=1)
        top = plot_df.sort_values('Max_Abs_Coef', ascending=False).head(top_n)

        x = np.arange(len(top))
        width = 0.28
        colors = ['#d9534f', '#f0ad4e', '#5cb85c']

        fig, ax = plt.subplots(figsize=(16, 6))
        for i, (cls, color) in enumerate(zip(['Down', 'Same', 'Up'], colors)):
            ax.bar(x + i * width, top[cls], width, label=cls, color=color, alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(top.index, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('Lasso Coefficient')
        ax.set_title(f'{self.ticker} | Top {top_n} Lasso Coefficients by Direction Class',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Class')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fname = f'{self.ticker}_logit_lasso_coefs.png'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"Lasso coefficient plot saved: '{fname}'")
        plt.show()

    def _plot_metrics_summary(self, results: dict):
        """Bar chart comparing accuracy and F1 across splits."""
        splits   = list(results.keys())
        accuracy = [results[s]['accuracy']    for s in splits]
        f1_macro = [results[s]['f1_macro']    for s in splits]
        f1_wt    = [results[s]['f1_weighted'] for s in splits]

        x     = np.arange(len(splits))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, accuracy, width, label='Accuracy',     color='steelblue',  alpha=0.85)
        ax.bar(x,         f1_macro, width, label='F1 (macro)',   color='darkorange',  alpha=0.85)
        ax.bar(x + width, f1_wt,    width, label='F1 (weighted)',color='seagreen',    alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'{self.ticker} | Multinomial Logit (Lasso) — Performance Summary',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fname = f'{self.ticker}_logit_metrics.png'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"Metrics summary plot saved: '{fname}'")
        plt.show()

    # ------------------------------------------------------------------ #
    #  Predict direction for next trading day                              #
    # ------------------------------------------------------------------ #

    def predict_tomorrow(self):
        """
        Predict the direction of tomorrow's return using the most recent
        available data point (mirrors predict_tomorrow in the RF class).
        """
        print("\n" + "=" * 60)
        print("NEXT-DAY DIRECTION PREDICTION")
        print("=" * 60)

        latest = self.raw_data.iloc[[-2]][self.feature_names].copy()
        latest = latest.ffill().bfill()

        latest_sc = self.scaler.transform(latest)

        direction  = self.model.predict(latest_sc)[0]
        proba      = self.model.predict_proba(latest_sc)[0]
        class_map  = {-1: 'Down ↓', 0: 'Same →', 1: 'Up ↑'}
        class_idx  = list(self.model.classes_)

        print(f"\nDate used as input : {self.raw_data.index[-2].strftime('%Y-%m-%d')}")
        print(f"Predicted direction: {class_map[direction]}")
        print(f"\nClass probabilities:")
        for cls, p in zip(class_idx, proba):
            print(f"  {class_map[cls]:10s}: {p*100:.2f}%")

        return direction, dict(zip(class_idx, proba))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # ── Configuration ────────────────────────────────────────────────────────
    TICKER    = 'NVDA'
    THRESHOLD = 0.001   # ±0.1% band → "Same"; set to 0.0 for pure up/down split

    # ── Initialize and run ───────────────────────────────────────────────────
    clf = MultinomialLogitLasso(ticker=TICKER, direction_threshold=THRESHOLD)

    # 1. Load data + engineer features (reuses RF pipeline)
    clf.load_and_engineer()

    # 2. Prepare train / test / validation splits
    clf.prepare_splits()

    # 3. Fit Multinomial Logit with Lasso (L1) regularization
    clf.train(cv_folds=5)

    # 4. Evaluate: accuracy + F1 on all three splits
    metrics = clf.evaluate()

    # 5. Print Lasso-selected drivers (interpretability)
    clf.print_drivers(top_n=20)

    # 6. Predict tomorrow's direction
    clf.predict_tomorrow()

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)