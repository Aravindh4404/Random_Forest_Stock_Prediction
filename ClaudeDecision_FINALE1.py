"""
S&P 500 Batch XGBoost + Decision Tree Stock Return Prediction
Using ONLY Pre-Generated Features from feature_datasets/

Features used:
- Return lags (1, 2, 3, 5, 10, 20)
- Sentiment lags (1, 2, 3, 5, 10, 20)
- VIX lags (1, 2, 3, 5, 10, 20)
- Financial ratios (9 ratios - aligned to release dates)
- Macro indicators (GDP, IPI, Unemployment - aligned to release dates)

Target: Next-day return
Model: Blended ensemble - XGBoost (70%) + Decision Tree (30%)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

XGB_BLEND_WEIGHT = 0.7  # 70% XGBoost, 30% Decision Tree
FEATURE_DATASET_DIR = 'feature_datasets'
TEST_SIZE = 0.2  # 20% of data for testing

# Exact features to use (in order)
REQUIRED_FEATURES = [
    # Return lags
    'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag5', 'Return_Lag10', 'Return_Lag20',

    # Sentiment lags
    'Sentiment_Lag1', 'Sentiment_Lag2', 'Sentiment_Lag3', 'Sentiment_Lag5', 'Sentiment_Lag10', 'Sentiment_Lag20',

    # VIX lags
    'VIX_Lag1', 'VIX_Lag2', 'VIX_Lag3', 'VIX_Lag5', 'VIX_Lag10', 'VIX_Lag20',

    # Financial ratios (release-date aligned)
    'roe', 'roa', 'op_margin', 'debt_to_equity', 'liquidity_ratio',
    'current_ratio', 'free_cf_margin', 'revenue_growth', 'ocf_to_assets',

    # GDP features (release-date aligned)
    'GDP_GDP_SA_PC_QOQ', 'GDP_GDP_SA_PC_YOY', 'GDP_GDP_NSA_PC_QOQ', 'GDP_GDP_NSA_PC_YOY',

    # IPI features (release-date aligned)
    'IPI_IPI', 'IPI_IPI_YOY', 'IPI_IPI_QOQ', 'IPI_IPI_SA', 'IPI_IPI_SA_YOY', 'IPI_IPI_SA_QOQ',

    # Unemployment features (release-date aligned)
    'UNEMP_UNRATE', 'UNEMP_UNRATE_PC1', 'UNEMP_UNRATE_PCH',
    'UNEMP_UNRATENSA', 'UNEMP_UNRATENSA_PC1', 'UNEMP_UNRATENSA_PCH'
]


# ══════════════════════════════════════════════════════════════════════════════
# STOCK RETURN PREDICTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class StockReturnPredictor:
    """
    Predicts next-day stock returns using pre-generated feature datasets.
    Uses only the specific features listed in REQUIRED_FEATURES.
    """

    def __init__(self, ticker='AAPL'):
        self.ticker = ticker
        self.data = None
        self.xgb_model = None
        self.dt_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_sc = None
        self.X_test_sc = None

    def load_data(self):
        """Load pre-generated feature dataset for ticker"""
        feature_path = Path(FEATURE_DATASET_DIR) / f'{self.ticker}_features.csv'

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        # Load the feature dataset
        df = pd.read_csv(feature_path, index_col=0, parse_dates=True)

        # Verify Return column exists (target)
        if 'Return' not in df.columns:
            raise ValueError(f"'Return' column not found in {feature_path}")

        self.data = df
        print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} total columns")

    def prepare_features(self):
        """
        Extract only the required features and create target.
        Target: Next-day return (already in the dataset as 'Return')
        """
        df = self.data.copy()

        # Target is current 'Return' column (next-day return)
        # We need to shift it so we're predicting the next day
        # Actually, Return is already pct_change() which is today's return
        # We want to predict tomorrow's return, so shift Return backward by -1
        df['Target'] = df['Return'].shift(-1)

        # Identify available features
        available_features = [f for f in REQUIRED_FEATURES if f in df.columns]
        missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]

        if missing_features:
            print(f"   ⚠ Missing {len(missing_features)} features: {missing_features[:5]}...")

        print(f"   ✓ Using {len(available_features)}/{len(REQUIRED_FEATURES)} features")

        # Extract features and target
        X = df[available_features].copy()
        y = df['Target'].copy()

        # Remove rows with NaN in target or too many NaN features
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Fill remaining NaNs in features
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Final check
        if len(X) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(X)} rows")

        self.feature_data = X
        self.target_data = y
        self.feature_names = list(X.columns)

        print(f"   ✓ Final dataset: {len(X)} samples, {len(self.feature_names)} features")

    def train_test_split(self, test_size=TEST_SIZE):
        """Split data chronologically into train and test sets"""
        X = self.feature_data
        y = self.target_data

        # Chronological split
        split_idx = int(len(X) * (1 - test_size))

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        # Standardize features
        self.X_train_sc = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )

        self.X_test_sc = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )

        print(f"   ✓ Train: {len(self.X_train)} samples, Test: {len(self.X_test)} samples")

    def train_models(self):
        """Train XGBoost and Decision Tree models"""

        # XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        self.xgb_model.fit(
            self.X_train_sc,
            self.y_train,
            eval_set=[(self.X_test_sc, self.y_test)],
            verbose=False
        )

        # Decision Tree model
        self.dt_model = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )

        self.dt_model.fit(self.X_train_sc, self.y_train)

        # Compute blended feature importance
        xgb_imp = self.xgb_model.feature_importances_
        dt_imp = self.dt_model.feature_importances_

        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': XGB_BLEND_WEIGHT * xgb_imp + (1 - XGB_BLEND_WEIGHT) * dt_imp,
            'xgb_importance': xgb_imp,
            'dt_importance': dt_imp
        }).sort_values('importance', ascending=False)

        print(f"   ✓ Models trained")

    def predict(self, X):
        """Make blended predictions"""
        xgb_pred = self.xgb_model.predict(X)
        dt_pred = self.dt_model.predict(X)
        return XGB_BLEND_WEIGHT * xgb_pred + (1 - XGB_BLEND_WEIGHT) * dt_pred

    def evaluate(self):
        """Evaluate model performance"""
        # Train predictions
        y_train_pred = self.predict(self.X_train_sc)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_dir_acc = np.mean((y_train_pred > 0) == (self.y_train > 0))

        # Test predictions
        y_test_pred = self.predict(self.X_test_sc)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_dir_acc = np.mean((y_test_pred > 0) == (self.y_test > 0))

        # Individual model directional accuracy
        xgb_test_pred = self.xgb_model.predict(self.X_test_sc)
        dt_test_pred = self.dt_model.predict(self.X_test_sc)
        xgb_dir_acc = np.mean((xgb_test_pred > 0) == (self.y_test > 0))
        dt_dir_acc = np.mean((dt_test_pred > 0) == (self.y_test > 0))

        results = {
            'ticker': self.ticker,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'num_features': len(self.feature_names),
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_dir_acc': train_dir_acc,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_dir_acc': test_dir_acc,
            'xgb_dir_acc': xgb_dir_acc,
            'dt_dir_acc': dt_dir_acc,
            'blend_weight': XGB_BLEND_WEIGHT,
            'top_feature': self.feature_importance.iloc[0]['feature'],
            'top_feature_importance': self.feature_importance.iloc[0]['importance']
        }

        return results

    def run_full_pipeline(self):
        """Execute complete pipeline: load -> prepare -> split -> train -> evaluate"""
        self.load_data()
        self.prepare_features()
        self.train_test_split()
        self.train_models()
        results = self.evaluate()
        return results


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def discover_feature_files(feature_dir=FEATURE_DATASET_DIR):
    """Find all ticker feature CSV files"""
    path = Path(feature_dir)
    if not path.exists():
        print(f"❌ Feature dataset directory not found: {feature_dir}")
        return []

    # Find all *_features.csv files
    feature_files = sorted(path.glob('*_features.csv'))

    # Exclude summary/error files
    feature_files = [f for f in feature_files if not f.name.startswith('_')]

    # Extract tickers
    tickers = [f.stem.replace('_features', '') for f in feature_files]

    return tickers


def process_single_ticker(ticker, verbose=False):
    """Process a single ticker through the full pipeline"""
    try:
        predictor = StockReturnPredictor(ticker=ticker)
        results = predictor.run_full_pipeline()

        if verbose:
            print(f"   ✓ R²={results['test_r2']:.4f}, DirAcc={results['test_dir_acc']:.2%}")

        return results, predictor, None

    except Exception as e:
        if verbose:
            print(f"   ✗ Error: {str(e)[:60]}")
        return None, None, str(e)


def batch_process_all_stocks(max_tickers=None, verbose=True, save_models=False):
    """
    Process all stocks in feature_datasets/ directory.

    Parameters:
    -----------
    max_tickers : int, optional
        Limit number of tickers to process
    verbose : bool
        Print detailed progress
    save_models : bool
        Keep trained models in memory (uses more RAM)

    Returns:
    --------
    results_df : DataFrame
        Results for all successful tickers
    errors_df : DataFrame
        Errors for failed tickers
    """
    print("=" * 80)
    print("S&P 500 XGBoost + Decision Tree — Using Pre-Generated Features")
    print("=" * 80)
    print(f"Blend: {XGB_BLEND_WEIGHT:.0%} XGBoost + {1 - XGB_BLEND_WEIGHT:.0%} Decision Tree")
    print(f"Features: {len(REQUIRED_FEATURES)} features")
    print(f"Feature categories: Return lags, Sentiment lags, VIX lags, Financials, Macro")
    print("=" * 80)

    # Discover tickers
    tickers = discover_feature_files()

    if not tickers:
        print(f"\n❌ No feature files found in {FEATURE_DATASET_DIR}/")
        print(f"   Please run batch_feature_generator_integrated.py first")
        return pd.DataFrame(), pd.DataFrame()

    if max_tickers:
        tickers = tickers[:max_tickers]

    print(f"\n→ Found {len(tickers)} tickers to process\n")

    results = []
    errors = []
    predictors = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>4}/{len(tickers)}] {ticker:<6}", end='')

        result, predictor, error = process_single_ticker(ticker, verbose=False)

        if result:
            results.append(result)
            if save_models:
                predictors[ticker] = predictor

            print(f"  ✓  R²={result['test_r2']:>7.4f}  "
                  f"Blend={result['test_dir_acc']:>5.1%}  "
                  f"XGB={result['xgb_dir_acc']:>5.1%}  "
                  f"DT={result['dt_dir_acc']:>5.1%}  "
                  f"Top: {result['top_feature']}")
        else:
            errors.append({'ticker': ticker, 'error': error})
            print(f"  ✗  {error[:60]}")

    # Create DataFrames
    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not results_df.empty:
        results_path = f'sp500_xgb_dt_results_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n→ Results saved to: {results_path}")

    if not errors_df.empty:
        errors_path = f'sp500_xgb_dt_errors_{timestamp}.csv'
        errors_df.to_csv(errors_path, index=False)
        print(f"→ Errors saved to: {errors_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"COMPLETED: {len(results)} successful, {len(errors)} failed")
    print("=" * 80)

    if not results_df.empty:
        print("\n📊 SUMMARY STATISTICS:")
        print("-" * 80)

        metrics = [
            ('test_r2', 'Test R²', '{:.4f}'),
            ('test_dir_acc', 'Directional Accuracy', '{:.2%}'),
            ('test_rmse', 'Test RMSE', '{:.6f}'),
            ('test_mae', 'Test MAE', '{:.6f}'),
        ]

        for col, label, fmt in metrics:
            print(f"\n{label}:")
            print(f"  Mean   : {fmt.format(results_df[col].mean())}")
            print(f"  Median : {fmt.format(results_df[col].median())}")
            print(f"  Std Dev: {fmt.format(results_df[col].std())}")
            print(f"  Min    : {fmt.format(results_df[col].min())}")
            print(f"  Max    : {fmt.format(results_df[col].max())}")

        # Top performers
        if len(results_df) >= 10:
            print("\n" + "=" * 80)
            print("🏆 TOP 10 PERFORMERS BY R²:")
            print("-" * 80)
            top_r2 = results_df.nlargest(10, 'test_r2')[
                ['ticker', 'test_r2', 'test_dir_acc', 'test_rmse', 'top_feature']
            ]
            print(top_r2.to_string(index=False))

            print("\n" + "=" * 80)
            print("🎯 TOP 10 PERFORMERS BY DIRECTIONAL ACCURACY:")
            print("-" * 80)
            top_acc = results_df.nlargest(10, 'test_dir_acc')[
                ['ticker', 'test_dir_acc', 'test_r2', 'xgb_dir_acc', 'dt_dir_acc']
            ]
            print(top_acc.to_string(index=False))

        # Feature importance summary
        print("\n" + "=" * 80)
        print("🔍 FEATURE IMPORTANCE SUMMARY:")
        print("-" * 80)
        top_features = results_df['top_feature'].value_counts().head(10)
        print("\nMost frequently top-ranked features:")
        for feat, count in top_features.items():
            print(f"  {feat:<30} : {count:>3} tickers")

    if save_models:
        return results_df, errors_df, predictors
    else:
        return results_df, errors_df


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TICKER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_single_ticker(ticker='AAPL', show_plots=True):
    """
    Detailed analysis for a single ticker with visualization.

    Parameters:
    -----------
    ticker : str
        Ticker symbol to analyze
    show_plots : bool
        Whether to generate plots

    Returns:
    --------
    predictor : StockReturnPredictor
        Trained predictor object
    results : dict
        Performance metrics
    """
    print("=" * 80)
    print(f"DETAILED ANALYSIS: {ticker}")
    print("=" * 80)

    predictor = StockReturnPredictor(ticker=ticker)
    results = predictor.run_full_pipeline()

    # Print results
    print("\n📈 PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"Train samples: {results['train_samples']}")
    print(f"Test samples : {results['test_samples']}")
    print(f"Features used: {results['num_features']}")
    print()
    print(f"Train R²               : {results['train_r2']:.4f}")
    print(f"Test R²                : {results['test_r2']:.4f}")
    print(f"Train Directional Acc  : {results['train_dir_acc']:.2%}")
    print(f"Test Directional Acc   : {results['test_dir_acc']:.2%}")
    print(f"  - XGBoost only       : {results['xgb_dir_acc']:.2%}")
    print(f"  - Decision Tree only : {results['dt_dir_acc']:.2%}")
    print()
    print(f"Train RMSE : {results['train_rmse']:.6f}")
    print(f"Test RMSE  : {results['test_rmse']:.6f}")
    print(f"Train MAE  : {results['train_mae']:.6f}")
    print(f"Test MAE   : {results['test_mae']:.6f}")

    # Feature importance
    print("\n🔍 TOP 20 FEATURES BY IMPORTANCE:")
    print("-" * 80)
    print(predictor.feature_importance.head(20).to_string(index=False))

    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{ticker} - Model Performance Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Actual vs Predicted (Test Set)
        ax1 = axes[0, 0]
        y_test_pred = predictor.predict(predictor.X_test_sc)
        ax1.scatter(predictor.y_test, y_test_pred, alpha=0.5, s=20)
        ax1.plot([predictor.y_test.min(), predictor.y_test.max()],
                 [predictor.y_test.min(), predictor.y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Return')
        ax1.set_ylabel('Predicted Return')
        ax1.set_title(f'Actual vs Predicted (Test Set)\nR² = {results["test_r2"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Prediction errors over time
        ax2 = axes[0, 1]
        errors = predictor.y_test - y_test_pred
        ax2.plot(predictor.y_test.index, errors, alpha=0.7, linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.fill_between(predictor.y_test.index, errors, 0, alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Prediction Errors Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Feature importance (top 15)
        ax3 = axes[1, 0]
        top_features = predictor.feature_importance.head(15)
        ax3.barh(range(len(top_features)), top_features['importance'])
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 15 Feature Importance')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')

        # Plot 4: Cumulative returns
        ax4 = axes[1, 1]

        # Calculate cumulative returns
        actual_cumret = (1 + predictor.y_test).cumprod()

        # Trading strategy: buy when predicted return > 0
        strategy_returns = predictor.y_test.copy()
        strategy_returns[y_test_pred <= 0] = 0  # Stay out when predicting down
        strategy_cumret = (1 + strategy_returns).cumprod()

        ax4.plot(actual_cumret.index, actual_cumret, label='Buy & Hold', linewidth=2)
        ax4.plot(strategy_cumret.index, strategy_cumret, label='Model Strategy', linewidth=2)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return')
        ax4.set_title('Cumulative Returns: Buy & Hold vs Model Strategy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save plot
        plot_path = f'{ticker}_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {plot_path}")
        plt.show()

    return predictor, results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Choose what to run:

    # Option 1: Batch process all stocks
    results_df, errors_df = batch_process_all_stocks(
        max_tickers=None,  # Set to a number to limit (e.g., 10 for testing)
        verbose=True,
        save_models=False
    )

    # Option 2: Detailed analysis of a single stock (uncomment to use)
    # predictor, results = analyze_single_ticker(
    #     ticker='AAPL',
    #     show_plots=True
    # )