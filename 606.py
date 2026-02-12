"""
S&P 500 Batch Random Forest Stock Return Prediction
Processes all available tickers and generates comprehensive results
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

class StockReturnPredictor:
    def __init__(self, ticker='NVDA'):
        self.ticker = ticker
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def load_data(self):
        """Load all data sources"""
        # 1. Stock data
        self.stock_data = pd.read_csv(f'stock_data/{self.ticker}.csv')
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], utc=True)
        self.stock_data['Date'] = self.stock_data['Date'].dt.tz_localize(None)
        self.stock_data.set_index('Date', inplace=True)

        # 2. Sentiment data
        try:
            self.sentiment_data = pd.read_csv(f'Data_Sentiment_Index/csv_files/{self.ticker}.csv',
                                             header=None, names=['Date', 'Sentiment'])
            self.sentiment_data['Date'] = pd.to_datetime(self.sentiment_data['Date'])
            self.sentiment_data.set_index('Date', inplace=True)
        except:
            self.sentiment_data = pd.DataFrame()

        # 3. VIX data (shared across all stocks)
        try:
            self.vix_data = pd.read_csv('VIX/vix_data.csv')
            self.vix_data['Date'] = pd.to_datetime(self.vix_data['Date'], format='%d-%m-%Y')
            self.vix_data.set_index('Date', inplace=True)
        except:
            self.vix_data = pd.DataFrame()

        # 4. Macroeconomic data (shared across all stocks)
        try:
            self.gdp_data = pd.read_csv('Macroeconomic_Data/GDP_quarterly.csv')
            self.gdp_data['observation_date'] = pd.to_datetime(self.gdp_data['observation_date'],
                                                               format='%d-%m-%Y')

            self.ipi_data = pd.read_csv('Macroeconomic_Data/IPI_quarterly.csv')
            self.ipi_data['observation_date'] = pd.to_datetime(self.ipi_data['observation_date'])

            self.unemp_data = pd.read_csv('Macroeconomic_Data/Unemployment_Rate_monthly.csv')
            self.unemp_data['observation_date'] = pd.to_datetime(self.unemp_data['observation_date'])
        except:
            self.gdp_data = pd.DataFrame()
            self.ipi_data = pd.DataFrame()
            self.unemp_data = pd.DataFrame()

        # 5. Quarterly financials (optional)
        try:
            self.income_stmt_q = pd.read_csv(f'financials_data/quarterly/{self.ticker}_income_statement_quarterly.csv')
            self.balance_sheet_q = pd.read_csv(f'financials_data/quarterly/{self.ticker}_balance_sheet_quarterly.csv')
            self.cash_flow_q = pd.read_csv(f'financials_data/quarterly/{self.ticker}_cash_flow_quarterly.csv')
        except:
            self.income_stmt_q = pd.DataFrame()
            self.balance_sheet_q = pd.DataFrame()
            self.cash_flow_q = pd.DataFrame()

    def engineer_features(self):
        """Create comprehensive feature set"""
        df = self.stock_data.copy()

        # Target: NEXT day's return
        df['Return'] = df['Close'].pct_change().shift(-1)

        # === PRICE-BASED FEATURES ===
        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

        # Moving averages (using yesterday's close to avoid leakage)
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()

        # Volatility features
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Return'].rolling(window).std()
            df[f'ATR_{window}'] = (df['High'] - df['Low']).rolling(window).mean()

        # Momentum indicators
        df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])

        # Volume features
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_Change'] = df['Volume'].pct_change()

        # === SENTIMENT FEATURES ===
        if not self.sentiment_data.empty:
            df = df.join(self.sentiment_data, how='left')
            df['Sentiment'] = df['Sentiment'].ffill()

            for lag in [1, 2, 3, 5]:
                df[f'Sentiment_Lag{lag}'] = df['Sentiment'].shift(lag)

            df['Sentiment_MA_5'] = df['Sentiment'].rolling(5).mean()
            df['Sentiment_MA_10'] = df['Sentiment'].rolling(10).mean()
            df['Sentiment_Change'] = df['Sentiment'].diff()

        # === VIX FEATURES ===
        if not self.vix_data.empty:
            vix_close = self.vix_data[['Close']].rename(columns={'Close': 'VIX'})
            df = df.join(vix_close, how='left')
            df['VIX'] = df['VIX'].ffill()

            df['VIX_Change'] = df['VIX'].pct_change()
            df['VIX_MA_5'] = df['VIX'].rolling(5).mean()
            df['VIX_MA_20'] = df['VIX'].rolling(20).mean()

        # === MACROECONOMIC FEATURES ===
        if not self.gdp_data.empty:
            gdp_daily = self._expand_to_daily(self.gdp_data, 'observation_date')
            df = df.join(gdp_daily, how='left')

        if not self.ipi_data.empty:
            ipi_daily = self._expand_to_daily(self.ipi_data, 'observation_date')
            df = df.join(ipi_daily, how='left')

        if not self.unemp_data.empty:
            unemp_daily = self._expand_to_daily(self.unemp_data, 'observation_date')
            df = df.join(unemp_daily, how='left')

        # Forward fill all data
        df = df.ffill()

        # === CALENDAR FEATURES ===
        df['Day_of_Week'] = df.index.day_of_week
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        df['Is_Month_End'] = (df.index.day >= 25).astype(int)
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)

        self.data = df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _expand_to_daily(self, df, date_col):
        """Expand quarterly/monthly data to daily frequency"""
        df = df.set_index(date_col)
        df = df.resample('D').ffill()
        return df

    def prepare_train_test(self, test_size=0.2):
        """Prepare training and testing sets"""
        df = self.data.dropna(subset=['Return']).copy()

        # Remove target AND raw price columns to prevent leakage
        feature_cols = [col for col in df.columns if col not in
                       ['Return', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Dividends', 'Stock Splits']]

        X = df[feature_cols]
        y = df['Return']

        # Drop columns with >50% NaN
        nan_pct = X.isna().sum() / len(X) * 100
        cols_to_keep = nan_pct[nan_pct <= 50].index.tolist()
        X = X[cols_to_keep]

        # Fill remaining NaN
        X = X.ffill().bfill()
        if X.isna().any().any():
            X = X.fillna(X.mean())

        # Remove any rows that still have NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient data: only {len(X)} samples")

        # Time-series split
        split_idx = int(len(X) * (1 - test_size))

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        # Scale features
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        self.feature_names = list(X.columns)

    def train_model(self):
        """Train Random Forest model"""
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        self.model.fit(self.X_train_scaled, self.y_train)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def evaluate_model(self):
        """Evaluate model and return metrics"""
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)

        # Calculate metrics
        metrics = {
            'ticker': self.ticker,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'num_features': len(self.feature_names),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_dir_acc': np.mean((y_train_pred > 0) == (self.y_train > 0)),
            'test_dir_acc': np.mean((y_test_pred > 0) == (self.y_test > 0)),
            'top_feature': self.feature_importance.iloc[0]['feature'],
            'top_feature_importance': self.feature_importance.iloc[0]['importance']
        }

        return metrics


def discover_tickers(stock_data_path='stock_data'):
    """Discover all available tickers from CSV files"""
    path = Path(stock_data_path)
    tickers = [f.stem for f in path.glob('*.csv')]
    return sorted(tickers)


def process_single_ticker(ticker, verbose=False):
    """Process a single ticker and return metrics"""
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {ticker}")
            print(f"{'='*60}")

        predictor = StockReturnPredictor(ticker=ticker)

        if verbose:
            print("Loading data...")
        predictor.load_data()

        if verbose:
            print("Engineering features...")
        predictor.engineer_features()

        if verbose:
            print("Preparing train/test split...")
        predictor.prepare_train_test(test_size=0.2)

        if verbose:
            print("Training model...")
        predictor.train_model()

        if verbose:
            print("Evaluating model...")
        metrics = predictor.evaluate_model()

        if verbose:
            print(f"\n✓ {ticker} completed successfully!")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test Dir Acc: {metrics['test_dir_acc']:.2%}")

        return metrics, predictor, None

    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"\n✗ {ticker} failed: {error_msg}")
        return None, None, error_msg


def batch_process_sp500(max_tickers=None, verbose=True, save_models=False):
    """Process all S&P 500 tickers"""

    print("="*80)
    print("S&P 500 RANDOM FOREST BATCH PROCESSING")
    print("="*80)

    # Discover tickers
    print("\nDiscovering available tickers...")
    all_tickers = discover_tickers()

    if max_tickers:
        all_tickers = all_tickers[:max_tickers]

    print(f"Found {len(all_tickers)} tickers to process")

    # Process each ticker
    results = []
    errors = []
    successful_predictors = {}

    for i, ticker in enumerate(all_tickers, 1):
        print(f"\n[{i}/{len(all_tickers)}] Processing {ticker}...", end=' ')

        metrics, predictor, error = process_single_ticker(ticker, verbose=False)

        if metrics:
            results.append(metrics)
            if save_models:
                successful_predictors[ticker] = predictor
            print(f"✓ R²={metrics['test_r2']:.3f}, DirAcc={metrics['test_dir_acc']:.1%}")
        else:
            errors.append({'ticker': ticker, 'error': error})
            print(f"✗ {error}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'sp500_rf_results_{timestamp}.csv', index=False)

    if not errors_df.empty:
        errors_df.to_csv(f'sp500_rf_errors_{timestamp}.csv', index=False)

    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"\nSuccessful: {len(results)} / {len(all_tickers)} tickers")
    print(f"Failed: {len(errors)} tickers")

    # Summary statistics
    if not results_df.empty:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        print(f"\nTest R² Score:")
        print(f"  Mean:   {results_df['test_r2'].mean():.4f}")
        print(f"  Median: {results_df['test_r2'].median():.4f}")
        print(f"  Min:    {results_df['test_r2'].min():.4f}")
        print(f"  Max:    {results_df['test_r2'].max():.4f}")

        print(f"\nTest Directional Accuracy:")
        print(f"  Mean:   {results_df['test_dir_acc'].mean():.2%}")
        print(f"  Median: {results_df['test_dir_acc'].median():.2%}")
        print(f"  Min:    {results_df['test_dir_acc'].min():.2%}")
        print(f"  Max:    {results_df['test_dir_acc'].max():.2%}")

        print(f"\nTest RMSE:")
        print(f"  Mean:   {results_df['test_rmse'].mean():.4f}")
        print(f"  Median: {results_df['test_rmse'].median():.4f}")

        # Top performers
        print("\n" + "="*80)
        print("TOP 10 PERFORMERS (by Test R²)")
        print("="*80)
        top_10 = results_df.nlargest(10, 'test_r2')[['ticker', 'test_r2', 'test_dir_acc', 'test_rmse']]
        print(top_10.to_string(index=False))

        # Top by directional accuracy
        print("\n" + "="*80)
        print("TOP 10 PERFORMERS (by Directional Accuracy)")
        print("="*80)
        top_10_dir = results_df.nlargest(10, 'test_dir_acc')[['ticker', 'test_dir_acc', 'test_r2', 'test_rmse']]
        print(top_10_dir.to_string(index=False))

        # Create visualizations
        create_summary_visualizations(results_df, timestamp)

    return results_df, errors_df


def create_summary_visualizations(results_df, timestamp):
    """Create summary plots for all tickers"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. R² distribution
    axes[0, 0].hist(results_df['test_r2'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results_df['test_r2'].mean(), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {results_df['test_r2'].mean():.3f}")
    axes[0, 0].axvline(0, color='gray', linestyle='-', linewidth=1)
    axes[0, 0].set_xlabel('Test R² Score')
    axes[0, 0].set_ylabel('Number of Tickers')
    axes[0, 0].set_title('Distribution of Test R² Scores Across S&P 500')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Directional Accuracy distribution
    axes[0, 1].hist(results_df['test_dir_acc'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(results_df['test_dir_acc'].mean(), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {results_df['test_dir_acc'].mean():.1%}")
    axes[0, 1].axvline(0.5, color='gray', linestyle='-', linewidth=1, label='Random (50%)')
    axes[0, 1].set_xlabel('Test Directional Accuracy')
    axes[0, 1].set_ylabel('Number of Tickers')
    axes[0, 1].set_title('Distribution of Directional Accuracy Across S&P 500')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. R² vs Directional Accuracy scatter
    axes[1, 0].scatter(results_df['test_r2'], results_df['test_dir_acc'], alpha=0.5, s=20)
    axes[1, 0].axhline(0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    axes[1, 0].axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Test R² Score')
    axes[1, 0].set_ylabel('Test Directional Accuracy')
    axes[1, 0].set_title('R² vs Directional Accuracy')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. RMSE distribution
    axes[1, 1].hist(results_df['test_rmse'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(results_df['test_rmse'].mean(), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {results_df['test_rmse'].mean():.3f}")
    axes[1, 1].set_xlabel('Test RMSE')
    axes[1, 1].set_ylabel('Number of Tickers')
    axes[1, 1].set_title('Distribution of RMSE Across S&P 500')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'sp500_rf_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"\nSummary visualization saved: sp500_rf_summary_{timestamp}.png")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Option 1: Process ALL S&P 500 tickers
    results_df, errors_df = batch_process_sp500(
        max_tickers=None,  # Set to None for all tickers, or a number for testing (e.g., 10)
        verbose=True,
        save_models=False  # Set to True if you want to save all models (uses lots of disk space)
    )

    # Option 2: Process just a sample for testing
    # results_df, errors_df = batch_process_sp500(max_tickers=20, verbose=True)