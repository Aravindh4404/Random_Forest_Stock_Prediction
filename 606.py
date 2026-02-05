"""
Random Forest Regression for Daily Stock Return Prediction
Incorporates: Stock prices, Sentiment, Financials, and Macroeconomic data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
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
        print("Loading data...")

        # 1. Stock data
        self.stock_data = pd.read_csv(f'stock_data/{self.ticker}.csv')
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], utc=True)
        # Convert to timezone-naive (remove timezone info)
        self.stock_data['Date'] = self.stock_data['Date'].dt.tz_localize(None)
        self.stock_data.set_index('Date', inplace=True)

        # 2. Sentiment data
        self.sentiment_data = pd.read_csv(f'Data_Sentiment_Index/csv_files/{self.ticker}.csv',
                                         header=None, names=['Date', 'Sentiment'])
        self.sentiment_data['Date'] = pd.to_datetime(self.sentiment_data['Date'])
        self.sentiment_data.set_index('Date', inplace=True)

        # 3. VIX data
        self.vix_data = pd.read_csv('VIX/vix_data.csv')
        self.vix_data['Date'] = pd.to_datetime(self.vix_data['Date'], format='%d-%m-%Y')
        self.vix_data.set_index('Date', inplace=True)

        # 4. Macroeconomic data
        self.gdp_data = pd.read_csv('Macroeconomic_Data/GDP_quarterly.csv')
        self.gdp_data['observation_date'] = pd.to_datetime(self.gdp_data['observation_date'],
                                                           format='%d-%m-%Y')

        self.ipi_data = pd.read_csv('Macroeconomic_Data/IPI_quarterly.csv')
        self.ipi_data['observation_date'] = pd.to_datetime(self.ipi_data['observation_date'])

        self.unemp_data = pd.read_csv('Macroeconomic_Data/Unemployment_Rate_monthly.csv')
        self.unemp_data['observation_date'] = pd.to_datetime(self.unemp_data['observation_date'])

        # 5. Quarterly financials
        self.income_stmt_q = pd.read_csv(f'financials_data/quarterly/{self.ticker}_income_statement_quarterly.csv')
        self.balance_sheet_q = pd.read_csv(f'financials_data/quarterly/{self.ticker}_balance_sheet_quarterly.csv')
        self.cash_flow_q = pd.read_csv(f'financials_data/quarterly/{self.ticker}_cash_flow_quarterly.csv')

        print("Data loaded successfully!")

    def engineer_features(self):
        """Create comprehensive feature set"""
        print("Engineering features...")

        # Start with stock data
        df = self.stock_data.copy()

        # Target: NEXT day's return (what we want to predict)
        # This ensures we use today's features to predict tomorrow's return
        df['Return'] = df['Close'].pct_change().shift(-1)  # Shift -1 to get tomorrow's return

        # === PRICE-BASED FEATURES ===
        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}']

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

        # High-Low range
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Close_to_High'] = (df['High'] - df['Close']) / df['Close']
        df['Close_to_Low'] = (df['Close'] - df['Low']) / df['Close']

        # === SENTIMENT FEATURES ===
        df = df.join(self.sentiment_data, how='left')
        df['Sentiment'] = df['Sentiment'].ffill()

        # Lagged sentiment
        for lag in [1, 2, 3, 5]:
            df[f'Sentiment_Lag{lag}'] = df['Sentiment'].shift(lag)

        # Sentiment momentum
        df['Sentiment_MA_5'] = df['Sentiment'].rolling(5).mean()
        df['Sentiment_MA_10'] = df['Sentiment'].rolling(10).mean()
        df['Sentiment_Change'] = df['Sentiment'].diff()

        # === VIX FEATURES ===
        vix_close = self.vix_data[['Close']].rename(columns={'Close': 'VIX'})
        df = df.join(vix_close, how='left')
        df['VIX'] = df['VIX'].ffill()

        # VIX features
        df['VIX_Change'] = df['VIX'].pct_change()
        df['VIX_MA_5'] = df['VIX'].rolling(5).mean()
        df['VIX_MA_20'] = df['VIX'].rolling(20).mean()

        # === MACROECONOMIC FEATURES ===
        # Forward-fill quarterly/monthly data to daily
        gdp_daily = self._expand_to_daily(self.gdp_data, 'observation_date')
        ipi_daily = self._expand_to_daily(self.ipi_data, 'observation_date')
        unemp_daily = self._expand_to_daily(self.unemp_data, 'observation_date')

        df = df.join(gdp_daily, how='left')
        df = df.join(ipi_daily, how='left')
        df = df.join(unemp_daily, how='left')

        # Forward fill macro data
        macro_cols = list(gdp_daily.columns) + list(ipi_daily.columns) + list(unemp_daily.columns)
        df[macro_cols] = df[macro_cols].ffill()

        # === FINANCIAL STATEMENT FEATURES ===
        financial_features = self._extract_financial_features()
        df = df.join(financial_features, how='left')
        df = df.ffill()  # Forward fill quarterly financials

        # === CALENDAR FEATURES ===
        df['Day_of_Week'] = df.index.day_of_week
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        df['Is_Month_End'] = (df.index.day >= 25).astype(int)
        df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)

        self.data = df
        print(f"Feature engineering complete! Shape: {df.shape}")

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

    def _extract_financial_features(self):
        """Extract key financial ratios and metrics"""
        # Parse quarterly financial statements (dates as columns)
        income_dates = [col for col in self.income_stmt_q.columns if col != 'Unnamed: 0']
        balance_dates = [col for col in self.balance_sheet_q.columns if col != 'Unnamed: 0']
        cf_dates = [col for col in self.cash_flow_q.columns if col != 'Unnamed: 0']

        # Create time series of key metrics
        financial_ts = []

        for date_str in income_dates:
            try:
                date = pd.to_datetime(date_str)

                # Income statement metrics
                revenue = self._get_metric(self.income_stmt_q, 'Total Revenue', date_str)
                net_income = self._get_metric(self.income_stmt_q, 'Net Income', date_str)
                gross_profit = self._get_metric(self.income_stmt_q, 'Gross Profit', date_str)
                operating_income = self._get_metric(self.income_stmt_q, 'Operating Income', date_str)
                ebitda = self._get_metric(self.income_stmt_q, 'EBITDA', date_str)

                # Balance sheet metrics
                total_assets = self._get_metric(self.balance_sheet_q, 'Total Assets', date_str)
                total_debt = self._get_metric(self.balance_sheet_q, 'Total Debt', date_str)
                cash = self._get_metric(self.balance_sheet_q, 'Cash And Cash Equivalents', date_str)
                stockholder_equity = self._get_metric(self.balance_sheet_q, 'Stockholders Equity', date_str)

                # Cash flow metrics
                operating_cf = self._get_metric(self.cash_flow_q, 'Operating Cash Flow', date_str)
                free_cf = self._get_metric(self.cash_flow_q, 'Free Cash Flow', date_str)

                # Calculate ratios
                features = {
                    'Revenue': revenue,
                    'Net_Income': net_income,
                    'Gross_Margin': gross_profit / revenue if revenue else np.nan,
                    'Operating_Margin': operating_income / revenue if revenue else np.nan,
                    'Net_Margin': net_income / revenue if revenue else np.nan,
                    'ROE': net_income / stockholder_equity if stockholder_equity else np.nan,
                    'ROA': net_income / total_assets if total_assets else np.nan,
                    'Debt_to_Equity': total_debt / stockholder_equity if stockholder_equity else np.nan,
                    'Cash_Ratio': cash / total_assets if total_assets else np.nan,
                    'Operating_CF_Margin': operating_cf / revenue if revenue else np.nan,
                    'FCF_Margin': free_cf / revenue if revenue else np.nan,
                }

                financial_ts.append({'Date': date, **features})
            except:
                continue

        if financial_ts:
            fin_df = pd.DataFrame(financial_ts)
            fin_df.set_index('Date', inplace=True)
            fin_df = fin_df.resample('D').ffill()
            return fin_df
        else:
            return pd.DataFrame()

    def _get_metric(self, df, metric_name, date_col):
        """Extract specific metric from financial statement"""
        try:
            row = df[df['Unnamed: 0'] == metric_name]
            if not row.empty and date_col in row.columns:
                value = row[date_col].values[0]
                return float(value) if pd.notna(value) else np.nan
        except:
            pass
        return np.nan

    def prepare_train_test(self, test_size=0.2):
        """Prepare training and testing sets"""
        print("Preparing train/test split...")

        # Remove rows with NaN in target (including the last row now, since we shifted)
        df = self.data.dropna(subset=['Return']).copy()

        print(f"After removing NaN targets: {len(df)} samples")

        # Remove target AND current day price features to prevent leakage
        feature_cols = [col for col in df.columns if col not in
                       ['Return', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Dividends', 'Stock Splits',
                        # Also remove features that use today's close in their calculation
                        'Close_to_High', 'Close_to_Low', 'Daily_Range',
                        'Price_to_SMA_5', 'Price_to_SMA_10', 'Price_to_SMA_20', 'Price_to_SMA_50']]

        X = df[feature_cols]
        y = df['Return']

        # Check NaN percentage per column
        nan_pct = X.isna().sum() / len(X) * 100
        print(f"\nFeatures with >50% NaN: {(nan_pct > 50).sum()}")

        # Drop columns with >50% NaN
        cols_to_keep = nan_pct[nan_pct <= 50].index.tolist()
        X = X[cols_to_keep]
        print(f"Kept {len(cols_to_keep)} features after dropping high-NaN columns")

        # For remaining NaN, use forward fill then backward fill
        X = X.ffill().bfill()

        # If still any NaN (at the very beginning), fill with column mean
        if X.isna().any().any():
            print("Filling remaining NaN with column means...")
            X = X.fillna(X.mean())

        # Remove any rows that still have NaN (shouldn't be any now)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        print(f"Final data shape: {X.shape}")

        if len(X) == 0:
            raise ValueError("No valid samples after cleaning! Check your data quality.")

        # Time-series split (no shuffling)
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

        print(f"\nTraining set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
        print(f"Number of features: {len(self.feature_names)}")

    def train_model(self, use_grid_search=True):
        """Train Random Forest model with optional hyperparameter tuning"""
        print("\nTraining Random Forest model...")

        if use_grid_search:
            print("Performing grid search for hyperparameter tuning...")

            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3]
            }

            rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            grid_search = GridSearchCV(
                rf_base, param_grid, cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )

            grid_search.fit(self.X_train_scaled, self.y_train)

            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")

        else:
            # Use sensible default parameters
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            self.model.fit(self.X_train_scaled, self.y_train)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nModel training complete!")

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)

        # Training metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)

        # Testing metrics
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)

        print("\nTRAINING METRICS:")
        print(f"  RMSE: {train_rmse:.6f}")
        print(f"  MAE:  {train_mae:.6f}")
        print(f"  R²:   {train_r2:.4f}")

        print("\nTESTING METRICS:")
        print(f"  RMSE: {test_rmse:.6f}")
        print(f"  MAE:  {test_mae:.6f}")
        print(f"  R²:   {test_r2:.4f}")

        # Directional accuracy
        train_dir_acc = np.mean((y_train_pred > 0) == (self.y_train > 0))
        test_dir_acc = np.mean((y_test_pred > 0) == (self.y_test > 0))

        print(f"\nDIRECTIONAL ACCURACY:")
        print(f"  Training: {train_dir_acc:.2%}")
        print(f"  Testing:  {test_dir_acc:.2%}")

        # Feature importance
        print("\nTOP 20 MOST IMPORTANT FEATURES:")
        for idx, row in self.feature_importance.head(20).iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.4f}")

        # Create visualizations
        self._plot_results(y_test_pred)

        return {
            'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
            'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
            'train_dir_acc': train_dir_acc, 'test_dir_acc': test_dir_acc
        }

    def _plot_results(self, y_pred):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()],
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        axes[0, 0].set_title('Actual vs Predicted Returns')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Returns')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature Importance (Top 15)
        top_features = self.feature_importance.head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importances')
        axes[1, 0].invert_yaxis()

        # 4. Time series of predictions
        axes[1, 1].plot(self.y_test.index, self.y_test.values,
                       label='Actual', alpha=0.7, linewidth=1)
        axes[1, 1].plot(self.y_test.index, y_pred,
                       label='Predicted', alpha=0.7, linewidth=1)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Returns')
        axes[1, 1].set_title('Time Series: Actual vs Predicted')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.ticker}_rf_results.png', dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved as '{self.ticker}_rf_results.png'")
        plt.show()

    def predict_tomorrow(self):
        """Predict next day return using most recent data"""
        print("\n" + "="*60)
        print("NEXT-DAY PREDICTION")
        print("="*60)

        # Get the most recent complete data point
        # Since we're predicting tomorrow, we can use today's complete data
        latest_data = self.data.iloc[[-2]][self.feature_names]  # Use -2 because -1 has NaN target

        # Check for NaN values
        if latest_data.isna().any().any():
            print("Warning: Latest data contains NaN values. Forward filling...")
            latest_data = latest_data.ffill().bfill()

        # Scale features
        latest_scaled = self.scaler.transform(latest_data)

        # Predict
        prediction = self.model.predict(latest_scaled)[0]

        # Get current price (from second-to-last row since last row has NaN return)
        current_price = self.stock_data['Close'].iloc[-2]
        predicted_price = current_price * (1 + prediction)

        print(f"\nCurrent date: {self.data.index[-2].strftime('%Y-%m-%d')}")
        print(f"Current price: ${current_price:.2f}")
        print(f"\nPredicted return for NEXT trading day: {prediction:.4f} ({prediction*100:.2f}%)")
        print(f"Predicted price: ${predicted_price:.2f}")
        print(f"\nNote: This predicts the return from {self.data.index[-2].strftime('%Y-%m-%d')} to the next trading day")

        # Show key features influencing prediction
        feature_values = latest_data.iloc[0]
        top_features_today = self.feature_importance.head(10)

        print("\nTop features affecting this prediction:")
        for idx, row in top_features_today.iterrows():
            feat_name = row['feature']
            if feat_name in feature_values.index:
                print(f"  {feat_name:30s} = {feature_values[feat_name]:10.4f}")

        return prediction, predicted_price

    def save_model(self, filename=None):
        """Save trained model"""
        if filename is None:
            filename = f'{self.ticker}_rf_model.pkl'

        import pickle
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\nModel saved to '{filename}'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("RANDOM FOREST STOCK RETURN PREDICTION")
    print("="*60)

    # Initialize predictor
    predictor = StockReturnPredictor(ticker='NVDA')

    # Load all data
    predictor.load_data()

    # Engineer features
    predictor.engineer_features()

    # Prepare train/test split
    predictor.prepare_train_test(test_size=0.2)

    # Train model (set use_grid_search=False for faster training)
    predictor.train_model(use_grid_search=False)

    # Evaluate model
    metrics = predictor.evaluate_model()

    # Predict tomorrow's return
    predictor.predict_tomorrow()

    # Save model
    predictor.save_model()

    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)