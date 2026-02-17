"""
ENHANCED FEATURE GENERATOR (based on claude_lagsentiment.py)
=============================================================

Builds daily feature datasets for all stocks with:
- Price/technical features and lags
- Sentiment + lags
- VIX + lags
- Financial fundamentals and ratios (aligned to release dates)
- Added: Dividends, market_cap, log_mktcap, book_to_market
- Macroeconomic series aligned to release dates

Outputs are written under ConstructionDataset/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCK_DATA_DIR = PROJECT_ROOT / 'stock_data'
SENTIMENT_DIR = PROJECT_ROOT / 'Data_Sentiment_Index' / 'csv_files'
FINANCIALS_DIR = PROJECT_ROOT / 'financials_data' / 'quarterly'
RELEASE_DATES_DIR = PROJECT_ROOT / 'Quarterly_Release_Dates'
MACRO_DIR = PROJECT_ROOT / 'Macroeconomic_Data'
VIX_DIR = PROJECT_ROOT / 'VIX'
OUTPUT_DIR = PROJECT_ROOT / 'ConstructionDataset'

# Parameters
FISCAL_QUARTER_TOLERANCE = 40  # Days tolerance for matching fiscal quarters
DEFAULT_REPORTING_LAG = 45  # Days to wait if actual release date is missing

# Lag windows for VIX and Sentiment
VIX_SENTIMENT_LAGS = [1, 2, 3, 5, 10, 20]

# Test with single ticker
TEST_TICKER = None  # Set to None to process all tickers

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _quarter_label_to_end_date(label):
    """Convert quarter label to end date"""
    label = str(label).strip().upper().replace('_', ' ')
    m = re.search(r'(\d{4})[^\d]*Q([1-4])', label)
    if not m:
        m = re.search(r'Q([1-4])[^\d]*(\d{4})', label)
        if m:
            q, yr = int(m.group(1)), int(m.group(2))
        else:
            raise ValueError(f"Cannot parse quarter label: {label}")
    else:
        yr, q = int(m.group(1)), int(m.group(2))

    quarter_ends = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}
    return pd.Timestamp(f'{yr}-{quarter_ends[q]}')


def load_all_release_dates(directory=RELEASE_DATES_DIR):
    """Load release dates for all tickers"""
    print("\nINFO: Loading quarterly release dates...")

    path = Path(directory)
    if not path.exists():
        print(f"WARNING: Release dates directory not found: {directory}")
        return {}

    all_files = sorted(path.glob('sp500_quarterly_dates_batch_*.xlsx'))

    if not all_files:
        print("WARNING: No release date files found")
        return {}

    all_tickers = {}

    for fpath in all_files:
        try:
            df = pd.read_excel(fpath, header=0, engine='openpyxl')
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)

            if len(cols) >= 3:
                rename = {cols[0]: 'Ticker', cols[1]: 'Company', cols[2]: 'CIK'}
                df.rename(columns=rename, inplace=True)
                df['Ticker'] = df['Ticker'].astype(str).str.strip()

                quarter_cols = [c for c in df.columns if c not in ('Ticker', 'Company', 'CIK')]

                for _, row in df.iterrows():
                    ticker = row['Ticker']
                    if pd.isna(ticker) or ticker == 'nan':
                        continue

                    ticker_map = {}
                    for qcol in quarter_cols:
                        release_val = row[qcol]
                        if pd.isna(release_val):
                            continue

                        try:
                            release_date = pd.to_datetime(release_val).tz_localize(None).normalize()
                            qend = _quarter_label_to_end_date(str(qcol))
                            ticker_map[qend] = release_date
                        except Exception:
                            continue

                    if ticker_map:
                        all_tickers[ticker] = ticker_map

        except Exception as e:
            print(f"WARNING: Error reading {fpath.name}: {e}")
            continue

    print(f"OK: Loaded release dates for {len(all_tickers)} tickers")
    return all_tickers


def load_macro_release_calendars(macro_dir=MACRO_DIR):
    """Load macroeconomic release date calendars - CORRECT VERSION"""
    print("\nINFO: Loading macro release calendars...")

    calendars = {
        'gdp': pd.DataFrame(),
        'ipi': pd.DataFrame(),
        'unemployment': pd.DataFrame()
    }

    # GDP and IPI Calendar
    print(f"   - Loading GDP/IPI calendar...")
    gdp_ipi_path = Path(macro_dir) / 'Calendar GDP AND IPI.csv'

    if gdp_ipi_path.exists():
        try:
            df = pd.read_csv(gdp_ipi_path)
            df.columns = [c.strip() for c in df.columns]

            # Parse quarter to observation date
            df['observation_date'] = df['Quarter'].apply(_quarter_label_to_end_date)

            # Parse release date - try multiple formats
            for fmt in ['%d-%b-%y', '%d-%m-%Y', '%Y-%m-%d']:
                try:
                    df['release_date'] = pd.to_datetime(df['Release date'], format=fmt, errors='coerce')
                    if not df['release_date'].isna().all():
                        break
                except:
                    continue

            if df['release_date'].isna().all():
                df['release_date'] = pd.to_datetime(df['Release date'], errors='coerce')

            df['release_date'] = df['release_date'].dt.tz_localize(None).dt.normalize()

            calendars['gdp'] = df[['observation_date', 'release_date']].dropna().copy()
            calendars['ipi'] = calendars['gdp'].copy()

            print(f"OK: Loaded {len(calendars['gdp'])} GDP/IPI quarters")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print(f"WARNING: File not found: {gdp_ipi_path}")

    # Unemployment Calendar
    print(f"   - Loading Unemployment calendar...")
    unemp_path = Path(macro_dir) / 'CALENDAR UNEMPLOYMENT RATE.csv'

    if unemp_path.exists():
        try:
            df = pd.read_csv(unemp_path)
            df.columns = [c.strip() for c in df.columns]

            # Get column names
            month_col = df.columns[0]
            release_col = df.columns[1]

            # Parse observation date (month)
            df['observation_date'] = pd.to_datetime(df[month_col], format='%b-%y', errors='coerce')

            # Parse release date - try multiple formats
            for fmt in ['%d-%b-%y', '%d-%m-%Y', '%Y-%m-%d']:
                try:
                    df['release_date'] = pd.to_datetime(df[release_col], format=fmt, errors='coerce')
                    if not df['release_date'].isna().all():
                        break
                except:
                    continue

            if df['release_date'].isna().all():
                df['release_date'] = pd.to_datetime(df[release_col], errors='coerce')

            df['release_date'] = df['release_date'].dt.tz_localize(None).dt.normalize()

            calendars['unemployment'] = df[['observation_date', 'release_date']].dropna()

            print(f"OK: Loaded {len(calendars['unemployment'])} unemployment months")
        except Exception as e:
            print(f"ERROR: {e}")
    else:
        print(f"WARNING: File not found: {unemp_path}")

    return calendars


# ============================================================================
# FEATURE GENERATOR CLASS
# ============================================================================

class FeatureGenerator:
    def __init__(self, ticker, release_dates, macro_calendars):
        self.ticker = ticker
        self.release_dates = release_dates.get(ticker, {})
        self.macro_calendars = macro_calendars
        self.stock_df = None
        self.final_df = None

    def load_stock_data(self):
        """Load stock data and compute returns + technical indicators"""
        stock_path = Path(STOCK_DATA_DIR) / f'{self.ticker}.csv'
        if not stock_path.exists():
            raise FileNotFoundError(f"Stock data not found: {stock_path}")

        df = pd.read_csv(stock_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        self.close_px = pd.to_numeric(df['Close'], errors='coerce')
        if 'Dividends' not in df.columns:
            df['Dividends'] = 0.0
        df['Dividends'] = pd.to_numeric(df['Dividends'], errors='coerce').fillna(0.0)
        for lag in VIX_SENTIMENT_LAGS:
            df[f'Dividends_Lag{lag}'] = df['Dividends'].shift(lag)

        # Returns
        df['Return'] = df['Close'].pct_change()

        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()

        # Volatility
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Return'].rolling(window).std()
            df[f'ATR_{window}'] = (df['High'] - df['Low']).rolling(window).mean()

        # RSI
        df['RSI_14'] = self._calc_rsi(df['Close'], 14)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # For predicting return at t, shift same-day technical/volume indicators to t-1.
        contemporaneous_to_lag1 = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'Volatility_5', 'ATR_5', 'Volatility_10', 'ATR_10', 'Volatility_20', 'ATR_20',
            'RSI_14',
            'MACD', 'MACD_Signal', 'MACD_Diff',
            'Volume_MA_20', 'Volume_Ratio',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
        ]
        for col in contemporaneous_to_lag1:
            if col in df.columns:
                df[col] = df[col].shift(1)

        # Keep dividends as a feature; drop raw OHLCV levels.
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Stock Splits']
        df = df[[c for c in df.columns if c not in drop_cols]]

        self.stock_df = df
        return df

    def _calc_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    def load_sentiment(self):
        """Load sentiment data"""
        sent_path = Path(SENTIMENT_DIR) / f'{self.ticker}.csv'
        if not sent_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(sent_path, header=None, names=['Date', 'Sentiment'])
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
        df.set_index('Date', inplace=True)
        return df.sort_index()

    def load_vix(self):
        """Load VIX data"""
        vix_path = Path(VIX_DIR) / 'vix_data.csv'
        if not vix_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(vix_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        if df['Date'].isna().any():
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.tz_localize(None).dt.normalize()
        df.set_index('Date', inplace=True)
        return df[['Close']].rename(columns={'Close': 'VIX'}).sort_index()

    @staticmethod
    def _safe_div(numerator, denominator):
        return numerator / denominator.replace(0, np.nan)

    def compute_financial_fundamentals(self):
        """Load quarterly fundamentals and ratio rules from the SQL_Queries notebook."""
        inc_path = Path(FINANCIALS_DIR) / f"{self.ticker}_income_statement_quarterly.csv"
        bal_path = Path(FINANCIALS_DIR) / f"{self.ticker}_balance_sheet_quarterly.csv"
        cf_path = Path(FINANCIALS_DIR) / f"{self.ticker}_cash_flow_quarterly.csv"

        if not all([inc_path.exists(), bal_path.exists(), cf_path.exists()]):
            return pd.DataFrame()

        try:
            inc = pd.read_csv(inc_path, index_col=0).T
            bal = pd.read_csv(bal_path, index_col=0).T
            cf = pd.read_csv(cf_path, index_col=0).T

            inc.index = pd.to_datetime(inc.index, utc=True).tz_localize(None).normalize()
            bal.index = pd.to_datetime(bal.index, utc=True).tz_localize(None).normalize()
            cf.index = pd.to_datetime(cf.index, utc=True).tz_localize(None).normalize()

            def get_col(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return pd.to_numeric(df[c], errors='coerce')
                return pd.Series(np.nan, index=df.index)

            f = pd.DataFrame(index=inc.index)

            f['net_income'] = get_col(inc, ['Net Income', 'netIncome'])
            f['total_revenue'] = get_col(inc, ['Total Revenue', 'totalRevenue'])
            f['operating_income'] = get_col(inc, ['Operating Income', 'operatingIncome'])

            f['total_assets'] = get_col(bal, ['Total Assets', 'totalAssets'])
            f['total_equity'] = get_col(bal, ['Stockholders Equity', 'totalStockholderEquity'])
            f['total_debt'] = get_col(bal, ['Total Debt', 'totalDebt'])
            f['current_assets'] = get_col(bal, ['Current Assets', 'totalCurrentAssets'])
            f['current_liabilities'] = get_col(bal, ['Current Liabilities', 'totalCurrentLiabilities'])
            f['cash_equivalents'] = get_col(bal, ['Cash And Cash Equivalents', 'cashAndCashEquivalents'])
            f['shares_outstanding'] = get_col(
                bal,
                [
                    'Ordinary Shares Number',
                    'Share Issued',
                    'Common Stock Shares Outstanding',
                    'commonStockSharesOutstanding',
                    'sharesOutstanding',
                ],
            )

            f['free_cash_flow'] = get_col(cf, ['Free Cash Flow', 'freeCashFlow'])
            f['operating_cash_flow'] = get_col(cf, ['Operating Cash Flow', 'totalCashFromOperatingActivities'])

            f['roe'] = self._safe_div(f['net_income'], f['total_equity']) * 100
            f['roa'] = self._safe_div(f['net_income'], f['total_assets']) * 100
            f['op_margin'] = self._safe_div(f['operating_income'], f['total_revenue']) * 100
            f['debt_to_equity'] = self._safe_div(f['total_debt'], f['total_equity'])
            f['liquidity_ratio'] = self._safe_div(f['cash_equivalents'], f['total_assets'])
            f['current_ratio'] = self._safe_div(f['current_assets'], f['current_liabilities'])
            f['free_cf_margin'] = self._safe_div(f['free_cash_flow'], f['total_revenue']) * 100
            f['ocf_to_assets'] = self._safe_div(f['operating_cash_flow'], f['total_assets'])
            f['prev_revenue'] = f['total_revenue'].shift(1)
            f['revenue_growth'] = self._safe_div(
                f['total_revenue'] - f['prev_revenue'],
                f['prev_revenue'],
            ) * 100

            return f.sort_index()

        except Exception as e:
            print(f"WARNING: Financial ratio error: {e}")
            return pd.DataFrame()

    def align_to_release_dates(self, data_df, is_financial=True):
        """Align quarterly/monthly data to release dates"""
        if data_df.empty:
            return pd.DataFrame()

        aligned_rows = []

        if is_financial:
            # Match with ticker-specific release dates
            for q_end, row in data_df.iterrows():
                found_date = None
                best_diff = pd.Timedelta(days=FISCAL_QUARTER_TOLERANCE)

                for key_date, r_date in self.release_dates.items():
                    diff = abs(key_date - q_end)
                    if diff < best_diff:
                        found_date = r_date
                        best_diff = diff

                if found_date:
                    final_date = found_date
                else:
                    final_date = q_end + pd.Timedelta(days=DEFAULT_REPORTING_LAG)

                aligned_rows.append((final_date.normalize(), row))

        if not aligned_rows:
            return pd.DataFrame()

        dates, rows = zip(*aligned_rows)
        aligned_df = pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))
        aligned_df = aligned_df.sort_index()
        aligned_df = aligned_df[~aligned_df.index.duplicated(keep='last')]

        return aligned_df

    def align_macro_to_release_dates(self, macro_df, calendar_df, label):
        """
        Align macro data to release dates - CORRECT VERSION.
        KEY: Uses RELEASE DATE as the index (when data becomes available).
        """
        if macro_df.empty or calendar_df.empty:
            return pd.DataFrame()

        if 'observation_date' in macro_df.columns:
            macro_df = macro_df.set_index('observation_date')

        macro_df = macro_df.sort_index()
        calendar_df = calendar_df.sort_values('observation_date')

        aligned_rows = []

        for obs_date, row in macro_df.iterrows():
            # Find matching observation date in calendar
            matches = calendar_df[
                abs(calendar_df['observation_date'] - obs_date) <= pd.Timedelta(days=5)
            ]

            if len(matches) > 0:
                # Use RELEASE DATE as the index
                release_date = matches.iloc[0]['release_date']
                aligned_rows.append((release_date, row))

        if not aligned_rows:
            return pd.DataFrame()

        dates, rows = zip(*aligned_rows)
        aligned_df = pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))
        aligned_df = aligned_df.sort_index()
        aligned_df = aligned_df[~aligned_df.index.duplicated(keep='last')]
        aligned_df.columns = [f'{label}_{c}' for c in aligned_df.columns]

        return aligned_df

    def create_daily_features(self, quarterly_df):
        """Forward-fill quarterly data to daily"""
        if quarterly_df.empty or self.stock_df is None:
            return pd.DataFrame()

        # Create container with stock index
        container = pd.DataFrame(index=self.stock_df.index)

        # Join and forward-fill
        combined = container.join(quarterly_df, how='outer')
        combined[quarterly_df.columns] = combined[quarterly_df.columns].ffill()

        # Trim back to stock dates
        return combined.loc[self.stock_df.index]

    def generate(self):
        """Generate complete feature dataset"""
        try:
            # 1. Load stock data with technicals
            self.load_stock_data()
            df = self.stock_df.copy()

            # 2. Sentiment with lags
            sentiment = self.load_sentiment()
            if not sentiment.empty:
                df = df.join(sentiment, how='left')
                df['Sentiment'] = df['Sentiment'].ffill()

                # Add sentiment lags
                for lag in VIX_SENTIMENT_LAGS:
                    df[f'Sentiment_Lag{lag}'] = df['Sentiment'].shift(lag)

            # 3. VIX with lags
            vix = self.load_vix()
            if not vix.empty:
                df = df.join(vix, how='left')
                df['VIX'] = df['VIX'].ffill()

                # Add VIX lags
                for lag in VIX_SENTIMENT_LAGS:
                    df[f'VIX_Lag{lag}'] = df['VIX'].shift(lag)

            # 4. Financial fundamentals/ratios (aligned to release dates)
            fundamentals = self.compute_financial_fundamentals()
            if not fundamentals.empty:
                aligned_f = self.align_to_release_dates(fundamentals, is_financial=True)
                daily_f = self.create_daily_features(aligned_f)
                if not daily_f.empty:
                    df = df.join(daily_f, how='left')

            # 4b. Market-cap/value features for t-prediction: use only t-1 info.
            if {'shares_outstanding', 'total_equity'}.issubset(df.columns):
                close_lag1 = self.close_px.reindex(df.index).ffill().shift(1)
                shares_lag1 = pd.to_numeric(df['shares_outstanding'], errors='coerce').shift(1)
                equity_lag1 = pd.to_numeric(df['total_equity'], errors='coerce').shift(1)

                df['market_cap'] = close_lag1 * shares_lag1
                df['log_mktcap'] = np.where(df['market_cap'] > 0, np.log(df['market_cap']), np.nan)
                df['book_to_market'] = self._safe_div(equity_lag1, pd.to_numeric(df['market_cap'], errors='coerce'))

            # 5. Macroeconomic Data (aligned to release dates)
            # GDP
            gdp_path = Path(MACRO_DIR) / 'GDP_quarterly.csv'
            if gdp_path.exists() and not self.macro_calendars['gdp'].empty:
                try:
                    gdp_df = pd.read_csv(gdp_path)
                    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'], errors='coerce')
                    gdp_df = gdp_df.set_index('observation_date')

                    aligned_gdp = self.align_macro_to_release_dates(
                        gdp_df,
                        self.macro_calendars['gdp'],
                        'GDP'
                    )

                    if not aligned_gdp.empty:
                        daily_gdp = self.create_daily_features(aligned_gdp)
                        df = df.join(daily_gdp, how='left')
                except:
                    pass

            # IPI
            ipi_path = Path(MACRO_DIR) / 'IPI_quarterly.csv'
            if ipi_path.exists() and not self.macro_calendars['ipi'].empty:
                try:
                    ipi_df = pd.read_csv(ipi_path)
                    ipi_df['observation_date'] = pd.to_datetime(ipi_df['observation_date'], errors='coerce')
                    ipi_df = ipi_df.set_index('observation_date')

                    aligned_ipi = self.align_macro_to_release_dates(
                        ipi_df,
                        self.macro_calendars['ipi'],
                        'IPI'
                    )

                    if not aligned_ipi.empty:
                        daily_ipi = self.create_daily_features(aligned_ipi)
                        df = df.join(daily_ipi, how='left')
                except:
                    pass

            # Unemployment
            unemp_path = Path(MACRO_DIR) / 'Unemployment_Rate_monthly.csv'
            if unemp_path.exists() and not self.macro_calendars['unemployment'].empty:
                try:
                    unemp_df = pd.read_csv(unemp_path)
                    unemp_df['observation_date'] = pd.to_datetime(unemp_df['observation_date'], errors='coerce')
                    unemp_df = unemp_df.set_index('observation_date')

                    aligned_unemp = self.align_macro_to_release_dates(
                        unemp_df,
                        self.macro_calendars['unemployment'],
                        'UNEMP'
                    )

                    if not aligned_unemp.empty:
                        daily_unemp = self.create_daily_features(aligned_unemp)
                        df = df.join(daily_unemp, how='left')
                except:
                    pass

            if 'ticker' not in df.columns:
                df.insert(0, 'ticker', self.ticker)

            self.final_df = df
            return df

        except Exception as e:
            raise Exception(f"Feature generation failed: {e}")

    def save(self, output_dir=OUTPUT_DIR):
        """Save feature dataset"""
        if self.final_df is None or self.final_df.empty:
            raise ValueError("No data to save")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f'{self.ticker}_features.csv'
        self.final_df.to_csv(output_path)
        return output_path


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

def discover_tickers(stock_data_path=STOCK_DATA_DIR):
    """Find all available tickers"""
    path = Path(stock_data_path)
    if not path.exists():
        return []
    return sorted([f.stem for f in path.glob('*.csv')])


def batch_process_stocks():
    """Process stocks and generate feature datasets"""
    print("=" * 80)
    print("INTEGRATED BATCH FEATURE GENERATOR - With Correct Macro Alignment")
    print("=" * 80)

    # Load global release dates
    release_dates = load_all_release_dates()
    macro_calendars = load_macro_release_calendars()

    # Discover tickers
    if TEST_TICKER:
        tickers = [TEST_TICKER]
        print(f"\nINFO: TEST MODE enabled. Processing single ticker: {TEST_TICKER}\n")
    else:
        tickers = discover_tickers()
        if not tickers:
            print("\nERROR: No tickers found in stock_data directory")
            return
        print(f"\nINFO: Found {len(tickers)} tickers to process\n")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>4}/{len(tickers)}] {ticker:<6}", end='  ')

        try:
            generator = FeatureGenerator(ticker, release_dates, macro_calendars)
            df = generator.generate()
            output_path = generator.save()

            # Stats
            num_features = len(df.columns)
            num_rows = len(df)
            date_range = f"{df.index[0].date()} to {df.index[-1].date()}"

            # Count feature types
            macro_cols = [c for c in df.columns if c.startswith(('GDP_', 'IPI_', 'UNEMP_'))]
            vix_cols = [c for c in df.columns if c.startswith('VIX')]
            sent_cols = [c for c in df.columns if c.startswith('Sentiment')]

            results.append({
                'ticker': ticker,
                'features': num_features,
                'rows': num_rows,
                'macro_features': len(macro_cols),
                'vix_features': len(vix_cols),
                'sentiment_features': len(sent_cols),
                'start_date': df.index[0],
                'end_date': df.index[-1],
                'file': output_path.name
            })

            print(f"OK: {num_features} features ({len(macro_cols)} macro, {len(vix_cols)} VIX, {len(sent_cols)} sentiment), {num_rows} rows")

        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})
            print(f"ERROR: {str(e)[:60]}")

    # Save summary
    print("\n" + "=" * 80)
    print(f"COMPLETED: {len(results)} successful, {len(errors)} failed")
    print("=" * 80)

    if results:
        results_df = pd.DataFrame(results)
        summary_path = Path(OUTPUT_DIR) / f'_processing_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(summary_path, index=False)
        print(f"\nOK: Summary saved to: {summary_path}")
        print(f"\nOK: Feature datasets saved to: {OUTPUT_DIR}/")

        print(f"\nFeature Statistics:")
        print(f"  Mean total features: {results_df['features'].mean():.0f}")
        print(f"  Mean macro features: {results_df['macro_features'].mean():.0f}")
        print(f"  Mean VIX features: {results_df['vix_features'].mean():.0f}")
        print(f"  Mean sentiment features: {results_df['sentiment_features'].mean():.0f}")
        print(f"  Mean rows: {results_df['rows'].mean():.0f}")

    if errors:
        errors_df = pd.DataFrame(errors)
        error_path = Path(OUTPUT_DIR) / f'_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        errors_df.to_csv(error_path, index=False)
        print(f"\nWARNING: Errors saved to: {error_path}")


if __name__ == '__main__':
    batch_process_stocks()
