"""
BATCH FEATURE GENERATOR: All Stocks with Release-Aligned Data
==============================================================

Generates daily feature datasets for all stocks with:
- Stock Returns
- Sentiment
- Technical Indicators (MACD, RSI, Bollinger Bands, etc.)
- Financial Ratios (aligned to release dates)
- Macroeconomic Data (GDP, IPI, Unemployment - aligned to release dates)
- VIX

All data starts from actual release dates, not quarter/observation dates.
Outputs saved to 'feature_datasets/' folder.

Usage:
------
python batch_feature_generator.py
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
STOCK_DATA_DIR = 'stock_data'
SENTIMENT_DIR = 'Data_Sentiment_Index/csv_files'
FINANCIALS_DIR = 'financials_data/quarterly'
RELEASE_DATES_DIR = 'Quarterly_Release_Dates'
MACRO_DIR = 'Macroeconomic_Data'
VIX_DIR = 'VIX'
OUTPUT_DIR = 'feature_datasets'

# Parameters
FISCAL_QUARTER_TOLERANCE = 40  # Days tolerance for matching fiscal quarters
DEFAULT_REPORTING_LAG = 45  # Days to wait if actual release date is missing

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
    print(f"\n→ Loading quarterly release dates...")

    path = Path(directory)
    if not path.exists():
        print(f"   ⚠ Release dates directory not found: {directory}")
        return {}

    all_files = sorted(path.glob('sp500_quarterly_dates_batch_*.xlsx'))

    if not all_files:
        print(f"   ⚠ No release date files found")
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
            print(f"   ⚠ Error reading {fpath.name}: {e}")
            continue

    print(f"   ✓ Loaded release dates for {len(all_tickers)} tickers")
    return all_tickers


def load_macro_release_dates(macro_dir=MACRO_DIR):
    """Load macroeconomic release dates"""
    print(f"\n→ Loading macro release dates...")

    macro_releases = {'gdp': pd.DataFrame(), 'ipi': pd.DataFrame(), 'unemployment': pd.DataFrame()}

    if not Path(macro_dir).exists():
        print(f"   ⚠ Directory not found: {macro_dir}")
        return macro_releases

    # Load GDP and IPI
    try:
        gdp_ipi_path = Path(macro_dir) / 'Calendar GDP AND IPI.csv'
        if gdp_ipi_path.exists():
            gdp_ipi = pd.read_csv(gdp_ipi_path)
            gdp_ipi.columns = [c.strip() for c in gdp_ipi.columns]

            quarter_col = [c for c in gdp_ipi.columns if 'quarter' in c.lower()][0]
            release_col = [c for c in gdp_ipi.columns if 'release' in c.lower() and 'date' in c.lower()][0]

            gdp_ipi['observation_date'] = gdp_ipi[quarter_col].apply(_quarter_label_to_end_date)
            gdp_ipi['release_date'] = pd.to_datetime(gdp_ipi[release_col]).dt.tz_localize(None).dt.normalize()

            macro_releases['gdp'] = gdp_ipi[['observation_date', 'release_date']].copy()
            macro_releases['ipi'] = macro_releases['gdp'].copy()
            print(f"   ✓ GDP/IPI: {len(gdp_ipi)} quarters")
    except Exception as e:
        print(f"   ⚠ GDP/IPI error: {e}")

    # Load Unemployment
    try:
        unemp_path = Path(macro_dir) / 'CALENDAR UNEMPLOYMENT RATE.csv'
        if unemp_path.exists():
            unemp = pd.read_csv(unemp_path)
            unemp.columns = [c.strip() for c in unemp.columns]

            first_col = unemp.columns[0]
            release_col = [c for c in unemp.columns if 'release' in c.lower()][0] if any('release' in c.lower() for c in unemp.columns) else unemp.columns[1]

            unemp['observation_date'] = pd.to_datetime(unemp[first_col], format='%b-%y', errors='coerce')
            if unemp['observation_date'].isna().all():
                unemp['observation_date'] = pd.to_datetime(unemp[first_col], errors='coerce')

            unemp['release_date'] = pd.to_datetime(unemp[release_col], errors='coerce')
            unemp['release_date'] = unemp['release_date'].dt.tz_localize(None).dt.normalize()

            macro_releases['unemployment'] = unemp[['observation_date', 'release_date']].dropna()
            print(f"   ✓ Unemployment: {len(macro_releases['unemployment'])} months")
    except Exception as e:
        print(f"   ⚠ Unemployment error: {e}")

    return macro_releases


# ============================================================================
# FEATURE GENERATOR CLASS
# ============================================================================

class FeatureGenerator:
    def __init__(self, ticker, release_dates, macro_releases):
        self.ticker = ticker
        self.release_dates = release_dates.get(ticker, {})
        self.macro_releases = macro_releases
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

        # Drop OHLCV columns, keep only features
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
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

    def compute_financial_ratios(self):
        """Load and compute financial ratios"""
        inc_path = Path(FINANCIALS_DIR) / f"{self.ticker}_income_statement_quarterly.csv"
        bal_path = Path(FINANCIALS_DIR) / f"{self.ticker}_balance_sheet_quarterly.csv"
        cf_path = Path(FINANCIALS_DIR) / f"{self.ticker}_cash_flow_quarterly.csv"

        if not all([inc_path.exists(), bal_path.exists(), cf_path.exists()]):
            return pd.DataFrame()

        try:
            # Load and transpose
            inc = pd.read_csv(inc_path, index_col=0).T
            bal = pd.read_csv(bal_path, index_col=0).T
            cf = pd.read_csv(cf_path, index_col=0).T

            inc.index = pd.to_datetime(inc.index, utc=True).tz_localize(None).normalize()
            bal.index = pd.to_datetime(bal.index, utc=True).tz_localize(None).normalize()
            cf.index = pd.to_datetime(cf.index, utc=True).tz_localize(None).normalize()

            # Extract metrics
            def get_col(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return pd.to_numeric(df[c], errors='coerce')
                return pd.Series(np.nan, index=df.index)

            ratios = pd.DataFrame(index=inc.index)

            # Income metrics
            net_income = get_col(inc, ['Net Income', 'netIncome'])
            revenue = get_col(inc, ['Total Revenue', 'totalRevenue'])
            op_income = get_col(inc, ['Operating Income', 'operatingIncome'])

            # Balance sheet metrics
            equity = get_col(bal, ['Stockholders Equity', 'totalStockholderEquity'])
            assets = get_col(bal, ['Total Assets', 'totalAssets'])
            debt = get_col(bal, ['Total Debt', 'totalDebt'])
            cash = get_col(bal, ['Cash And Cash Equivalents', 'cashAndCashEquivalents'])
            curr_assets = get_col(bal, ['Current Assets', 'totalCurrentAssets'])
            curr_liab = get_col(bal, ['Current Liabilities', 'totalCurrentLiabilities'])

            # Cash flow metrics
            fcf = get_col(cf, ['Free Cash Flow', 'freeCashFlow'])
            ocf = get_col(cf, ['Operating Cash Flow', 'totalCashFromOperatingActivities'])

            # Compute ratios
            ratios['roe'] = (net_income / equity.replace(0, np.nan)) * 100
            ratios['roa'] = (net_income / assets.replace(0, np.nan)) * 100
            ratios['op_margin'] = (op_income / revenue.replace(0, np.nan)) * 100
            ratios['debt_to_equity'] = debt / equity.replace(0, np.nan)
            ratios['liquidity_ratio'] = cash / assets.replace(0, np.nan)
            ratios['current_ratio'] = curr_assets / curr_liab.replace(0, np.nan)
            ratios['free_cf_margin'] = (fcf / revenue.replace(0, np.nan)) * 100
            ratios['revenue_growth'] = revenue.pct_change() * 100
            ratios['ocf_to_assets'] = ocf / assets.replace(0, np.nan)

            return ratios.sort_index()

        except Exception as e:
            print(f"      ⚠ Financial ratio error: {e}")
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

    def align_macro_to_release_dates(self, macro_df, release_calendar, label):
        """Align macro data to release dates"""
        if macro_df.empty or release_calendar.empty:
            return pd.DataFrame()

        macro_df = macro_df.copy()
        if 'observation_date' in macro_df.columns:
            macro_df = macro_df.set_index('observation_date')

        release_rows = []
        for obs_date, row in macro_df.iterrows():
            matches = release_calendar[
                abs(release_calendar['observation_date'] - obs_date) <= pd.Timedelta(days=5)
            ]

            if len(matches) > 0:
                release_date = matches.iloc[0]['release_date'].normalize()
                release_rows.append((release_date, row))

        if not release_rows:
            return pd.DataFrame()

        dates, rows = zip(*release_rows)
        aligned = pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))
        aligned = aligned.sort_index()
        aligned.columns = [f'{label}_{c}' for c in aligned.columns]

        return aligned

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

            # 2. Sentiment
            sentiment = self.load_sentiment()
            if not sentiment.empty:
                df = df.join(sentiment, how='left')
                df['Sentiment'] = df['Sentiment'].ffill()

            # 3. VIX
            vix = self.load_vix()
            if not vix.empty:
                df = df.join(vix, how='left')
                df['VIX'] = df['VIX'].ffill()

            # 4. Financial Ratios (aligned to release dates)
            ratios = self.compute_financial_ratios()
            if not ratios.empty:
                aligned_ratios = self.align_to_release_dates(ratios, is_financial=True)
                daily_ratios = self.create_daily_features(aligned_ratios)
                if not daily_ratios.empty:
                    df = df.join(daily_ratios, how='left')

            # 5. Macroeconomic Data (aligned to release dates)
            # GDP
            gdp_path = Path(MACRO_DIR) / 'GDP_quarterly.csv'
            if gdp_path.exists() and not self.macro_releases['gdp'].empty:
                try:
                    gdp_df = pd.read_csv(gdp_path)
                    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'], errors='coerce')
                    aligned_gdp = self.align_macro_to_release_dates(gdp_df, self.macro_releases['gdp'], 'GDP')
                    daily_gdp = self.create_daily_features(aligned_gdp)
                    if not daily_gdp.empty:
                        df = df.join(daily_gdp, how='left')
                except:
                    pass

            # IPI
            ipi_path = Path(MACRO_DIR) / 'IPI_quarterly.csv'
            if ipi_path.exists() and not self.macro_releases['ipi'].empty:
                try:
                    ipi_df = pd.read_csv(ipi_path)
                    ipi_df['observation_date'] = pd.to_datetime(ipi_df['observation_date'], errors='coerce')
                    aligned_ipi = self.align_macro_to_release_dates(ipi_df, self.macro_releases['ipi'], 'IPI')
                    daily_ipi = self.create_daily_features(aligned_ipi)
                    if not daily_ipi.empty:
                        df = df.join(daily_ipi, how='left')
                except:
                    pass

            # Unemployment
            unemp_path = Path(MACRO_DIR) / 'Unemployment_Rate_monthly.csv'
            if unemp_path.exists() and not self.macro_releases['unemployment'].empty:
                try:
                    unemp_df = pd.read_csv(unemp_path)
                    unemp_df['observation_date'] = pd.to_datetime(unemp_df['observation_date'], errors='coerce')
                    aligned_unemp = self.align_macro_to_release_dates(unemp_df, self.macro_releases['unemployment'], 'UNEMP')
                    daily_unemp = self.create_daily_features(aligned_unemp)
                    if not daily_unemp.empty:
                        df = df.join(daily_unemp, how='left')
                except:
                    pass

            self.final_df = df
            return df

        except Exception as e:
            raise Exception(f"Feature generation failed: {e}")

    def save(self, output_dir=OUTPUT_DIR):
        """Save feature dataset"""
        if self.final_df is None or self.final_df.empty:
            raise ValueError("No data to save")

        Path(output_dir).mkdir(exist_ok=True)
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


def batch_process_all_stocks():
    """Process all stocks and generate feature datasets"""
    print("=" * 80)
    print("BATCH FEATURE GENERATOR - All Stocks")
    print("=" * 80)

    # Load global release dates
    release_dates = load_all_release_dates()
    macro_releases = load_macro_release_dates()

    # Discover tickers
    tickers = discover_tickers()
    if not tickers:
        print("\n❌ No tickers found in stock_data directory")
        return

    print(f"\n→ Found {len(tickers)} tickers to process\n")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results = []
    errors = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>4}/{len(tickers)}] {ticker:<6}", end='  ')

        try:
            generator = FeatureGenerator(ticker, release_dates, macro_releases)
            df = generator.generate()
            output_path = generator.save()

            # Stats
            num_features = len(df.columns)
            num_rows = len(df)
            date_range = f"{df.index[0].date()} to {df.index[-1].date()}"

            results.append({
                'ticker': ticker,
                'features': num_features,
                'rows': num_rows,
                'start_date': df.index[0],
                'end_date': df.index[-1],
                'file': output_path.name
            })

            print(f"✓  {num_features} features, {num_rows} rows, {date_range}")

        except Exception as e:
            errors.append({'ticker': ticker, 'error': str(e)})
            print(f"✗  {str(e)[:60]}")

    # Save summary
    print("\n" + "=" * 80)
    print(f"COMPLETED: {len(results)} successful, {len(errors)} failed")
    print("=" * 80)

    if results:
        results_df = pd.DataFrame(results)
        summary_path = Path(OUTPUT_DIR) / f'_processing_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")
        print(f"\n✓ Feature datasets saved to: {OUTPUT_DIR}/")

        print(f"\nFeature Statistics:")
        print(f"  Mean features: {results_df['features'].mean():.0f}")
        print(f"  Mean rows: {results_df['rows'].mean():.0f}")

    if errors:
        errors_df = pd.DataFrame(errors)
        error_path = Path(OUTPUT_DIR) / f'_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        errors_df.to_csv(error_path, index=False)
        print(f"\n⚠ Errors saved to: {error_path}")


if __name__ == '__main__':
    batch_process_all_stocks()