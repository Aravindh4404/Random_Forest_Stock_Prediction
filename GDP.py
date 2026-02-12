"""
BATCH FEATURE GENERATOR: AAPL with Release-Aligned Macro Data (FINAL)
======================================================================

Uses exact filenames from your Macroeconomic_Data folder.

Usage:
------
python batch_feature_generator_AAPL_final.py
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
MACRO_DIR = 'Macroeconomic_Data'  # Exact folder name from screenshot
VIX_DIR = 'VIX'
OUTPUT_DIR = 'feature_datasets'

# Parameters
FISCAL_QUARTER_TOLERANCE = 40
DEFAULT_REPORTING_LAG = 45
TICKER = 'AAPL'

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


def load_macro_release_calendars():
    """Load macroeconomic release date calendars"""
    print(f"\n→ Loading macro release calendars...")

    calendars = {
        'gdp': pd.DataFrame(),
        'ipi': pd.DataFrame(),
        'unemployment': pd.DataFrame()
    }

    # GDP and IPI Calendar - using exact filename from screenshot
    print(f"   - Loading GDP/IPI calendar...")
    gdp_ipi_path = Path(MACRO_DIR) / 'Calendar GDP AND IPI.csv'

    print(f"     Looking at: {gdp_ipi_path}")
    print(f"     Exists: {gdp_ipi_path.exists()}")

    if gdp_ipi_path.exists():
        try:
            df = pd.read_csv(gdp_ipi_path)
            df.columns = [c.strip() for c in df.columns]

            print(f"     Columns found: {df.columns.tolist()}")

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

            print(f"     ✓ Loaded {len(calendars['gdp'])} quarters")
        except Exception as e:
            print(f"     ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"     ⚠ File not found")

    # Unemployment Calendar - using exact filename from screenshot
    print(f"   - Loading Unemployment calendar...")
    unemp_path = Path(MACRO_DIR) / 'CALENDAR UNEMPLOYMENT RATE.csv'

    print(f"     Looking at: {unemp_path}")
    print(f"     Exists: {unemp_path.exists()}")

    if unemp_path.exists():
        try:
            df = pd.read_csv(unemp_path)
            df.columns = [c.strip() for c in df.columns]

            print(f"     Columns found: {df.columns.tolist()}")

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

            print(f"     ✓ Loaded {len(calendars['unemployment'])} months")
        except Exception as e:
            print(f"     ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"     ⚠ File not found")

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

        # Drop OHLCV columns
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

            ratios = pd.DataFrame(index=inc.index)

            net_income = get_col(inc, ['Net Income', 'netIncome'])
            revenue = get_col(inc, ['Total Revenue', 'totalRevenue'])
            op_income = get_col(inc, ['Operating Income', 'operatingIncome'])

            equity = get_col(bal, ['Stockholders Equity', 'totalStockholderEquity'])
            assets = get_col(bal, ['Total Assets', 'totalAssets'])
            debt = get_col(bal, ['Total Debt', 'totalDebt'])
            cash = get_col(bal, ['Cash And Cash Equivalents', 'cashAndCashEquivalents'])
            curr_assets = get_col(bal, ['Current Assets', 'totalCurrentAssets'])
            curr_liab = get_col(bal, ['Current Liabilities', 'totalCurrentLiabilities'])

            fcf = get_col(cf, ['Free Cash Flow', 'freeCashFlow'])
            ocf = get_col(cf, ['Operating Cash Flow', 'totalCashFromOperatingActivities'])

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
        """Align quarterly data to release dates"""
        if data_df.empty:
            return pd.DataFrame()

        aligned_rows = []

        if is_financial:
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
        Align macro data to release dates.
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

        container = pd.DataFrame(index=self.stock_df.index)
        combined = container.join(quarterly_df, how='outer')
        combined[quarterly_df.columns] = combined[quarterly_df.columns].ffill()

        return combined.loc[self.stock_df.index]

    def generate(self):
        """Generate complete feature dataset"""
        try:
            print(f"\n→ Processing {self.ticker}")

            # 1. Stock data
            print(f"   - Loading stock data...")
            self.load_stock_data()
            df = self.stock_df.copy()
            print(f"     ✓ {len(df)} trading days")

            # 2. Sentiment
            print(f"   - Loading sentiment...")
            sentiment = self.load_sentiment()
            if not sentiment.empty:
                df = df.join(sentiment, how='left')
                df['Sentiment'] = df['Sentiment'].ffill()
                print(f"     ✓ Sentiment added")

            # 3. VIX
            print(f"   - Loading VIX...")
            vix = self.load_vix()
            if not vix.empty:
                df = df.join(vix, how='left')
                df['VIX'] = df['VIX'].ffill()
                print(f"     ✓ VIX added")

            # 4. Financial Ratios
            print(f"   - Computing financial ratios...")
            ratios = self.compute_financial_ratios()
            if not ratios.empty:
                aligned_ratios = self.align_to_release_dates(ratios, is_financial=True)
                daily_ratios = self.create_daily_features(aligned_ratios)
                if not daily_ratios.empty:
                    df = df.join(daily_ratios, how='left')
                    print(f"     ✓ {len(aligned_ratios)} quarters aligned")

            # 5. GDP
            print(f"   - Loading GDP...")
            gdp_path = Path(MACRO_DIR) / 'GDP_quarterly.csv'
            if gdp_path.exists() and not self.macro_calendars['gdp'].empty:
                try:
                    gdp_df = pd.read_csv(gdp_path)
                    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
                    gdp_df = gdp_df.set_index('observation_date')

                    aligned_gdp = self.align_macro_to_release_dates(
                        gdp_df,
                        self.macro_calendars['gdp'],
                        'GDP'
                    )

                    if not aligned_gdp.empty:
                        daily_gdp = self.create_daily_features(aligned_gdp)
                        df = df.join(daily_gdp, how='left')
                        print(f"     ✓ {len(aligned_gdp)} GDP releases aligned")
                        print(f"       First release: {aligned_gdp.index[0].date()}")
                        print(f"       Last release: {aligned_gdp.index[-1].date()}")
                except Exception as e:
                    print(f"     ✗ GDP error: {e}")
            else:
                print(f"     ⚠ GDP data or calendar not found")

            # 6. IPI
            print(f"   - Loading IPI...")
            ipi_path = Path(MACRO_DIR) / 'IPI_quarterly.csv'
            if ipi_path.exists() and not self.macro_calendars['ipi'].empty:
                try:
                    ipi_df = pd.read_csv(ipi_path)
                    ipi_df['observation_date'] = pd.to_datetime(ipi_df['observation_date'])
                    ipi_df = ipi_df.set_index('observation_date')

                    aligned_ipi = self.align_macro_to_release_dates(
                        ipi_df,
                        self.macro_calendars['ipi'],
                        'IPI'
                    )

                    if not aligned_ipi.empty:
                        daily_ipi = self.create_daily_features(aligned_ipi)
                        df = df.join(daily_ipi, how='left')
                        print(f"     ✓ {len(aligned_ipi)} IPI releases aligned")
                        print(f"       First release: {aligned_ipi.index[0].date()}")
                        print(f"       Last release: {aligned_ipi.index[-1].date()}")
                except Exception as e:
                    print(f"     ✗ IPI error: {e}")
            else:
                print(f"     ⚠ IPI data or calendar not found")

            # 7. Unemployment
            print(f"   - Loading Unemployment...")
            unemp_path = Path(MACRO_DIR) / 'Unemployment_Rate_monthly.csv'
            if unemp_path.exists() and not self.macro_calendars['unemployment'].empty:
                try:
                    unemp_df = pd.read_csv(unemp_path)
                    unemp_df['observation_date'] = pd.to_datetime(unemp_df['observation_date'])
                    unemp_df = unemp_df.set_index('observation_date')

                    aligned_unemp = self.align_macro_to_release_dates(
                        unemp_df,
                        self.macro_calendars['unemployment'],
                        'UNEMP'
                    )

                    if not aligned_unemp.empty:
                        daily_unemp = self.create_daily_features(aligned_unemp)
                        df = df.join(daily_unemp, how='left')
                        print(f"     ✓ {len(aligned_unemp)} Unemployment releases aligned")
                        print(f"       First release: {aligned_unemp.index[0].date()}")
                        print(f"       Last release: {aligned_unemp.index[-1].date()}")
                except Exception as e:
                    print(f"     ✗ Unemployment error: {e}")
            else:
                print(f"     ⚠ Unemployment data or calendar not found")

            self.final_df = df

            print(f"\n   ✓ Final dataset: {len(df)} rows × {len(df.columns)} features")
            print(f"   ✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")

            return df

        except Exception as e:
            raise Exception(f"Feature generation failed: {e}")

    def save(self, output_dir=OUTPUT_DIR):
        """Save feature dataset"""
        if self.final_df is None or self.final_df.empty:
            raise ValueError("No data to save")

        Path(output_dir).mkdir(exist_ok=True)
        output_path = Path(output_dir) / f'{self.ticker}_features.csv'

        try:
            self.final_df.to_csv(output_path)
            print(f"\n→ Saved to: {output_path}")
            return output_path
        except PermissionError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_path = Path(output_dir) / f'{self.ticker}_features_{timestamp}.csv'
            self.final_df.to_csv(alt_path)
            print(f"\n→ Saved to: {alt_path} (original locked)")
            return alt_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Process AAPL stock"""
    print("=" * 80)
    print("FEATURE GENERATOR - AAPL (Release-Aligned Macro Data)")
    print("=" * 80)

    release_dates = load_all_release_dates()
    macro_calendars = load_macro_release_calendars()

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    try:
        generator = FeatureGenerator(TICKER, release_dates, macro_calendars)
        df = generator.generate()
        output_path = generator.save()

        print("\n" + "=" * 80)
        print("COMPLETED SUCCESSFULLY")
        print("=" * 80)

        # Show macro columns
        macro_cols = [c for c in df.columns if c.startswith(('GDP_', 'IPI_', 'UNEMP_'))]
        if macro_cols:
            print(f"\nMacro columns added: {len(macro_cols)}")
            print(f"Columns: {macro_cols}")
            print("\nFirst 10 rows with macro data:")
            print(df[macro_cols].dropna(how='all').head(10))
        else:
            print("\n⚠ No macro columns were added")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()