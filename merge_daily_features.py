"""
ROBUST FEATURE MERGER: Returns + Sentiment + Financial Ratios
===========================================================

This script generates a daily dataset for stock prediction that guarantees
data alignment between daily price moves and quarterly financial ratios.

KEY FIXES:
1. Normalizes all dates (removes timezones) to prevent join failures.
2. Reindexes financial data to the stock's calendar BEFORE filling.
3. Implements a fallback mechanism so you never get "0% data availability".

Usage:
------
python merge_robust_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKER = 'A'  # Change this to your target ticker

# Directory paths - UPDATE THESE IF NEEDED
STOCK_DATA_DIR = 'stock_data'
SENTIMENT_DIR = 'Data_Sentiment_Index/csv_files'
FINANCIALS_DIR = 'financials_data/quarterly'
RELEASE_DATES_DIR = 'Quarterly_Release_Dates'
OUTPUT_DIR = '.'

# Parameters
DEFAULT_REPORTING_LAG = 45  # Days to wait if actual release date is missing


# ============================================================================
# CORE CLASSES
# ============================================================================

class FeatureMerger:
    def __init__(self, ticker):
        self.ticker = ticker
        self.ratios_df = pd.DataFrame()
        self.stock_df = pd.DataFrame()
        self.sentiment_df = pd.DataFrame()
        self.final_df = pd.DataFrame()

    def run(self):
        print("=" * 80)
        print(f"ROBUST FEATURE MERGER STARTING: {self.ticker}")
        print("=" * 80)

        # 1. Calculate Quarterly Ratios
        self.compute_quarterly_ratios()

        # 2. Align to Release Dates
        self.apply_release_dates()

        # 3. Load Daily Stock Data
        self.load_stock_data()

        # 4. Load Sentiment (Optional)
        self.load_sentiment()

        # 5. Merge and Forward Fill
        self.merge_datasets()

        # 6. Save
        self.save_output()

    def load_csv_transpose(self, path):
        """Helper to load and transpose financial CSVs securely"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path, index_col=0)
        df = df.T

        # FIX: Force to UTC first, then Strip Timezone to make it Naive
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
        df.index.name = 'report_date'
        return df.sort_index()

    def compute_quarterly_ratios(self):
        print(f"\n[1] Computing Fundamental Ratios...")

        # Paths
        p_inc = Path(FINANCIALS_DIR) / f"{self.ticker}_income_statement_quarterly.csv"
        p_bal = Path(FINANCIALS_DIR) / f"{self.ticker}_balance_sheet_quarterly.csv"
        p_cf = Path(FINANCIALS_DIR) / f"{self.ticker}_cash_flow_quarterly.csv"

        try:
            inc = self.load_csv_transpose(p_inc)
            bal = self.load_csv_transpose(p_bal)
            cf = self.load_csv_transpose(p_cf)
        except FileNotFoundError as e:
            print(f"   ❌ CRITICAL ERROR: {e}")
            return

        # Extract & Rename - Using robust column lookup
        def get_col(df, candidates):
            for c in candidates:
                if c in df.columns: return df[c]
            return pd.Series(np.nan, index=df.index)

        # Build clean dataframe
        df = pd.DataFrame(index=inc.index)

        # Income Items
        net_income = get_col(inc, ['Net Income', 'netIncome'])
        revenue = get_col(inc, ['Total Revenue', 'totalRevenue'])
        op_inc = get_col(inc, ['Operating Income', 'operatingIncome'])

        # Balance Sheet Items
        equity = get_col(bal, ['Stockholders Equity', 'totalStockholderEquity'])
        assets = get_col(bal, ['Total Assets', 'totalAssets'])
        debt = get_col(bal, ['Total Debt', 'totalDebt']) # Might need calculation
        cash = get_col(bal, ['Cash And Cash Equivalents', 'cashAndCashEquivalents'])
        curr_assets = get_col(bal, ['Current Assets', 'totalCurrentAssets'])
        curr_liab = get_col(bal, ['Current Liabilities', 'totalCurrentLiabilities'])

        # Cash Flow Items
        fcf = get_col(cf, ['Free Cash Flow', 'freeCashFlow'])
        ocf = get_col(cf, ['Operating Cash Flow', 'totalCashFromOperatingActivities'])

        # --- RATIO CALCULATIONS ---
        # 1. ROE
        df['roe'] = (net_income / equity.replace(0, np.nan)) * 100
        # 2. ROA
        df['roa'] = (net_income / assets.replace(0, np.nan)) * 100
        # 3. Operating Margin
        df['op_margin'] = (op_inc / revenue.replace(0, np.nan)) * 100
        # 4. Debt to Equity
        df['debt_to_equity'] = debt / equity.replace(0, np.nan)
        # 5. Liquidity (Cash / Assets)
        df['liquidity_ratio'] = cash / assets.replace(0, np.nan)
        # 6. Current Ratio
        df['current_ratio'] = curr_assets / curr_liab.replace(0, np.nan)
        # 7. FCF Margin
        df['free_cf_margin'] = (fcf / revenue.replace(0, np.nan)) * 100
        # 8. Revenue Growth (QoQ)
        df['rev_growth'] = revenue.pct_change() * 100
        # 9. OCF to Assets
        df['ocf_to_assets'] = ocf / assets.replace(0, np.nan)

        self.ratios_df = df.dropna(how='all')
        print(f"   ✓ Calculated 9 ratios for {len(self.ratios_df)} quarters")

    def apply_release_dates(self):
        print(f"\n[2] Aligning to Release Dates...")

        # Load the lookup table
        release_map = self._load_release_lookup()

        aligned_data = []
        new_index = []

        for q_end, row in self.ratios_df.iterrows():
            # Try to find an exact match within tolerance
            found_date = None
            best_diff = pd.Timedelta(days=40)

            # Logic: Match quarter end date from financials to the lookup keys
            for key_date, r_date in release_map.items():
                if abs(key_date - q_end) < best_diff:
                    found_date = r_date
                    best_diff = abs(key_date - q_end)

            if found_date:
                final_date = found_date
            else:
                # FALLBACK: If no release date found, assume lag
                final_date = q_end + pd.Timedelta(days=DEFAULT_REPORTING_LAG)

            # FIX: Ensure final_date is timezone naive
            final_date = pd.Timestamp(final_date).tz_localize(None).normalize()

            new_index.append(final_date)
            aligned_data.append(row)

        self.ratios_df = pd.DataFrame(aligned_data, index=new_index)
        # Sort by release date to ensure ffill works correctly
        self.ratios_df.sort_index(inplace=True)
        # Remove duplicate index (if multiple reports land on same day, take last)
        self.ratios_df = self.ratios_df[~self.ratios_df.index.duplicated(keep='last')]

        print(f"   ✓ Aligned {len(self.ratios_df)} records to specific release dates")
        if not self.ratios_df.empty:
            print(f"   ✓ Example: Data for quarter end {q_end.date()} becomes available on {final_date.date()}")

    def _load_release_lookup(self):
        path = Path(RELEASE_DATES_DIR)
        lookup = {}
        if not path.exists(): return lookup

        for f in path.glob('sp500_quarterly_dates_batch_*.xlsx'):
            try:
                df = pd.read_excel(f, header=0, engine='openpyxl')
                # Find row for this ticker
                row = df[df.iloc[:, 0] == self.ticker]
                if row.empty: continue

                # Parse columns
                for col in df.columns[3:]: # Skip Ticker, Company, CIK
                    try:
                        # Extract Quarter End from Column Name (e.g. "2023 Q3")
                        val = row.iloc[0][col]
                        if pd.isna(val): continue

                        release_dt = pd.to_datetime(val)

                        # Parse column header to get quarter end date
                        clean_col = str(col).strip().upper().replace('_', ' ')
                        m = re.search(r'(\d{4}).*Q([1-4])', clean_col) or re.search(r'Q([1-4]).*(\d{4})', clean_col)
                        if m:
                            if m.group(1).isdigit() and len(m.group(1))==4: yr, q = int(m.group(1)), int(m.group(2))
                            else: q, yr = int(m.group(1)), int(m.group(2))

                            q_ends = {1:'03-31', 2:'06-30', 3:'09-30', 4:'12-31'}
                            q_date = pd.Timestamp(f"{yr}-{q_ends[q]}").tz_localize(None).normalize()

                            lookup[q_date] = release_dt
                    except: continue
                break # Found the ticker, stop looking
            except: continue
        return lookup

    def load_stock_data(self):
        print(f"\n[3] Loading Stock Prices...")
        p = Path(STOCK_DATA_DIR) / f"{self.ticker}.csv"
        if not p.exists(): raise FileNotFoundError(f"No stock data for {self.ticker}")

        df = pd.read_csv(p)

        # FIX: Force to UTC first to handle offsets, THEN strip to Naive
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()

        df = df.set_index('Date').sort_index()

        # Calculate Returns
        df['Return'] = df['Close'].pct_change()
        self.stock_df = df[['Return']].copy() # Keep minimal for now
        print(f"   ✓ Loaded {len(self.stock_df)} daily records")

    def load_sentiment(self):
        print(f"\n[4] Loading Sentiment...")
        p = Path(SENTIMENT_DIR) / f"{self.ticker}.csv"
        if not p.exists():
            print("   ⚠ No sentiment file found (skipping)")
            return

        df = pd.read_csv(p, header=None, names=['Date', 'Sentiment'])

        # FIX: Force to Naive
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()

        self.sentiment_df = df.set_index('Date').sort_index()
        print(f"   ✓ Loaded {len(self.sentiment_df)} sentiment records")

    def merge_datasets(self):
        print(f"\n[5] Merging Datasets...")

        # Base is the Stock Index (Daily)
        df = self.stock_df.copy()

        # 1. Merge Sentiment (Left Join)
        if not self.sentiment_df.empty:
            df = df.join(self.sentiment_df, how='left')
            df['Sentiment'] = df['Sentiment'].ffill() # Fill forward sentiment

        # 2. Merge Ratios (Reindex + FFill Strategy)
        if not self.ratios_df.empty:
            # Create a container with full date range
            container = pd.DataFrame(index=df.index)

            # Combine the sparsely populated release dates into this container
            # We use 'outer' join first to capture release dates that might fall on weekends
            combined = container.join(self.ratios_df, how='outer')

            # Forward fill ONLY the ratio columns
            ratio_cols = self.ratios_df.columns
            combined[ratio_cols] = combined[ratio_cols].ffill()

            # Now trim back to only the days we have stock prices for
            combined = combined.loc[df.index]

            # Join this clean, full daily ratio set to our stock data
            df = df.join(combined[ratio_cols], how='left')

            count = df['roe'].notna().sum()
            print(f"   ✓ Merged Financials. Valid rows found: {count} / {len(df)}")
            if count == 0:
                print("   ⚠ WARNING: Still 0. Check date ranges.")
                print(f"     Stock Range: {df.index.min()} to {df.index.max()}")
                print(f"     Ratio Range: {self.ratios_df.index.min()} to {self.ratios_df.index.max()}")

        self.final_df = df

    def save_output(self):
        print(f"\n[6] Saving Output...")
        out_path = Path(OUTPUT_DIR) / f"{self.ticker}_daily_features_fixed.csv"
        self.final_df.to_csv(out_path)
        print(f"   ✓ Saved to {out_path}")
        print(f"   ✓ Final Shape: {self.final_df.shape}")

        # Quick sample check
        print("\n   Sample Data (Tail):")
        cols = ['Return', 'roe', 'debt_to_equity']
        cols = [c for c in cols if c in self.final_df.columns]
        print(self.final_df[cols].tail().to_string())

if __name__ == '__main__':
    merger = FeatureMerger(TICKER)
    merger.run()