"""
Quarterly Fundamental Ratios Calculator
========================================

This script processes quarterly financial statements and computes fundamental ratios
matching the SQL query logic for stock prediction models.

Usage:
------
1. Place your financial statement CSV files in the same directory or specify the path
2. Update the TICKER variable below
3. Run the script: python compute_quarterly_ratios.py

Input Files Required:
--------------------
- {TICKER}_income_statement_quarterly.csv
- {TICKER}_balance_sheet_quarterly.csv
- {TICKER}_cash_flow_quarterly.csv

Output:
-------
- {TICKER}_quarterly_fundamental_ratios.csv

Ratios Computed:
----------------
- ROE (Return on Equity)
- ROA (Return on Assets)
- Operating Margin
- Debt to Equity
- Liquidity Ratio
- Current Ratio
- Free Cash Flow Margin
- Operating Cash Flow to Assets
- Revenue Growth (QoQ)
- Market Cap (if stock prices provided)
- Log Market Cap
- Book to Market
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

TICKER = 'AAPL'  # Change this to your stock ticker

# Directory paths
FINANCIALS_DIR = 'financials_data/quarterly'  # Where your financial CSVs are located
STOCK_DATA_DIR = 'stock_data'  # Where your stock price CSVs are located (optional)
OUTPUT_DIR = '.'  # Where to save the output CSV

# Features
INCLUDE_MARKET_METRICS = False  # Set to True if you have stock price data
USE_RELEASE_DATES = False  # Set to True if you have release date data

# Release dates configuration (only if USE_RELEASE_DATES = True)
RELEASE_DATES_DIR = 'Quarterly_Release_Dates'
FISCAL_QUARTER_TOLERANCE = 40  # Days tolerance for matching fiscal quarters


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_financial_statement(filepath):
    """
    Load and transpose financial statement CSV files.
    Files should have metrics as rows and dates as columns.
    """
    print(f"   Loading {filepath.name}...")
    df = pd.read_csv(filepath, index_col=0)

    # Transpose: dates become rows, metrics become columns
    df = df.T

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    df.index.name = 'report_date'

    # Sort by date
    df = df.sort_index()

    print(f"   ✓ Loaded {len(df)} quarters")
    return df


def extract_financial_metrics(income_df, balance_df, cashflow_df):
    """
    Extract key metrics from financial statements and merge into single dataframe.
    """
    # Income Statement metrics
    income_metrics = income_df[[
        'Net Income',
        'Total Revenue',
        'Operating Income'
    ]].copy()
    income_metrics.columns = ['net_income', 'total_revenue', 'operating_income']

    # Balance Sheet metrics
    balance_metrics = balance_df[[
        'Total Assets',
        'Stockholders Equity',
        'Total Debt',
        'Current Assets',
        'Current Liabilities',
        'Cash And Cash Equivalents',
        'Ordinary Shares Number'
    ]].copy()
    balance_metrics.columns = [
        'total_assets',
        'total_equity',
        'total_debt',
        'current_assets',
        'current_liabilities',
        'cash_equivalents',
        'shares_outstanding'
    ]

    # Cash Flow metrics
    cashflow_metrics = cashflow_df[[
        'Free Cash Flow',
        'Operating Cash Flow'
    ]].copy()
    cashflow_metrics.columns = ['free_cash_flow', 'operating_cash_flow']

    # Merge all metrics
    fundamentals = income_metrics.join(balance_metrics, how='outer')
    fundamentals = fundamentals.join(cashflow_metrics, how='outer')

    # Fill forward any missing values
    fundamentals = fundamentals.ffill()

    return fundamentals


def compute_fundamental_ratios(fundamentals_df):
    """
    Compute all fundamental ratios from raw financial metrics.
    Matches the SQL query logic.
    """
    df = fundamentals_df.copy()

    # ROE: (Net Income / Total Equity) * 100
    df['roe'] = (df['net_income'] / df['total_equity'].replace(0, np.nan)) * 100

    # ROA: (Net Income / Total Assets) * 100
    df['roa'] = (df['net_income'] / df['total_assets'].replace(0, np.nan)) * 100

    # Operating Margin: (Operating Income / Total Revenue) * 100
    df['op_margin'] = (df['operating_income'] / df['total_revenue'].replace(0, np.nan)) * 100

    # Debt to Equity: Total Debt / Total Equity
    df['debt_to_equity'] = df['total_debt'] / df['total_equity'].replace(0, np.nan)

    # Liquidity Ratio: Cash / Total Assets
    df['liquidity_ratio'] = df['cash_equivalents'] / df['total_assets'].replace(0, np.nan)

    # Current Ratio: Current Assets / Current Liabilities
    df['current_ratio'] = df['current_assets'] / df['current_liabilities'].replace(0, np.nan)

    # Free Cash Flow Margin: (Free Cash Flow / Total Revenue) * 100
    df['free_cf_margin'] = (df['free_cash_flow'] / df['total_revenue'].replace(0, np.nan)) * 100

    # Operating Cash Flow to Assets: Operating Cash Flow / Total Assets
    df['ocf_to_assets'] = df['operating_cash_flow'] / df['total_assets'].replace(0, np.nan)

    # Revenue Growth (quarter over quarter): ((Current - Previous) / Previous) * 100
    df['prev_revenue'] = df['total_revenue'].shift(1)
    df['revenue_growth'] = ((df['total_revenue'] - df['prev_revenue']) / df['prev_revenue'].replace(0, np.nan)) * 100

    return df


def add_market_metrics(fundamentals_df, stock_prices_df):
    """
    Add market-based metrics that require stock price data.
    Stock prices are matched to quarter end dates.
    """
    df = fundamentals_df.copy()

    # For each quarter end date, find the closest stock price
    df['close_q'] = np.nan

    for date in df.index:
        # Find closest stock price (within 5 days of quarter end)
        try:
            # Look for prices on or after quarter end date
            valid_prices = stock_prices_df[
                (stock_prices_df.index >= date) &
                (stock_prices_df.index <= date + pd.Timedelta(days=5))
                ]

            if not valid_prices.empty:
                # Use the first available price after quarter end
                df.loc[date, 'close_q'] = valid_prices['Close'].iloc[0]
        except (IndexError, KeyError):
            pass

    # Market Cap: Close Price * Shares Outstanding
    df['market_cap'] = df['close_q'] * df['shares_outstanding']

    # Log Market Cap
    df['log_mktcap'] = np.log(df['market_cap'].replace(0, np.nan))

    # Book to Market: Total Equity / Market Cap
    df['book_to_market'] = df['total_equity'] / df['market_cap'].replace(0, np.nan)

    return df


def add_quarter_info(df):
    """
    Add year and quarter columns.
    """
    df = df.copy()
    df['yr'] = df.index.year
    df['qtr'] = df.index.quarter
    df['quarter_end_date'] = df.index

    return df


def load_release_dates(ticker):
    """
    Load release dates from Excel files (optional).
    Only used if USE_RELEASE_DATES = True.
    """
    import re

    def quarter_label_to_end_date(label):
        """Convert quarter label to end date"""
        label = str(label).strip().upper().replace('_', ' ')
        m = re.search(r'(\d{4})[^\d]*Q([1-4])', label)
        if not m:
            m = re.search(r'Q([1-4])[^\d]*(\d{4})', label)
            if m:
                q, yr = int(m.group(1)), int(m.group(2))
            else:
                return None
        else:
            yr, q = int(m.group(1)), int(m.group(2))

        quarter_ends = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}
        return pd.Timestamp(f'{yr}-{quarter_ends[q]}')

    path = Path(RELEASE_DATES_DIR)
    if not path.exists():
        return {}

    all_files = sorted(path.glob('sp500_quarterly_dates_batch_*.xlsx'))

    for fpath in all_files:
        try:
            df = pd.read_excel(fpath, header=0, engine='openpyxl')
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)

            if len(cols) >= 3:
                rename = {cols[0]: 'Ticker', cols[1]: 'Company', cols[2]: 'CIK'}
                df.rename(columns=rename, inplace=True)
                df['Ticker'] = df['Ticker'].astype(str).str.strip()

                ticker_row = df[df['Ticker'] == ticker]
                if len(ticker_row) > 0:
                    quarter_cols = [c for c in df.columns if c not in ('Ticker', 'Company', 'CIK')]
                    ticker_map = {}

                    for qcol in quarter_cols:
                        release_val = ticker_row.iloc[0][qcol]
                        if pd.isna(release_val):
                            continue

                        try:
                            release_date = pd.to_datetime(release_val)
                            qend = quarter_label_to_end_date(str(qcol))
                            if qend:
                                ticker_map[qend] = release_date
                        except Exception:
                            continue

                    if ticker_map:
                        return ticker_map
        except Exception:
            continue

    return {}


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_ticker():
    """
    Main function to process a single ticker and compute all fundamental ratios.
    """
    print("=" * 80)
    print(f"QUARTERLY FUNDAMENTAL RATIOS CALCULATOR")
    print("=" * 80)
    print(f"\nProcessing Ticker: {TICKER}")
    print("-" * 80)

    # 1. Load financial statements
    print("\n[1] Loading Financial Statements")
    print("-" * 80)

    financials_path = Path(FINANCIALS_DIR)

    income_path = financials_path / f"{TICKER}_income_statement_quarterly.csv"
    balance_path = financials_path / f"{TICKER}_balance_sheet_quarterly.csv"
    cashflow_path = financials_path / f"{TICKER}_cash_flow_quarterly.csv"

    # Check if files exist
    if not income_path.exists():
        raise FileNotFoundError(f"Income statement not found: {income_path}")
    if not balance_path.exists():
        raise FileNotFoundError(f"Balance sheet not found: {balance_path}")
    if not cashflow_path.exists():
        raise FileNotFoundError(f"Cash flow statement not found: {cashflow_path}")

    # Load the data
    income_df = load_financial_statement(income_path)
    balance_df = load_financial_statement(balance_path)
    cashflow_df = load_financial_statement(cashflow_path)

    # 2. Extract and merge financial metrics
    print("\n[2] Extracting Financial Metrics")
    print("-" * 80)
    fundamentals = extract_financial_metrics(income_df, balance_df, cashflow_df)
    print(f"   ✓ Merged data for {len(fundamentals)} quarters")

    # 3. Compute fundamental ratios
    print("\n[3] Computing Fundamental Ratios")
    print("-" * 80)
    fundamentals = compute_fundamental_ratios(fundamentals)

    ratios_computed = [
        'roe', 'roa', 'op_margin', 'debt_to_equity', 'liquidity_ratio',
        'current_ratio', 'free_cf_margin', 'revenue_growth', 'ocf_to_assets'
    ]
    print(f"   ✓ Computed {len(ratios_computed)} fundamental ratios")
    for ratio in ratios_computed:
        print(f"      - {ratio}")

    # 4. Add market metrics if requested
    if INCLUDE_MARKET_METRICS:
        print("\n[4] Adding Market-Based Metrics")
        print("-" * 80)
        stock_path = Path(STOCK_DATA_DIR) / f"{TICKER}.csv"

        if stock_path.exists():
            try:
                stock_df = pd.read_csv(stock_path, parse_dates=['Date'], index_col='Date')

                # Handle timezone if present
                if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
                    stock_df.index = stock_df.index.tz_localize(None)

                fundamentals = add_market_metrics(fundamentals, stock_df)
                print(f"   ✓ Added market metrics (market_cap, log_mktcap, book_to_market)")
                print(f"   ✓ Stock prices from {stock_df.index[0].date()} to {stock_df.index[-1].date()}")
            except Exception as e:
                print(f"   ✗ Could not add market metrics: {e}")
                print(f"   → Continuing without market metrics")
        else:
            print(f"   ✗ Stock price file not found: {stock_path}")
            print(f"   → Continuing without market metrics")

    # 5. Add release dates if requested
    if USE_RELEASE_DATES:
        print("\n[5] Loading Release Dates")
        print("-" * 80)
        release_dates = load_release_dates(TICKER)

        if release_dates:
            print(f"   ✓ Found {len(release_dates)} release dates")
            fundamentals['release_date'] = pd.NaT
            fundamentals['days_after_quarter'] = np.nan

            for quarter_end in fundamentals.index:
                for qend, rdate in release_dates.items():
                    if abs((quarter_end - qend).days) <= FISCAL_QUARTER_TOLERANCE:
                        fundamentals.loc[quarter_end, 'release_date'] = rdate
                        fundamentals.loc[quarter_end, 'days_after_quarter'] = (rdate - quarter_end).days
                        break
        else:
            print(f"   ✗ No release dates found for {TICKER}")

    # 6. Add quarter information
    fundamentals = add_quarter_info(fundamentals)

    # 7. Add ticker column
    fundamentals['ticker'] = TICKER

    # 8. Reorder columns
    column_order = [
        'ticker', 'yr', 'qtr', 'quarter_end_date',
    ]

    if USE_RELEASE_DATES and 'release_date' in fundamentals.columns:
        column_order.extend(['release_date', 'days_after_quarter'])

    column_order.extend([
        # Core ratios
        'roe', 'roa', 'op_margin', 'debt_to_equity', 'liquidity_ratio',
        'current_ratio', 'free_cf_margin', 'revenue_growth', 'ocf_to_assets',
    ])

    if INCLUDE_MARKET_METRICS:
        column_order.extend(['market_cap', 'log_mktcap', 'book_to_market'])

    column_order.extend([
        # Raw fundamentals
        'net_income', 'total_revenue', 'operating_income',
        'total_assets', 'total_equity', 'total_debt',
        'current_assets', 'current_liabilities', 'cash_equivalents',
        'shares_outstanding', 'free_cash_flow', 'operating_cash_flow',
        'prev_revenue'
    ])

    if INCLUDE_MARKET_METRICS and 'close_q' in fundamentals.columns:
        column_order.append('close_q')

    # Only include columns that exist
    available_cols = [col for col in column_order if col in fundamentals.columns]
    fundamentals = fundamentals[available_cols]

    # 9. Save to CSV
    print("\n[6] Saving Output")
    print("-" * 80)
    output_path = Path(OUTPUT_DIR) / f"{TICKER}_quarterly_fundamental_ratios.csv"
    fundamentals.to_csv(output_path, index=False)
    print(f"   ✓ Saved to: {output_path}")

    # 10. Display summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Ticker: {TICKER}")
    print(f"Total Quarters: {len(fundamentals)}")
    print(
        f"Date Range: {fundamentals['quarter_end_date'].min().date()} to {fundamentals['quarter_end_date'].max().date()}")
    print(f"Total Columns: {len(fundamentals.columns)}")
    print(f"Output File: {output_path}")

    # 11. Display sample data
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT (First 3 Quarters)")
    print("=" * 80)

    display_cols = [
        'ticker', 'yr', 'qtr', 'total_revenue', 'net_income',
        'roe', 'roa', 'op_margin', 'revenue_growth', 'debt_to_equity'
    ]
    available_display = [c for c in display_cols if c in fundamentals.columns]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format',
                  lambda x: f'{x:,.2f}' if pd.notna(x) and abs(x) > 0.01 else f'{x:.4f}' if pd.notna(x) else 'NaN')

    print(fundamentals[available_display].head(3).to_string(index=False))

    # 12. Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    stats_cols = ['roe', 'roa', 'op_margin', 'revenue_growth', 'debt_to_equity',
                  'liquidity_ratio', 'current_ratio']
    available_stats = [col for col in stats_cols if col in fundamentals.columns]

    print("\nMean Values:")
    print(fundamentals[available_stats].mean().to_string())

    print("\nStd Deviation:")
    print(fundamentals[available_stats].std().to_string())

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)

    return fundamentals


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == '__main__':
    try:
        result = process_ticker()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the following files exist:")
        print(f"   - {FINANCIALS_DIR}/{TICKER}_income_statement_quarterly.csv")
        print(f"   - {FINANCIALS_DIR}/{TICKER}_balance_sheet_quarterly.csv")
        print(f"   - {FINANCIALS_DIR}/{TICKER}_cash_flow_quarterly.csv")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()