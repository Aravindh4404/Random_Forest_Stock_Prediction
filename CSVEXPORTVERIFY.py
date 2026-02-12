"""
Verification Script: Check Release Date Alignment
Shows exactly when financial/macro data appears in features for a single stock
EXPORTS ALL RESULTS TO CSV FILES
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
TICKER = 'AAPL'  # Change this to test different stocks
QUARTERLY_DATES_DIR = 'Quarterly_Release_Dates'
MACRO_RELEASE_DIR = 'Macroeconomic_Data'
FISCAL_QUARTER_TOLERANCE = 40


# ─────────────────────────────────────────────────────────────────────────────


def _quarter_label_to_end_date(label):
    """Convert quarter label to end date"""
    label = str(label).strip().upper().replace('_', ' ')
    import re
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


def load_release_dates_for_ticker(ticker, directory=QUARTERLY_DATES_DIR):
    """Load release dates for a specific ticker"""
    path = Path(directory)
    if not path.exists():
        print(f"❌ Directory not found: {directory}")
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
                            qend = _quarter_label_to_end_date(str(qcol))
                            ticker_map[qend] = release_date
                        except Exception:
                            continue

                    if ticker_map:
                        return ticker_map
        except Exception as e:
            continue

    return {}


def load_financial_data(ticker):
    """Load financial statements"""
    financials = {}

    for name, path in [
        ('income_statement', f'financials_data/quarterly/{ticker}_income_statement_quarterly.csv'),
        ('balance_sheet', f'financials_data/quarterly/{ticker}_balance_sheet_quarterly.csv'),
        ('cash_flow', f'financials_data/quarterly/{ticker}_cash_flow_quarterly.csv')
    ]:
        try:
            if Path(path).exists():
                financials[name] = pd.read_csv(path)
            else:
                financials[name] = pd.DataFrame()
        except Exception:
            financials[name] = pd.DataFrame()

    return financials


def parse_financial_dates(fin_df):
    """Parse dates from financial dataframe"""
    if fin_df.empty:
        return pd.DataFrame()

    fin_df = fin_df.copy()
    first_col = fin_df.columns[0]
    first_vals = fin_df[first_col].dropna().astype(str).head(5).tolist()
    looks_like_dates = sum(1 for v in first_vals
                           if any(c.isdigit() for c in v) and ('-' in v or '/' in v))

    if looks_like_dates < len(first_vals) // 2:
        # Transposed layout
        fin_df = fin_df.set_index(first_col)
        valid_cols = [c for c in fin_df.columns if pd.to_datetime(str(c), errors='coerce') is not pd.NaT]
        if not valid_cols:
            return pd.DataFrame()
        fin_df = fin_df[valid_cols].T
        fin_df.index = pd.to_datetime(fin_df.index)
    else:
        # Standard layout
        date_col = first_col
        for cand in ['Date', 'date', 'Period', 'period', 'observation_date']:
            if cand in fin_df.columns:
                date_col = cand
                break
        fin_df[date_col] = pd.to_datetime(fin_df[date_col], errors='coerce')
        fin_df = fin_df.dropna(subset=[date_col])
        fin_df = fin_df.set_index(date_col)

    return fin_df.sort_index()


def verify_alignment(ticker):
    """Main verification function with CSV export"""
    print("=" * 80)
    print(f"RELEASE DATE ALIGNMENT VERIFICATION FOR: {ticker}")
    print("=" * 80)

    # Timestamp for output files
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # CSV exports list
    csv_exports = []

    # 1. Load release dates
    print("\n1. LOADING RELEASE DATES")
    print("-" * 80)
    release_dates = load_release_dates_for_ticker(ticker)

    if not release_dates:
        print(f"❌ No release dates found for {ticker}")
        return

    # Export 1: Release dates
    release_df = pd.DataFrame([
        {
            'ticker': ticker,
            'quarter_end': qend.date(),
            'release_date': rdate.date(),
            'days_after_quarter': (rdate - qend).days
        }
        for qend, rdate in sorted(release_dates.items())
    ])

    csv_file_1 = f'{ticker}_release_dates_{ts}.csv'
    release_df.to_csv(csv_file_1, index=False)
    csv_exports.append(csv_file_1)

    print(f"✓ Found {len(release_dates)} quarterly release dates")
    print(f"✓ Exported to: {csv_file_1}\n")
    print(release_df.to_string(index=False))

    # 2. Load and check financial data
    print("\n\n2. FINANCIAL DATA ALIGNMENT")
    print("-" * 80)
    financials = load_financial_data(ticker)

    all_matches = []

    for fin_type, fin_df in financials.items():
        if fin_df.empty:
            print(f"\n{fin_type.upper()}: No data found")
            continue

        print(f"\n{fin_type.upper()}:")
        print("-" * 80)

        parsed = parse_financial_dates(fin_df)
        if parsed.empty:
            print("  Could not parse dates")
            continue

        print(f"\n  Raw financial statement dates (first 5):")
        for date in parsed.index[:5]:
            print(f"    {date.date()}")

        # Match with release dates
        print(f"\n  {'Financial Quarter':<18} {'Release Date':<15} {'Match Type':<20} {'Days Diff':<10}")
        print("  " + "-" * 70)

        matched = []
        unmatched = []

        for qend in parsed.index:
            best_match = None
            best_diff = pd.Timedelta(days=FISCAL_QUARTER_TOLERANCE)

            for lookup_qend in release_dates:
                diff = abs(qend - lookup_qend)
                if diff <= best_diff:
                    best_diff = diff
                    best_match = lookup_qend

            if best_match is not None:
                release_date = release_dates[best_match]
                match_type = "Exact" if best_diff.days == 0 else f"Fiscal (±{best_diff.days}d)"

                match_record = {
                    'ticker': ticker,
                    'statement_type': fin_type,
                    'financial_quarter_end': qend.date(),
                    'matched_calendar_quarter': best_match.date(),
                    'release_date': release_date.date(),
                    'match_type': match_type,
                    'days_difference': best_diff.days,
                    'days_after_quarter': (release_date - qend).days
                }

                matched.append(match_record)
                all_matches.append(match_record)

                print(f"  {qend.date()!s:<18} {release_date.date()!s:<15} {match_type:<20} {best_diff.days:<10}")
            else:
                unmatched.append(qend)

        print(f"\n  Summary: {len(matched)} matched, {len(unmatched)} unmatched")

        if unmatched:
            print(f"  Unmatched quarters: {', '.join(str(d.date()) for d in unmatched[:5])}")

    # Export 2: Financial alignment
    if all_matches:
        matches_df = pd.DataFrame(all_matches)
        csv_file_2 = f'{ticker}_financial_alignment_{ts}.csv'
        matches_df.to_csv(csv_file_2, index=False)
        csv_exports.append(csv_file_2)
        print(f"\n✓ Exported financial alignment to: {csv_file_2}")

    # 3. Show example of when data appears in features
    print("\n\n3. FEATURE AVAILABILITY TIMELINE")
    print("-" * 80)

    timeline_records = []

    if financials['income_statement'].empty:
        print("No income statement data to demonstrate")
    else:
        print("\nExample: When does Q1 2024 revenue appear in daily features?")
        print("-" * 80)

        parsed = parse_financial_dates(financials['income_statement'])

        # Find Q1 2024 or latest quarter
        target_quarter = pd.Timestamp('2024-03-31')
        if target_quarter not in parsed.index:
            # Use most recent quarter
            target_quarter = parsed.index[-1]

        # Find matching release date
        best_match = None
        best_diff = pd.Timedelta(days=FISCAL_QUARTER_TOLERANCE)

        for lookup_qend in release_dates:
            diff = abs(target_quarter - lookup_qend)
            if diff <= best_diff:
                best_diff = diff
                best_match = lookup_qend

        if best_match:
            release_date = release_dates[best_match]

            print(f"\nQuarter End:        {target_quarter.date()}")
            print(f"Release Date:       {release_date.date()}")
            print(f"Days After Quarter: {(release_date - target_quarter).days}")

            print(f"\nTimeline:")
            print(f"  {target_quarter.date()}: Quarter ends (data NOT yet available)")
            print(f"  {release_date.date()}: Earnings released (data becomes available)")
            print(f"  {release_date.date()} onward: Features contain this quarter's data (forward-filled daily)")

            # Get sample metrics
            sample_cols = [c for c in parsed.columns if not c.startswith('Unnamed')][:3]

            for sample_col in sample_cols:
                sample_value = parsed.loc[target_quarter, sample_col]

                timeline_records.append({
                    'ticker': ticker,
                    'quarter_end': target_quarter.date(),
                    'release_date': release_date.date(),
                    'days_lag': (release_date - target_quarter).days,
                    'metric': sample_col,
                    'value': sample_value,
                    'available_from': release_date.date(),
                    'note': f"Data NOT available before {release_date.date()}"
                })

            if sample_cols:
                print(f"\nExample metrics from this quarter:")
                for col in sample_cols[:3]:
                    val = parsed.loc[target_quarter, col]
                    print(f"  {col}: {val}")
                    print(f"    Available in features starting: {release_date.date()}")

    # Export 3: Timeline
    if timeline_records:
        timeline_df = pd.DataFrame(timeline_records)
        csv_file_3 = f'{ticker}_feature_availability_{ts}.csv'
        timeline_df.to_csv(csv_file_3, index=False)
        csv_exports.append(csv_file_3)
        print(f"\n✓ Exported feature availability to: {csv_file_3}")

    # 4. Show comparison with stock price data
    print("\n\n4. VERIFICATION WITH STOCK DATA")
    print("-" * 80)

    price_records = []

    try:
        stock_data = pd.read_csv(f'stock_data/{ticker}.csv')
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None)
        stock_data.set_index('Date', inplace=True)

        if best_match:
            # Show stock prices around release date
            window_start = release_date - pd.Timedelta(days=3)
            window_end = release_date + pd.Timedelta(days=3)

            window_data = stock_data.loc[window_start:window_end, 'Close']

            print(f"\nStock prices around release date ({release_date.date()}):")
            print("-" * 60)
            print(f"{'Date':<15} {'Close Price':<15} {'Financial Data Available':<30}")
            print("-" * 60)

            for date, price in window_data.items():
                available = date >= release_date
                available_text = "YES ✓" if available else "NO ✗"
                marker = " <-- RELEASE DATE" if date.date() == release_date.date() else ""

                price_records.append({
                    'ticker': ticker,
                    'date': date.date(),
                    'close_price': price,
                    'quarter_end': target_quarter.date(),
                    'release_date': release_date.date(),
                    'financial_data_available': available,
                    'is_release_date': date.date() == release_date.date()
                })

                print(f"{date.date()!s:<15} ${price:<14.2f} {available_text:<30}{marker}")

        # Export 4: Price verification
        if price_records:
            price_df = pd.DataFrame(price_records)
            csv_file_4 = f'{ticker}_price_verification_{ts}.csv'
            price_df.to_csv(csv_file_4, index=False)
            csv_exports.append(csv_file_4)
            print(f"\n✓ Exported price verification to: {csv_file_4}")

    except Exception as e:
        print(f"Could not load stock data: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Exported {len(csv_exports)} CSV files:")
    for f in csv_exports:
        print(f"  - {f}")
    print("\nThese files contain:")
    print("  1. Release dates for each quarter")
    print("  2. Financial statement alignment with release dates")
    print("  3. Feature availability timeline")
    print("  4. Stock price verification around release dates")
    print("=" * 80)


if __name__ == '__main__':
    verify_alignment(TICKER)