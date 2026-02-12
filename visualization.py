"""
Data Alignment Verification Tool
==================================
This tool helps you visualize and verify how release dates work:
1. Shows when financial data becomes available vs when it was reported
2. Shows when macro data (GDP, IPI, Unemployment) becomes available
3. Displays timeline visualizations
4. Exports detailed CSV reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import functions from main script
import sys

sys.path.insert(0, '/home/claude')
from decisiontreexgboost_with_macro_releases import (
    load_release_date_lookup,
    load_macro_release_dates,
    _quarter_label_to_end_date
)


class DataAlignmentVerifier:
    """Verify and visualize release date alignment"""

    def __init__(self, ticker='NVDA'):
        self.ticker = ticker
        self.release_lookup = None
        self.macro_releases = None

    def load_all_release_data(self):
        """Load all release date information"""
        print("=" * 80)
        print(f"LOADING RELEASE DATE DATA FOR {self.ticker}")
        print("=" * 80)

        # Load financial release dates
        print("\n1. Loading quarterly financial release dates...")
        self.release_lookup = load_release_date_lookup()

        if self.ticker in self.release_lookup:
            print(f"   ✓ Found {len(self.release_lookup[self.ticker])} quarters for {self.ticker}")
        else:
            print(f"   ✗ No release dates found for {self.ticker}")

        # Load macro release dates
        print("\n2. Loading macroeconomic release dates...")
        self.macro_releases = load_macro_release_dates()

        for key, df in self.macro_releases.items():
            if not df.empty:
                print(f"   ✓ {key.upper()}: {len(df)} releases")
            else:
                print(f"   ✗ {key.upper()}: No data")

    def verify_financial_alignment(self, save_csv=True):
        """Verify financial data alignment for the ticker"""

        if self.ticker not in self.release_lookup:
            print(f"\n❌ No financial release dates for {self.ticker}")
            return None

        print(f"\n{'=' * 80}")
        print(f"FINANCIAL DATA ALIGNMENT FOR {self.ticker}")
        print(f"{'=' * 80}\n")

        # Load raw financial data
        try:
            income_stmt = pd.read_csv(
                f'financials_data/quarterly/{self.ticker}_income_statement_quarterly.csv'
            )
        except:
            print(f"❌ Could not load income statement for {self.ticker}")
            return None

        # Detect layout and parse dates
        first_col = income_stmt.columns[0]
        first_vals = income_stmt[first_col].dropna().astype(str).head(5).tolist()
        looks_like_dates = sum(
            1 for v in first_vals
            if any(c.isdigit() for c in v) and ('-' in v or '/' in v)
        )

        if looks_like_dates < len(first_vals) // 2:
            # Transposed layout
            income_stmt = income_stmt.set_index(first_col)
            valid_cols = []
            for c in income_stmt.columns:
                try:
                    pd.to_datetime(str(c))
                    valid_cols.append(c)
                except:
                    pass
            income_stmt = income_stmt[valid_cols].T
            income_stmt.index = pd.to_datetime(income_stmt.index)
        else:
            # Standard layout
            date_col = first_col
            for cand in ['Date', 'date', 'Period', 'period']:
                if cand in income_stmt.columns:
                    date_col = cand
                    break
            income_stmt[date_col] = pd.to_datetime(income_stmt[date_col])
            income_stmt = income_stmt.set_index(date_col)

        income_stmt = income_stmt.sort_index()

        # Build alignment report
        ticker_releases = self.release_lookup[self.ticker]
        alignment_data = []

        for qend, row in income_stmt.iterrows():
            # Find matching release date
            best_match = None
            best_diff = pd.Timedelta(days=10)

            for lookup_qend in ticker_releases:
                diff = abs(qend - lookup_qend)
                if diff <= best_diff:
                    best_diff = diff
                    best_match = lookup_qend

            if best_match is not None:
                release_date = ticker_releases[best_match]
                days_delay = (release_date - qend).days

                alignment_data.append({
                    'Quarter_End': qend.strftime('%Y-%m-%d'),
                    'Release_Date': release_date.strftime('%Y-%m-%d'),
                    'Days_After_Quarter_End': days_delay,
                    'Matched_Quarter': best_match.strftime('%Y-%m-%d'),
                    'Match_Diff_Days': best_diff.days,
                    'Data_Available_From': release_date.strftime('%Y-%m-%d')
                })

        alignment_df = pd.DataFrame(alignment_data)

        if save_csv:
            filename = f'verification_{self.ticker}_financial_alignment.csv'
            alignment_df.to_csv(filename, index=False)
            print(f"✓ Saved alignment report: {filename}\n")

        # Print summary
        print(f"{'Quarter End':<15} {'Release Date':<15} {'Days Delay':<12} {'Status'}")
        print("-" * 65)

        for _, row in alignment_df.iterrows():
            qend = row['Quarter_End']
            release = row['Release_Date']
            delay = row['Days_After_Quarter_End']

            # Color-code by delay
            if delay < 30:
                status = "⚡ Fast"
            elif delay < 45:
                status = "✓ Normal"
            else:
                status = "⏱ Slow"

            print(f"{qend:<15} {release:<15} {delay:<12} {status}")

        print(f"\n📊 Summary Statistics:")
        print(f"   Average delay: {alignment_df['Days_After_Quarter_End'].mean():.1f} days")
        print(f"   Median delay:  {alignment_df['Days_After_Quarter_End'].median():.1f} days")
        print(f"   Min delay:     {alignment_df['Days_After_Quarter_End'].min()} days")
        print(f"   Max delay:     {alignment_df['Days_After_Quarter_End'].max()} days")

        return alignment_df

    def verify_macro_alignment(self, save_csv=True):
        """Verify macroeconomic data alignment"""

        print(f"\n{'=' * 80}")
        print(f"MACROECONOMIC DATA ALIGNMENT")
        print(f"{'=' * 80}\n")

        all_macro_reports = {}

        # GDP Alignment
        if 'gdp' in self.macro_releases and not self.macro_releases['gdp'].empty:
            print("📈 GDP RELEASE SCHEDULE:")
            print("-" * 70)

            gdp_df = self.macro_releases['gdp'].copy()
            gdp_df['Days_After_Quarter'] = (
                    gdp_df['release_date'] - gdp_df['observation_date']
            ).dt.days

            print(f"{'Quarter':<20} {'Quarter End':<15} {'Release Date':<15} {'Days Delay'}")
            print("-" * 70)

            for _, row in gdp_df.iterrows():
                quarter = row.get('Quarter', 'N/A')
                qend = row['observation_date'].strftime('%Y-%m-%d')
                release = row['release_date'].strftime('%Y-%m-%d')
                delay = row['Days_After_Quarter']

                print(f"{quarter:<20} {qend:<15} {release:<15} {delay}")

            print(f"\n   Average delay: {gdp_df['Days_After_Quarter'].mean():.1f} days")

            all_macro_reports['gdp'] = gdp_df

            if save_csv:
                gdp_df.to_csv('verification_GDP_alignment.csv', index=False)
                print(f"   ✓ Saved: verification_GDP_alignment.csv")

        # IPI Alignment
        if 'ipi' in self.macro_releases and not self.macro_releases['ipi'].empty:
            print("\n📊 IPI (Industrial Production) RELEASE SCHEDULE:")
            print("-" * 70)

            ipi_df = self.macro_releases['ipi'].copy()
            ipi_df['Days_After_Quarter'] = (
                    ipi_df['release_date'] - ipi_df['observation_date']
            ).dt.days

            print(f"{'Quarter':<20} {'Quarter End':<15} {'Release Date':<15} {'Days Delay'}")
            print("-" * 70)

            for _, row in ipi_df.iterrows():
                quarter = row.get('Quarter', 'N/A')
                qend = row['observation_date'].strftime('%Y-%m-%d')
                release = row['release_date'].strftime('%Y-%m-%d')
                delay = row['Days_After_Quarter']

                print(f"{quarter:<20} {qend:<15} {release:<15} {delay}")

            print(f"\n   Average delay: {ipi_df['Days_After_Quarter'].mean():.1f} days")

            all_macro_reports['ipi'] = ipi_df

            if save_csv:
                ipi_df.to_csv('verification_IPI_alignment.csv', index=False)
                print(f"   ✓ Saved: verification_IPI_alignment.csv")

        # Unemployment Alignment
        if 'unemployment' in self.macro_releases and not self.macro_releases['unemployment'].empty:
            print("\n💼 UNEMPLOYMENT RATE RELEASE SCHEDULE:")
            print("-" * 70)

            unemp_df = self.macro_releases['unemployment'].copy()
            unemp_df['Days_After_Month'] = (
                    unemp_df['release_date'] - unemp_df['observation_date']
            ).dt.days

            print(f"{'Month':<15} {'Release Date':<15} {'Days After Month'}")
            print("-" * 70)

            for _, row in unemp_df.tail(12).iterrows():  # Show last 12 months
                month = row['observation_date'].strftime('%Y-%m')
                release = row['release_date'].strftime('%Y-%m-%d')
                delay = row['Days_After_Month']

                print(f"{month:<15} {release:<15} {delay}")

            print(f"\n   Average delay: {unemp_df['Days_After_Month'].mean():.1f} days")
            print(f"   (Showing last 12 months, total: {len(unemp_df)} months)")

            all_macro_reports['unemployment'] = unemp_df

            if save_csv:
                unemp_df.to_csv('verification_Unemployment_alignment.csv', index=False)
                print(f"   ✓ Saved: verification_Unemployment_alignment.csv")

        return all_macro_reports

    def visualize_timeline(self, start_date='2024-01-01', end_date='2025-12-31'):
        """Create a visual timeline showing when data becomes available"""

        print(f"\n{'=' * 80}")
        print(f"CREATING TIMELINE VISUALIZATION")
        print(f"{'=' * 80}\n")

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f'Data Release Timeline: {self.ticker}', fontsize=14, fontweight='bold')

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # ─── Plot 1: Financial Data Releases ───
        ax1 = axes[0]
        ax1.set_title('Quarterly Financial Data Releases', fontsize=12, pad=10)

        if self.ticker in self.release_lookup:
            ticker_releases = self.release_lookup[self.ticker]

            for qend, release_date in ticker_releases.items():
                if start <= release_date <= end:
                    # Quarter end marker
                    ax1.axvline(qend, color='lightblue', alpha=0.3, linestyle='--', linewidth=1)

                    # Release date marker
                    ax1.axvline(release_date, color='darkblue', alpha=0.8, linewidth=2)

                    # Arrow showing the delay
                    delay_days = (release_date - qend).days
                    ax1.annotate('', xy=(release_date, 0.5), xytext=(qend, 0.5),
                                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

                    # Label
                    ax1.text(release_date, 0.6, f"{qend.strftime('%Y Q%q')}\n{delay_days}d",
                             ha='center', va='bottom', fontsize=8, rotation=45)

            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Financial\nData', fontsize=10)
            ax1.set_xlim(start, end)
            ax1.grid(True, alpha=0.3)
            ax1.legend(['Quarter End', 'Release Date'], loc='upper right', fontsize=8)

        # ─── Plot 2: GDP/IPI Releases ───
        ax2 = axes[1]
        ax2.set_title('GDP & IPI Release Schedule', fontsize=12, pad=10)

        if 'gdp' in self.macro_releases and not self.macro_releases['gdp'].empty:
            gdp_df = self.macro_releases['gdp']

            for _, row in gdp_df.iterrows():
                qend = row['observation_date']
                release = row['release_date']

                if start <= release <= end:
                    # Quarter end
                    ax2.axvline(qend, color='lightgreen', alpha=0.3, linestyle='--', linewidth=1)

                    # Release date
                    ax2.axvline(release, color='darkgreen', alpha=0.8, linewidth=2)

                    # Arrow
                    delay_days = (release - qend).days
                    ax2.annotate('', xy=(release, 0.5), xytext=(qend, 0.5),
                                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

                    # Label
                    quarter = row.get('Quarter', qend.strftime('%Y Q%q'))
                    ax2.text(release, 0.6, f"{quarter}\n{delay_days}d",
                             ha='center', va='bottom', fontsize=8, rotation=45)

            ax2.set_ylim(0, 1)
            ax2.set_ylabel('GDP/IPI', fontsize=10)
            ax2.set_xlim(start, end)
            ax2.grid(True, alpha=0.3)

        # ─── Plot 3: Unemployment Releases ───
        ax3 = axes[2]
        ax3.set_title('Unemployment Rate Release Schedule', fontsize=12, pad=10)

        if 'unemployment' in self.macro_releases and not self.macro_releases['unemployment'].empty:
            unemp_df = self.macro_releases['unemployment']

            for _, row in unemp_df.iterrows():
                obs_date = row['observation_date']
                release = row['release_date']

                if start <= release <= end:
                    # Observation month end
                    ax3.axvline(obs_date, color='lightcoral', alpha=0.3, linestyle='--', linewidth=1)

                    # Release date
                    ax3.axvline(release, color='darkred', alpha=0.8, linewidth=2)

                    # Arrow
                    delay_days = (release - obs_date).days
                    ax3.annotate('', xy=(release, 0.5), xytext=(obs_date, 0.5),
                                 arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

                    # Label (only show every other month to avoid crowding)
                    if obs_date.month % 2 == 0:
                        ax3.text(release, 0.6, f"{obs_date.strftime('%Y-%m')}\n{delay_days}d",
                                 ha='center', va='bottom', fontsize=7, rotation=45)

            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Unemployment', fontsize=10)
            ax3.set_xlim(start, end)
            ax3.set_xlabel('Date', fontsize=10)
            ax3.grid(True, alpha=0.3)

        # Format x-axes
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_yticks([])

        plt.tight_layout()

        filename = f'verification_{self.ticker}_timeline.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Timeline visualization saved: {filename}")
        plt.close()

        return filename

    def demonstrate_data_flow(self, sample_date='2024-07-15'):
        """
        Demonstrate what data is available on a specific date
        """
        print(f"\n{'=' * 80}")
        print(f"DATA AVAILABLE AS OF: {sample_date}")
        print(f"{'=' * 80}\n")

        check_date = pd.Timestamp(sample_date)

        print(f"🗓️  On {check_date.strftime('%Y-%m-%d')}, the following data is available:\n")

        # Financial data
        if self.ticker in self.release_lookup:
            print(f"📊 FINANCIAL DATA ({self.ticker}):")
            ticker_releases = self.release_lookup[self.ticker]

            available_quarters = []
            for qend, release_date in sorted(ticker_releases.items()):
                if release_date <= check_date:
                    available_quarters.append((qend, release_date))

            if available_quarters:
                latest_qend, latest_release = available_quarters[-1]
                days_ago = (check_date - latest_release).days

                print(f"   ✓ Latest available: {latest_qend.strftime('%Y Q%q')}")
                print(f"     (Quarter ended {latest_qend.strftime('%Y-%m-%d')}, ")
                print(f"      released {latest_release.strftime('%Y-%m-%d')}, {days_ago} days ago)")

                print(f"\n   Available quarters ({len(available_quarters)} total):")
                for qend, release in available_quarters[-5:]:  # Last 5
                    print(f"     • {qend.strftime('%Y Q%q')} (released {release.strftime('%Y-%m-%d')})")
            else:
                print("   ✗ No financial data available yet")

        # GDP data
        print(f"\n💰 GDP DATA:")
        if 'gdp' in self.macro_releases and not self.macro_releases['gdp'].empty:
            gdp_available = self.macro_releases['gdp'][
                self.macro_releases['gdp']['release_date'] <= check_date
                ]

            if not gdp_available.empty:
                latest = gdp_available.iloc[-1]
                quarter = latest.get('Quarter', latest['observation_date'].strftime('%Y Q%q'))
                days_ago = (check_date - latest['release_date']).days

                print(f"   ✓ Latest available: {quarter}")
                print(f"     (Released {latest['release_date'].strftime('%Y-%m-%d')}, {days_ago} days ago)")
            else:
                print("   ✗ No GDP data available yet")

        # Unemployment data
        print(f"\n💼 UNEMPLOYMENT DATA:")
        if 'unemployment' in self.macro_releases and not self.macro_releases['unemployment'].empty:
            unemp_available = self.macro_releases['unemployment'][
                self.macro_releases['unemployment']['release_date'] <= check_date
                ]

            if not unemp_available.empty:
                latest = unemp_available.iloc[-1]
                days_ago = (check_date - latest['release_date']).days

                print(f"   ✓ Latest available: {latest['observation_date'].strftime('%Y-%m')}")
                print(f"     (Released {latest['release_date'].strftime('%Y-%m-%d')}, {days_ago} days ago)")
            else:
                print("   ✗ No unemployment data available yet")

    def run_full_verification(self):
        """Run all verification steps"""
        print("\n" + "=" * 80)
        print("COMPLETE DATA ALIGNMENT VERIFICATION")
        print("=" * 80)

        # Load data
        self.load_all_release_data()

        # Verify financial alignment
        self.verify_financial_alignment(save_csv=True)

        # Verify macro alignment
        self.verify_macro_alignment(save_csv=True)

        # Create timeline
        self.visualize_timeline()

        # Demonstrate data flow
        self.demonstrate_data_flow('2024-07-15')
        self.demonstrate_data_flow('2024-10-31')

        print("\n" + "=" * 80)
        print("✅ VERIFICATION COMPLETE")
        print("=" * 80)
        print("\nGenerated files:")
        print(f"  • verification_{self.ticker}_financial_alignment.csv")
        print(f"  • verification_GDP_alignment.csv")
        print(f"  • verification_IPI_alignment.csv")
        print(f"  • verification_Unemployment_alignment.csv")
        print(f"  • verification_{self.ticker}_timeline.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Example: Verify NVDA
    verifier = DataAlignmentVerifier(ticker='NVDA')
    verifier.run_full_verification()

    print("\n" + "=" * 80)
    print("You can also verify other tickers:")
    print("=" * 80)
    print("""
# Example for another ticker:
verifier = DataAlignmentVerifier(ticker='AAPL')
verifier.run_full_verification()

# Or just verify specific aspects:
verifier = DataAlignmentVerifier(ticker='TSLA')
verifier.load_all_release_data()
verifier.verify_financial_alignment()
verifier.visualize_timeline(start_date='2023-01-01', end_date='2025-12-31')
verifier.demonstrate_data_flow('2024-08-01')
    """)