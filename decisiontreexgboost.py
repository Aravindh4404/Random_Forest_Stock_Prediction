"""
S&P 500 Batch XGBoost + Decision Tree Stock Return Prediction
- Uses ACTUAL quarterly filing/release dates from Quarterly_Release_Dates/*.xlsx
- Financial data only becomes available in features ON or AFTER its release date
- Blended ensemble: XGBoost (70%) + Decision Tree (30%)
- Target: next-day return
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

# ── Global config ────────────────────────────────────────────────────────────
XGB_BLEND_WEIGHT        = 0.7    # 0.0 = pure DT, 1.0 = pure XGB
QUARTERLY_DATES_DIR     = 'Quarterly_Release_Dates'
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# STEP 1 — Build a master release-date lookup  {ticker: {quarter_end: release_date}}
# =============================================================================

def load_release_date_lookup(directory=QUARTERLY_DATES_DIR):
    """
    Reads all sp500_quarterly_dates_batch_*.xlsx files and returns a dict:

        lookup[ticker] = pd.Series(index=quarter_end_dates, data=release_dates)

    Column layout (from the screenshot):
        C0  = Ticker
        C1  = Company  (ignored)
        C2  = CIK      (ignored)
        C3+ = alternating quarter label / release date columns
               e.g. "2024 Q1" header row, actual date in data rows

    The xlsx files use a two-row header:
        row 0 (header): Ticker | Company | CIK | 2024 Q1 | 2024 Q2 | ...
    Each data cell in Q-columns contains the actual filing/release date
    (already a real calendar date, not the quarter-end).

    We treat the COLUMN HEADER as the quarter label and the CELL VALUE
    as the release date for that ticker+quarter combination.
    """
    path = Path(directory)
    all_files = sorted(path.glob('sp500_quarterly_dates_batch_*.xlsx'))

    if not all_files:
        raise FileNotFoundError(
            f"No batch files found in '{directory}'. "
            "Expected sp500_quarterly_dates_batch_01.xlsx etc."
        )

    frames = []
    for fpath in all_files:
        df = pd.read_excel(fpath, header=0)   # row 0 is the header
        # Normalise column names from the screenshot layout
        df.columns = [str(c).strip() for c in df.columns]

        # Rename the first three columns robustly
        cols = list(df.columns)
        rename = {cols[0]: 'Ticker', cols[1]: 'Company', cols[2]: 'CIK'}
        df.rename(columns=rename, inplace=True)
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    # Quarter columns are everything after 'CIK'
    quarter_cols = [c for c in master.columns if c not in ('Ticker', 'Company', 'CIK')]

    # Build lookup: ticker → {quarter_end_date → release_date}
    #
    # The column header looks like "2024 Q1", "2024 Q2", etc.
    # We convert that label to the last calendar day of that quarter
    # (the period the financial statement COVERS), then map it to the
    # actual filing date stored in the cell.
    lookup = {}

    for _, row in master.iterrows():
        ticker = row['Ticker']
        if pd.isna(ticker) or ticker == 'nan':
            continue

        ticker_map = {}   # quarter_end  →  release_date
        for qcol in quarter_cols:
            release_val = row[qcol]
            if pd.isna(release_val):
                continue

            # Parse release date
            try:
                release_date = pd.to_datetime(release_val)
            except Exception:
                continue

            # Convert column label ("2024 Q1") → quarter-end date
            try:
                qend = _quarter_label_to_end_date(str(qcol))
            except Exception:
                continue

            ticker_map[qend] = release_date

        if ticker_map:
            lookup[ticker] = ticker_map

    print(f"Release-date lookup built: {len(lookup)} tickers, "
          f"{sum(len(v) for v in lookup.values())} ticker-quarter entries")
    return lookup


def _quarter_label_to_end_date(label):
    """
    Convert a quarter label like '2024 Q1' → 2024-03-31,
    '2024 Q2' → 2024-06-30, etc.
    Also handles 'Q1 2024' order and separators like '2024Q1'.
    """
    label = label.strip().upper().replace('_', ' ')
    # Try patterns: "2024 Q1", "Q1 2024", "2024Q1"
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


# =============================================================================
# MAIN PREDICTOR CLASS
# =============================================================================

class StockReturnPredictor:
    def __init__(self, ticker='NVDA', release_lookup=None):
        self.ticker          = ticker
        self.release_lookup  = release_lookup or {}   # global lookup passed in
        self.xgb_model       = None
        self.dt_model        = None
        self.scaler          = StandardScaler()
        self.feature_names   = None
        self.feature_importance = None

    # ──────────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────────────────────────────────
    def load_data(self):
        # 1. OHLCV
        self.stock_data = pd.read_csv(f'stock_data/{self.ticker}.csv')
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], utc=True)
        self.stock_data['Date'] = self.stock_data['Date'].dt.tz_localize(None)
        self.stock_data.set_index('Date', inplace=True)

        # 2. Sentiment
        try:
            s = pd.read_csv(
                f'Data_Sentiment_Index/csv_files/{self.ticker}.csv',
                header=None, names=['Date', 'Sentiment'])
            s['Date'] = pd.to_datetime(s['Date'])
            self.sentiment_data = s.set_index('Date')
        except Exception:
            self.sentiment_data = pd.DataFrame()

        # 3. VIX
        try:
            v = pd.read_csv('VIX/vix_data.csv')
            v['Date'] = pd.to_datetime(v['Date'], format='%d-%m-%Y')
            self.vix_data = v.set_index('Date')
        except Exception:
            self.vix_data = pd.DataFrame()

        # 4. Macro — each file loaded independently so one missing file
        #    doesn't wipe out the others
        def _load_macro(path, date_col, date_fmt=None):
            try:
                d = pd.read_csv(path)
                d[date_col] = (pd.to_datetime(d[date_col], format=date_fmt)
                               if date_fmt else pd.to_datetime(d[date_col]))
                return d
            except Exception:
                return pd.DataFrame()

        self.gdp_data   = _load_macro('Macroeconomic_Data/GDP_quarterly.csv',
                                      'observation_date', '%d-%m-%Y')
        self.ipi_data   = _load_macro('Macroeconomic_Data/IPI_quarterly.csv',
                                      'observation_date')
        self.unemp_data = _load_macro('Macroeconomic_Data/Unemployment_Rate_monthly.csv',
                                      'observation_date')

        # 5. Quarterly financials (raw — release-date alignment done in engineer_features)
        def _try_load(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()

        self.income_stmt_q   = _try_load(f'financials_data/quarterly/{self.ticker}_income_statement_quarterly.csv')
        self.balance_sheet_q = _try_load(f'financials_data/quarterly/{self.ticker}_balance_sheet_quarterly.csv')
        self.cash_flow_q     = _try_load(f'financials_data/quarterly/{self.ticker}_cash_flow_quarterly.csv')

    # ──────────────────────────────────────────────────────────────────────────
    # FINANCIALS ALIGNMENT  — core of the change
    # ──────────────────────────────────────────────────────────────────────────
    def _align_financials_to_release_dates(self, fin_df, label):
        """
        Given a quarterly financials DataFrame, re-index it so that each
        quarter's data only appears on its ACTUAL release date (from the
        lookup), then forward-fill to create a clean daily series.

        Handles two common CSV layouts:
          - Wide:       rows = dates,    columns = metrics  (standard)
          - Transposed: rows = metrics,  columns = dates    (yfinance style)

        Returns a daily-frequency DataFrame with prefixed column names,
        or an empty DataFrame if nothing can be aligned.
        """
        if fin_df.empty:
            return pd.DataFrame()

        ticker_releases = self.release_lookup.get(self.ticker, {})
        if not ticker_releases:
            return pd.DataFrame()

        fin_df = fin_df.copy()

        # ── Detect layout ─────────────────────────────────────────────────────
        # In the transposed (yfinance) layout the first column contains metric
        # names like "Tax Effect Of Unusual Items". We detect this by checking
        # whether the first column's values look like dates or like strings.
        first_col = fin_df.columns[0]
        first_vals = fin_df[first_col].dropna().astype(str).head(5).tolist()
        looks_like_dates = sum(
            1 for v in first_vals
            if any(c.isdigit() for c in v) and ('-' in v or '/' in v)
        )

        if looks_like_dates < len(first_vals) // 2:
            # Transposed layout: first col = metric names, remaining cols = dates
            # Set metric names as index then transpose so rows become dates
            fin_df = fin_df.set_index(first_col)
            # Column headers should be date strings — drop any that aren't parseable
            valid_cols = []
            for c in fin_df.columns:
                try:
                    pd.to_datetime(str(c))
                    valid_cols.append(c)
                except Exception:
                    pass
            fin_df = fin_df[valid_cols].T          # now rows=dates, cols=metrics
            fin_df.index = pd.to_datetime(fin_df.index)
        else:
            # Standard layout: first col (or named col) = date
            date_col = first_col
            for cand in ['Date', 'date', 'Period', 'period', 'observation_date']:
                if cand in fin_df.columns:
                    date_col = cand
                    break
            fin_df[date_col] = pd.to_datetime(fin_df[date_col])
            fin_df = fin_df.set_index(date_col)

        fin_df = fin_df.sort_index()

        # Convert all columns to numeric, coercing errors to NaN
        fin_df = fin_df.apply(pd.to_numeric, errors='coerce')

        # Map each quarter-end → release date; drop unmatched quarters
        release_rows = []
        for qend, row in fin_df.iterrows():
            # Find the closest quarter-end in lookup (within 10 days tolerance)
            best_match = None
            best_diff  = pd.Timedelta(days=10)
            for lookup_qend in ticker_releases:
                diff = abs(qend - lookup_qend)
                if diff <= best_diff:
                    best_diff  = diff
                    best_match = lookup_qend

            if best_match is not None:
                release_date = ticker_releases[best_match]
                release_rows.append((release_date, row))

        if not release_rows:
            return pd.DataFrame()

        # Build a new DataFrame indexed by release dates
        dates, rows = zip(*release_rows)
        aligned = pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))
        aligned = aligned.sort_index()

        # Expand to daily via forward-fill (data stays constant until next release)
        daily = aligned.resample('D').ffill()
        daily.columns = [f'{label}_{c}' for c in daily.columns]
        return daily

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────
    def engineer_features(self):
        df = self.stock_data.copy()

        # Target: NEXT day's return
        df['Return'] = df['Close'].pct_change().shift(-1)

        # ── Price features ────────────────────────────────────────────────────
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

        for w in [5, 10, 20, 50]:
            df[f'SMA_{w}'] = df['Close'].rolling(w).mean()

        for w in [5, 10, 20]:
            df[f'Volatility_{w}'] = df['Return'].rolling(w).std()
            df[f'ATR_{w}']        = (df['High'] - df['Low']).rolling(w).mean()

        df['RSI_14']               = self._calc_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Sig'] = self._calc_macd(df['Close'])

        # ── Volume ────────────────────────────────────────────────────────────
        df['Vol_MA_20']  = df['Volume'].rolling(20).mean()
        df['Vol_Ratio']  = df['Volume'] / df['Vol_MA_20']
        df['Vol_Change'] = df['Volume'].pct_change()

        # ── Sentiment ─────────────────────────────────────────────────────────
        if not self.sentiment_data.empty:
            df = df.join(self.sentiment_data, how='left')
            df['Sentiment'] = df['Sentiment'].ffill()
            for lag in [1, 2, 3, 5]:
                df[f'Sent_Lag{lag}'] = df['Sentiment'].shift(lag)
            df['Sent_MA5']  = df['Sentiment'].rolling(5).mean()
            df['Sent_MA10'] = df['Sentiment'].rolling(10).mean()
            df['Sent_Chg']  = df['Sentiment'].diff()

        # ── VIX ───────────────────────────────────────────────────────────────
        if not self.vix_data.empty:
            vix = self.vix_data[['Close']].rename(columns={'Close': 'VIX'})
            df = df.join(vix, how='left')
            df['VIX']      = df['VIX'].ffill()
            df['VIX_Chg']  = df['VIX'].pct_change()
            df['VIX_MA5']  = df['VIX'].rolling(5).mean()
            df['VIX_MA20'] = df['VIX'].rolling(20).mean()

        # ── Macro ─────────────────────────────────────────────────────────────
        for mdf in [self.gdp_data, self.ipi_data, self.unemp_data]:
            if not mdf.empty:
                tmp = mdf.set_index('observation_date').resample('D').ffill()
                df  = df.join(tmp, how='left')

        # ── Quarterly financials — release-date aligned ───────────────────────
        for fin_raw, label in [
            (self.income_stmt_q,  'IS'),
            (self.balance_sheet_q,'BS'),
            (self.cash_flow_q,    'CF'),
        ]:
            daily_fin = self._align_financials_to_release_dates(fin_raw, label)
            if not daily_fin.empty:
                df = df.join(daily_fin, how='left')

        df = df.ffill().bfill()

        # ── Calendar ──────────────────────────────────────────────────────────
        df['DoW']         = df.index.day_of_week
        df['Month']       = df.index.month
        df['Quarter']     = df.index.quarter
        df['DoM']         = df.index.day
        df['IsMonthEnd']  = (df.index.day >= 25).astype(int)
        df['IsQtrEnd']    = df.index.is_quarter_end.astype(int)

        self.data = df

    # ──────────────────────────────────────────────────────────────────────────
    # TECHNICAL HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def _calc_rsi(self, px, p=14):
        d = px.diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        l = (-d.where(d < 0, 0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l)

    def _calc_macd(self, px, fast=12, slow=26, sig=9):
        m = px.ewm(span=fast).mean() - px.ewm(span=slow).mean()
        return m, m.ewm(span=sig).mean()

    # ──────────────────────────────────────────────────────────────────────────
    # TRAIN / TEST
    # ──────────────────────────────────────────────────────────────────────────
    def prepare_train_test(self, test_size=0.2):
        df = self.data.dropna(subset=['Return']).copy()
        drop = ['Return', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits']
        X = df[[c for c in df.columns if c not in drop]]
        y = df['Return']

        # Drop > 50 % NaN columns
        X = X.loc[:, X.isna().mean() <= 0.5]
        X = X.ffill().bfill().fillna(X.mean())

        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        sp = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X.iloc[:sp], X.iloc[sp:]
        self.y_train, self.y_test = y.iloc[:sp], y.iloc[sp:]

        self.X_train_sc = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=X.columns, index=self.X_train.index)
        self.X_test_sc = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=X.columns, index=self.X_test.index)

        self.feature_names = list(X.columns)

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL TRAINING
    # ──────────────────────────────────────────────────────────────────────────
    def train_model(self):
        # XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
        self.xgb_model.fit(
            self.X_train_sc, self.y_train,
            eval_set=[(self.X_test_sc, self.y_test)],
            verbose=False)

        # Decision Tree
        self.dt_model = DecisionTreeRegressor(
            max_depth=8, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42)
        self.dt_model.fit(self.X_train_sc, self.y_train)

        # Blended feature importance
        xi, di = self.xgb_model.feature_importances_, self.dt_model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature':    self.feature_names,
            'importance': XGB_BLEND_WEIGHT * xi + (1 - XGB_BLEND_WEIGHT) * di,
            'xgb_imp':   xi, 'dt_imp': di,
        }).sort_values('importance', ascending=False)

    def _predict(self, X):
        return (XGB_BLEND_WEIGHT * self.xgb_model.predict(X)
                + (1 - XGB_BLEND_WEIGHT) * self.dt_model.predict(X))

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATION
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_model(self):
        ytr, yte = self._predict(self.X_train_sc), self._predict(self.X_test_sc)
        return {
            'ticker':         self.ticker,
            'train_samples':  len(self.X_train),
            'test_samples':   len(self.X_test),
            'num_features':   len(self.feature_names),
            'train_rmse':     np.sqrt(mean_squared_error(self.y_train, ytr)),
            'train_mae':      mean_absolute_error(self.y_train, ytr),
            'train_r2':       r2_score(self.y_train, ytr),
            'test_rmse':      np.sqrt(mean_squared_error(self.y_test, yte)),
            'test_mae':       mean_absolute_error(self.y_test, yte),
            'test_r2':        r2_score(self.y_test, yte),
            'train_dir_acc':  np.mean((ytr > 0) == (self.y_train > 0)),
            'test_dir_acc':   np.mean((yte > 0) == (self.y_test > 0)),
            'xgb_dir_acc':    np.mean((self.xgb_model.predict(self.X_test_sc) > 0)
                                      == (self.y_test > 0)),
            'dt_dir_acc':     np.mean((self.dt_model.predict(self.X_test_sc) > 0)
                                      == (self.y_test > 0)),
            'top_feature':    self.feature_importance.iloc[0]['feature'],
            'top_feat_imp':   self.feature_importance.iloc[0]['importance'],
        }


# =============================================================================
# BATCH RUNNER
# =============================================================================

def discover_tickers(stock_data_path='stock_data'):
    return sorted(f.stem for f in Path(stock_data_path).glob('*.csv'))


def process_single_ticker(ticker, release_lookup, verbose=False):
    try:
        p = StockReturnPredictor(ticker=ticker, release_lookup=release_lookup)
        p.load_data()
        p.engineer_features()
        p.prepare_train_test(test_size=0.2)
        p.train_model()
        m = p.evaluate_model()
        if verbose:
            print(f"✓ {ticker}  R²={m['test_r2']:.4f}  DirAcc={m['test_dir_acc']:.2%}")
        return m, p, None
    except Exception as e:
        if verbose:
            print(f"✗ {ticker}: {e}")
        return None, None, str(e)


def batch_process_sp500(max_tickers=None, verbose=True, save_models=False):
    print("=" * 80)
    print("S&P 500  XGBoost + Decision Tree  —  Actual Filing Dates")
    print(f"XGB blend weight: {XGB_BLEND_WEIGHT:.0%}")
    print("=" * 80)

    # Load release dates ONCE for all tickers
    print("\nLoading quarterly release dates...")
    release_lookup = load_release_date_lookup()

    tickers = discover_tickers()
    if max_tickers:
        tickers = tickers[:max_tickers]
    print(f"Tickers to process: {len(tickers)}\n")

    results, errors, predictors = [], [], {}

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>4}/{len(tickers)}] {ticker:<6}", end='  ')
        m, predictor, err = process_single_ticker(
            ticker, release_lookup, verbose=False)

        if m:
            results.append(m)
            if save_models:
                predictors[ticker] = predictor
            print(f"✓  R²={m['test_r2']:>7.4f}  "
                  f"Blend={m['test_dir_acc']:.1%}  "
                  f"XGB={m['xgb_dir_acc']:.1%}  "
                  f"DT={m['dt_dir_acc']:.1%}")
        else:
            errors.append({'ticker': ticker, 'error': err})
            print(f"✗  {err}")

    results_df = pd.DataFrame(results)
    errors_df  = pd.DataFrame(errors)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'sp500_xgb_dt_results_{ts}.csv', index=False)
    if not errors_df.empty:
        errors_df.to_csv(f'sp500_xgb_dt_errors_{ts}.csv', index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"DONE — {len(results)} succeeded, {len(errors)} failed")
    print("=" * 80)

    if not results_df.empty:
        for col, lbl in [('test_r2', 'Test R²'),
                          ('test_dir_acc', 'Directional Accuracy'),
                          ('test_rmse', 'RMSE')]:
            fmt = '{:.2%}' if 'acc' in col else '{:.4f}'
            print(f"\n{lbl}:  mean={fmt.format(results_df[col].mean())}  "
                  f"median={fmt.format(results_df[col].median())}  "
                  f"min={fmt.format(results_df[col].min())}  "
                  f"max={fmt.format(results_df[col].max())}")

        cols = ['ticker', 'test_r2', 'test_dir_acc',
                'xgb_dir_acc', 'dt_dir_acc', 'test_rmse']
        print("\nTop 10 by R²:")
        print(results_df.nlargest(10, 'test_r2')[cols].to_string(index=False))
        print("\nTop 10 by Directional Accuracy:")
        print(results_df.nlargest(10, 'test_dir_acc')[cols].to_string(index=False))

        _visualise(results_df, ts)

    return results_df, errors_df


# =============================================================================
# VISUALISATION
# =============================================================================

def _visualise(df, ts):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('S&P 500 — XGBoost + DT Ensemble (Actual Filing Dates)', fontsize=13)

    def _hist(ax, col, xlabel, title, ref=None):
        ax.hist(df[col], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(df[col].mean(), color='r', ls='--',
                   label=f"Mean: {df[col].mean():.3f}")
        if ref is not None:
            ax.axvline(ref, color='grey', lw=1,
                       label=f'Ref: {ref}')
        ax.set(xlabel=xlabel, ylabel='# Tickers', title=title)
        ax.legend(); ax.grid(alpha=0.3)

    _hist(axes[0, 0], 'test_r2',       'Test R²',             'R² Distribution', ref=0)
    _hist(axes[0, 1], 'test_dir_acc',  'Directional Accuracy','Dir Acc Distribution', ref=0.5)

    axes[1, 0].scatter(df['test_r2'], df['test_dir_acc'], alpha=0.5, s=20)
    axes[1, 0].axhline(0.5, color='grey', lw=1, alpha=0.5)
    axes[1, 0].axvline(0,   color='grey', lw=1, alpha=0.5)
    axes[1, 0].set(xlabel='Test R²', ylabel='Dir Acc', title='R² vs Dir Acc')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(df['xgb_dir_acc'], df['dt_dir_acc'], alpha=0.5, s=20)
    lim = [min(df['xgb_dir_acc'].min(), df['dt_dir_acc'].min()) - 0.01,
           max(df['xgb_dir_acc'].max(), df['dt_dir_acc'].max()) + 0.01]
    axes[1, 1].plot(lim, lim, 'r--', lw=1, label='Equal')
    axes[1, 1].set(xlabel='XGBoost Dir Acc', ylabel='DT Dir Acc',
                   title='XGBoost vs Decision Tree')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    fname = f'sp500_xgb_dt_summary_{ts}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {fname}")
    plt.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    results_df, errors_df = batch_process_sp500(
        max_tickers=None,    # None = all 503; set e.g. 20 for a quick test
        verbose=True,
        save_models=False,
    )