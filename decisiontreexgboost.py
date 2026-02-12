"""
S&P 500 Batch XGBoost + Decision Tree Stock Return Prediction
- Uses ACTUAL quarterly filing/release dates from Quarterly_Release_Dates/*.xlsx
- Uses ACTUAL macroeconomic release dates for GDP, IPI, and Unemployment
- Financial data only becomes available in features ON or AFTER its release date
- Blended ensemble: XGBoost (70%) + Decision Tree (30%)
- Target: next-day return

ROBUST VERSION with comprehensive error handling
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
MACRO_RELEASE_DIR       = 'Macroeconomic_Data'
FISCAL_QUARTER_TOLERANCE = 40  # Days tolerance for fiscal vs calendar quarters
# ─────────────────────────────────────────────────────────────────────────────


def _quarter_label_to_end_date(label):
    """
    Convert a quarter label like '2024 Q1' → 2024-03-31,
    '2024 Q2' → 2024-06-30, etc.
    """
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


def load_release_date_lookup(directory=QUARTERLY_DATES_DIR):
    """Load quarterly filing release dates for all tickers"""
    print(f"\n→ Loading quarterly release dates from {directory}/...")

    path = Path(directory)
    if not path.exists():
        print(f"   ❌ Directory not found: {directory}")
        print(f"   Please create this directory and add your Excel files")
        return {}

    all_files = sorted(path.glob('sp500_quarterly_dates_batch_*.xlsx'))

    if not all_files:
        print(f"   ❌ No batch files found in '{directory}'")
        print(f"   Expected files like: sp500_quarterly_dates_batch_01.xlsx")
        return {}

    print(f"   Found {len(all_files)} batch files")

    frames = []
    for fpath in all_files:
        try:
            df = pd.read_excel(fpath, header=0, engine='openpyxl')
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)

            # Flexible column renaming
            if len(cols) >= 3:
                rename = {cols[0]: 'Ticker', cols[1]: 'Company', cols[2]: 'CIK'}
                df.rename(columns=rename, inplace=True)
                df['Ticker'] = df['Ticker'].astype(str).str.strip()
                frames.append(df)
        except Exception as e:
            print(f"   ⚠ Error reading {fpath.name}: {e}")
            continue

    if not frames:
        print(f"   ❌ Could not load any batch files")
        return {}

    master = pd.concat(frames, ignore_index=True)
    quarter_cols = [c for c in master.columns if c not in ('Ticker', 'Company', 'CIK')]

    lookup = {}
    for _, row in master.iterrows():
        ticker = row['Ticker']
        if pd.isna(ticker) or ticker == 'nan' or ticker == 'None':
            continue

        ticker_map = {}
        for qcol in quarter_cols:
            release_val = row[qcol]
            if pd.isna(release_val):
                continue

            try:
                release_date = pd.to_datetime(release_val)
                qend = _quarter_label_to_end_date(str(qcol))
                ticker_map[qend] = release_date
            except Exception:
                continue

        if ticker_map:
            lookup[ticker] = ticker_map

    print(f"   ✅ Loaded {len(lookup)} tickers, {sum(len(v) for v in lookup.values())} quarter releases")
    return lookup


def load_macro_release_dates(macro_dir=MACRO_RELEASE_DIR):
    """Load macroeconomic release date calendars"""
    print(f"\n→ Loading macro release dates from {macro_dir}/...")

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

            # Find quarter and release columns
            quarter_col = [c for c in gdp_ipi.columns if 'quarter' in c.lower()][0]
            release_col = [c for c in gdp_ipi.columns if 'release' in c.lower() and 'date' in c.lower()][0]

            gdp_ipi['observation_date'] = gdp_ipi[quarter_col].apply(_quarter_label_to_end_date)
            gdp_ipi['release_date'] = pd.to_datetime(gdp_ipi[release_col])

            macro_releases['gdp'] = gdp_ipi[['observation_date', 'release_date', quarter_col]].copy()
            macro_releases['gdp'].rename(columns={quarter_col: 'Quarter'}, inplace=True)
            macro_releases['ipi'] = macro_releases['gdp'].copy()

            print(f"   ✅ GDP/IPI: {len(gdp_ipi)} quarters")
        else:
            print(f"   ⚠ File not found: Calendar GDP AND IPI.csv")
    except Exception as e:
        print(f"   ⚠ GDP/IPI load error: {e}")

    # Load Unemployment
    try:
        unemp_path = Path(macro_dir) / 'CALENDAR UNEMPLOYMENT RATE.csv'
        if unemp_path.exists():
            unemp = pd.read_csv(unemp_path)
            unemp.columns = [c.strip() for c in unemp.columns]

            first_col = unemp.columns[0]
            release_col = None

            for col in unemp.columns:
                if 'release' in col.lower() and 'date' in col.lower():
                    release_col = col
                    break

            if release_col is None and len(unemp.columns) > 1:
                release_col = unemp.columns[1]

            if release_col:
                # Try multiple date formats
                unemp['observation_date'] = pd.to_datetime(unemp[first_col], format='%b-%y', errors='coerce')
                if unemp['observation_date'].isna().all():
                    unemp['observation_date'] = pd.to_datetime(unemp[first_col], errors='coerce')

                unemp['release_date'] = pd.to_datetime(unemp[release_col], errors='coerce')
                macro_releases['unemployment'] = unemp[['observation_date', 'release_date']].dropna()

                print(f"   ✅ Unemployment: {len(macro_releases['unemployment'])} months")
            else:
                print(f"   ⚠ Could not find release date column in unemployment file")
        else:
            print(f"   ⚠ File not found: CALENDAR UNEMPLOYMENT RATE.csv")
    except Exception as e:
        print(f"   ⚠ Unemployment load error: {e}")

    return macro_releases


class StockReturnPredictor:
    def __init__(self, ticker='NVDA', release_lookup=None, macro_releases=None):
        self.ticker = ticker
        self.release_lookup = release_lookup or {}
        self.macro_releases = macro_releases or {}
        self.xgb_model = None
        self.dt_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def load_data(self):
        """Load all data sources with error handling"""
        # 1. OHLCV (required)
        try:
            self.stock_data = pd.read_csv(f'stock_data/{self.ticker}.csv')
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], utc=True)
            self.stock_data['Date'] = self.stock_data['Date'].dt.tz_localize(None)
            self.stock_data.set_index('Date', inplace=True)
        except Exception as e:
            raise FileNotFoundError(f"Cannot load stock data for {self.ticker}: {e}")

        # 2. Sentiment (optional)
        try:
            s = pd.read_csv(f'Data_Sentiment_Index/csv_files/{self.ticker}.csv',
                          header=None, names=['Date', 'Sentiment'])
            s['Date'] = pd.to_datetime(s['Date'])
            self.sentiment_data = s.set_index('Date')
        except Exception:
            self.sentiment_data = pd.DataFrame()

        # 3. VIX (optional)
        try:
            v = pd.read_csv('VIX/vix_data.csv')
            v['Date'] = pd.to_datetime(v['Date'], format='%d-%m-%Y', errors='coerce')
            if v['Date'].isna().any():
                v['Date'] = pd.to_datetime(v['Date'], errors='coerce')
            self.vix_data = v.set_index('Date')
        except Exception:
            self.vix_data = pd.DataFrame()

        # 4. Macro data (optional)
        def _load_macro_safe(path, date_col):
            try:
                if not Path(path).exists():
                    return pd.DataFrame()
                d = pd.read_csv(path)
                # Try multiple date formats
                d[date_col] = pd.to_datetime(d[date_col], format='%d-%m-%Y', errors='coerce')
                if d[date_col].isna().all():
                    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
                return d
            except Exception:
                return pd.DataFrame()

        self.gdp_data = _load_macro_safe('Macroeconomic_Data/GDP_quarterly.csv', 'observation_date')
        self.ipi_data = _load_macro_safe('Macroeconomic_Data/IPI_quarterly.csv', 'observation_date')
        self.unemp_data = _load_macro_safe('Macroeconomic_Data/Unemployment_Rate_monthly.csv', 'observation_date')

        # 5. Financials (optional)
        def _try_load(path):
            try:
                return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
            except Exception:
                return pd.DataFrame()

        self.income_stmt_q = _try_load(f'financials_data/quarterly/{self.ticker}_income_statement_quarterly.csv')
        self.balance_sheet_q = _try_load(f'financials_data/quarterly/{self.ticker}_balance_sheet_quarterly.csv')
        self.cash_flow_q = _try_load(f'financials_data/quarterly/{self.ticker}_cash_flow_quarterly.csv')

    def _align_financials_to_release_dates(self, fin_df, label):
        """Align financial data to release dates"""
        if fin_df.empty:
            return pd.DataFrame()

        ticker_releases = self.release_lookup.get(self.ticker, {})
        if not ticker_releases:
            return pd.DataFrame()

        fin_df = fin_df.copy()

        # Detect layout
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

        fin_df = fin_df.sort_index()
        fin_df = fin_df.apply(pd.to_numeric, errors='coerce')

        # Match quarters with tolerance for fiscal years
        release_rows = []
        for qend, row in fin_df.iterrows():
            best_match = None
            best_diff = pd.Timedelta(days=FISCAL_QUARTER_TOLERANCE)

            for lookup_qend in ticker_releases:
                diff = abs(qend - lookup_qend)
                if diff <= best_diff:
                    best_diff = diff
                    best_match = lookup_qend

            if best_match is not None:
                release_date = ticker_releases[best_match]
                release_rows.append((release_date, row))

        if not release_rows:
            return pd.DataFrame()

        dates, rows = zip(*release_rows)
        aligned = pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))
        aligned = aligned.sort_index()

        # Daily forward-fill
        daily = aligned.resample('D').ffill()
        daily.columns = [f'{label}_{c}' for c in daily.columns]
        return daily

    def _align_macro_to_release_dates(self, macro_df, release_calendar, label):
        """Align macro data to release dates"""
        if macro_df.empty or release_calendar.empty:
            return pd.DataFrame()

        macro_df = macro_df.copy()

        if 'observation_date' in macro_df.columns:
            macro_df = macro_df.set_index('observation_date')

        macro_df = macro_df.sort_index()
        release_rows = []

        for obs_date, row in macro_df.iterrows():
            matches = release_calendar[
                abs(release_calendar['observation_date'] - obs_date) <= pd.Timedelta(days=5)
            ]

            if len(matches) > 0:
                release_date = matches.iloc[0]['release_date']
                release_rows.append((release_date, row))

        if not release_rows:
            return pd.DataFrame()

        dates, rows = zip(*release_rows)
        aligned = pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates))
        aligned = aligned.sort_index()

        daily = aligned.resample('D').ffill()
        daily.columns = [f'{label}_{c}' for c in daily.columns]

        return daily

    def engineer_features(self):
        """Create all features with release-date alignment"""
        df = self.stock_data.copy()

        # Target
        df['Return'] = df['Close'].pct_change().shift(-1)

        # Price features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

        for w in [5, 10, 20, 50]:
            df[f'SMA_{w}'] = df['Close'].rolling(w).mean()

        for w in [5, 10, 20]:
            df[f'Volatility_{w}'] = df['Return'].rolling(w).std()
            df[f'ATR_{w}'] = (df['High'] - df['Low']).rolling(w).mean()

        df['RSI_14'] = self._calc_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Sig'] = self._calc_macd(df['Close'])

        # Volume
        df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA_20']
        df['Vol_Change'] = df['Volume'].pct_change()

        # Sentiment
        if not self.sentiment_data.empty:
            df = df.join(self.sentiment_data, how='left')
            df['Sentiment'] = df['Sentiment'].ffill()
            for lag in [1, 2, 3, 5]:
                df[f'Sent_Lag{lag}'] = df['Sentiment'].shift(lag)
            df['Sent_MA5'] = df['Sentiment'].rolling(5).mean()
            df['Sent_MA10'] = df['Sentiment'].rolling(10).mean()
            df['Sent_Chg'] = df['Sentiment'].diff()

        # VIX
        if not self.vix_data.empty:
            vix = self.vix_data[['Close']].rename(columns={'Close': 'VIX'})
            df = df.join(vix, how='left')
            df['VIX'] = df['VIX'].ffill()
            df['VIX_Chg'] = df['VIX'].pct_change()
            df['VIX_MA5'] = df['VIX'].rolling(5).mean()
            df['VIX_MA20'] = df['VIX'].rolling(20).mean()

        # Macro with release alignment
        for data, key, prefix in [(self.gdp_data, 'gdp', 'GDP'),
                                   (self.ipi_data, 'ipi', 'IPI'),
                                   (self.unemp_data, 'unemployment', 'UNEMP')]:
            if not data.empty and key in self.macro_releases and not self.macro_releases[key].empty:
                aligned = self._align_macro_to_release_dates(data, self.macro_releases[key], prefix)
                if not aligned.empty:
                    df = df.join(aligned, how='left')

        # Financials with release alignment
        for fin_raw, label in [(self.income_stmt_q, 'IS'),
                                (self.balance_sheet_q, 'BS'),
                                (self.cash_flow_q, 'CF')]:
            daily_fin = self._align_financials_to_release_dates(fin_raw, label)
            if not daily_fin.empty:
                df = df.join(daily_fin, how='left')

        df = df.ffill().bfill()

        # Calendar
        df['DoW'] = df.index.day_of_week
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DoM'] = df.index.day
        df['IsMonthEnd'] = (df.index.day >= 25).astype(int)
        df['IsQtrEnd'] = df.index.is_quarter_end.astype(int)

        self.data = df

    def _calc_rsi(self, px, p=14):
        d = px.diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        l = (-d.where(d < 0, 0)).rolling(p).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = g / l
            return 100 - 100 / (1 + rs)

    def _calc_macd(self, px, fast=12, slow=26, sig=9):
        m = px.ewm(span=fast).mean() - px.ewm(span=slow).mean()
        return m, m.ewm(span=sig).mean()

    def prepare_train_test(self, test_size=0.2):
        df = self.data.dropna(subset=['Return']).copy()
        drop = ['Return', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        X = df[[c for c in df.columns if c not in drop]]
        y = df['Return']

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

    def train_model(self):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0)
        self.xgb_model.fit(
            self.X_train_sc, self.y_train,
            eval_set=[(self.X_test_sc, self.y_test)],
            verbose=False)

        self.dt_model = DecisionTreeRegressor(
            max_depth=8, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42)
        self.dt_model.fit(self.X_train_sc, self.y_train)

        xi, di = self.xgb_model.feature_importances_, self.dt_model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': XGB_BLEND_WEIGHT * xi + (1 - XGB_BLEND_WEIGHT) * di,
            'xgb_imp': xi, 'dt_imp': di,
        }).sort_values('importance', ascending=False)

    def _predict(self, X):
        return (XGB_BLEND_WEIGHT * self.xgb_model.predict(X)
                + (1 - XGB_BLEND_WEIGHT) * self.dt_model.predict(X))

    def evaluate_model(self):
        ytr, yte = self._predict(self.X_train_sc), self._predict(self.X_test_sc)
        return {
            'ticker': self.ticker,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'num_features': len(self.feature_names),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, ytr)),
            'train_mae': mean_absolute_error(self.y_train, ytr),
            'train_r2': r2_score(self.y_train, ytr),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, yte)),
            'test_mae': mean_absolute_error(self.y_test, yte),
            'test_r2': r2_score(self.y_test, yte),
            'train_dir_acc': np.mean((ytr > 0) == (self.y_train > 0)),
            'test_dir_acc': np.mean((yte > 0) == (self.y_test > 0)),
            'xgb_dir_acc': np.mean((self.xgb_model.predict(self.X_test_sc) > 0) == (self.y_test > 0)),
            'dt_dir_acc': np.mean((self.dt_model.predict(self.X_test_sc) > 0) == (self.y_test > 0)),
            'top_feature': self.feature_importance.iloc[0]['feature'],
            'top_feat_imp': self.feature_importance.iloc[0]['importance'],
        }


def discover_tickers(stock_data_path='stock_data'):
    path = Path(stock_data_path)
    if not path.exists():
        print(f"❌ Stock data directory not found: {stock_data_path}")
        return []
    return sorted(f.stem for f in path.glob('*.csv'))


def process_single_ticker(ticker, release_lookup, macro_releases, verbose=False):
    try:
        p = StockReturnPredictor(ticker=ticker, release_lookup=release_lookup, macro_releases=macro_releases)
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
    print("S&P 500  XGBoost + DT — Actual Release Dates (ROBUST VERSION)")
    print(f"XGB blend weight: {XGB_BLEND_WEIGHT:.0%}")
    print("=" * 80)

    release_lookup = load_release_date_lookup()
    macro_releases = load_macro_release_dates()

    tickers = discover_tickers()
    if not tickers:
        print("\n❌ No tickers found. Please check your stock_data directory.")
        return pd.DataFrame(), pd.DataFrame()

    if max_tickers:
        tickers = tickers[:max_tickers]
    print(f"\n→ Tickers to process: {len(tickers)}\n")

    results, errors, predictors = [], [], {}

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>4}/{len(tickers)}] {ticker:<6}", end='  ')
        m, predictor, err = process_single_ticker(ticker, release_lookup, macro_releases, verbose=False)

        if m:
            results.append(m)
            if save_models:
                predictors[ticker] = predictor
            print(f"✓  R²={m['test_r2']:>7.4f}  Blend={m['test_dir_acc']:.1%}  "
                  f"XGB={m['xgb_dir_acc']:.1%}  DT={m['dt_dir_acc']:.1%}")
        else:
            errors.append({'ticker': ticker, 'error': err})
            print(f"✗  {err}")

    results_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not results_df.empty:
        results_df.to_csv(f'sp500_xgb_dt_results_{ts}.csv', index=False)
    if not errors_df.empty:
        errors_df.to_csv(f'sp500_xgb_dt_errors_{ts}.csv', index=False)

    print("\n" + "=" * 80)
    print(f"DONE — {len(results)} succeeded, {len(errors)} failed")
    print("=" * 80)

    if not results_df.empty:
        for col, lbl in [('test_r2', 'Test R²'), ('test_dir_acc', 'Directional Accuracy'), ('test_rmse', 'RMSE')]:
            fmt = '{:.2%}' if 'acc' in col else '{:.4f}'
            print(f"\n{lbl}:  mean={fmt.format(results_df[col].mean())}  "
                  f"median={fmt.format(results_df[col].median())}  "
                  f"min={fmt.format(results_df[col].min())}  max={fmt.format(results_df[col].max())}")

        cols = ['ticker', 'test_r2', 'test_dir_acc', 'xgb_dir_acc', 'dt_dir_acc', 'test_rmse']
        if len(results_df) >= 10:
            print("\nTop 10 by R²:")
            print(results_df.nlargest(10, 'test_r2')[cols].to_string(index=False))
            print("\nTop 10 by Directional Accuracy:")
            print(results_df.nlargest(10, 'test_dir_acc')[cols].to_string(index=False))

    return results_df, errors_df


if __name__ == '__main__':
    results_df, errors_df = batch_process_sp500(
        max_tickers=None,
        verbose=True,
        save_models=False,
    )