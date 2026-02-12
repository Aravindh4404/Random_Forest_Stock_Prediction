import io
import time
from datetime import datetime
import pandas as pd
import requests
import yfinance as yf

UA = "Alejandro Alvarado your.email@example.com (research)"  # pon tu email real
BATCH_SIZE = 50
SLEEP_SEC = 0.25  # baja más lento si te bloquea

TARGET_QS = [f"2024 Q{i}" for i in range(1, 5)] + [f"2025 Q{i}" for i in range(1, 5)]

def qkey_from_dt(dt: datetime) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year} Q{q}"

def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.read_html(io.StringIO(r.text), attrs={"id": "constituents"})[0]
    df.rename(columns={"Symbol": "Ticker", "Security": "Company"}, inplace=True)
    return df[["Ticker", "Company"]]

def fetch_missing():
    df=pd.DataFrame({
        "Ticker": ["BRK-B", "BF-B"],
        "Company": ["Berkshire Hathaway Inc.", "Brown-Forman Corporation"]
    })
    return df[["Ticker", "Company"]]

def yahoo_earnings_by_quarter(ticker: str) -> dict:
    """
    Devuelve un dict con claves '2024 Q1'..'2025 Q4' y valores 'YYYY-MM-DD' (earnings date).
    Usa yfinance.get_earnings_dates, filtra 2024-2025 y asigna por trimestre calendario.
    """
    out = {k: "" for k in TARGET_QS}

    # yfinance devuelve un DataFrame indexado por fecha/hora (earnings date)
    # limit=60 suele cubrir varios años para muchos tickers (no siempre).
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=60)
    except Exception:
        return out

    if ed is None or len(ed) == 0:
        return out

    # Normalizar índice a datetime (sin tz)
    idx = pd.to_datetime(ed.index, errors="coerce")
    idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx

    # Filtrar rango 2024-01-01 a 2025-12-31
    mask = (idx >= pd.Timestamp("2024-01-01")) & (idx <= pd.Timestamp("2025-12-31"))
    dates = idx[mask].sort_values()

    for dt in dates:
        qk = qkey_from_dt(dt.to_pydatetime())
        if qk in out and out[qk] == "":
            out[qk] = dt.date().isoformat()

    return out

def run_batches():
    sp = fetch_missing()
    n = len(sp)
    batches = [(i, min(i + BATCH_SIZE, n)) for i in range(0, n, BATCH_SIZE)]

    for b, (start, end) in enumerate(batches, 1):
        chunk = sp.iloc[start:end].copy()
        rows = []

        for _, row in chunk.iterrows():
            tkr = row["Ticker"]
            qdates = yahoo_earnings_by_quarter(tkr)

            rec = {
                "Ticker": tkr,
                "Company": row["Company"],
                **{k: qdates[k] for k in TARGET_QS},
                "Source (Yahoo via yfinance)": "https://finance.yahoo.com (yfinance)"
            }
            rows.append(rec)
            time.sleep(SLEEP_SEC)

        outdf = pd.DataFrame(rows)
        cols = ["Ticker", "Company"] + TARGET_QS + ["Source (Yahoo via yfinance)"]
        outdf = outdf[cols]

        outname = f"sp500_yahoo_earnings_calendar_2024_2025_batch_{b:02d}.xlsx"
        outdf.to_excel(outname, index=False)
        print(f"Wrote {outname} ({start+1}-{end} of {n})")

if __name__ == "__main__":
    run_batches()