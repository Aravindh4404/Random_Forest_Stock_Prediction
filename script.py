import pandas as pd
import requests
import time
import io
from datetime import datetime

UA = "Your Name your.email@example.com (research; contact if needed)"  # <-- REQUIRED by SEC
BATCH_SIZE = 50
SLEEP_SEC = 0.2  # be polite; increase if you get rate-limited

TARGET_QS = [f"2024 Q{i}" for i in range(1,5)] + [f"2025 Q{i}" for i in range(1,5)]

def qkey_from_date(d):
    dt = datetime.strptime(d, "%Y-%m-%d")
    q = (dt.month - 1)//3 + 1
    return f"{dt.year} Q{q}"

def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9",
    }

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.read_html(io.StringIO(r.text), attrs={"id": "constituents"})[0]
    df.rename(columns={"Symbol": "Ticker", "Security": "Company", "CIK": "CIK"}, inplace=True)
    df["CIK"] = df["CIK"].astype(str).str.zfill(10)

    return df[["Ticker", "Company", "CIK"]]


def sec_get_submissions(cik10):
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    r.raise_for_status()
    return r.json(), url

def extract_quarter_filing_dates(sub_json):
    recent = sub_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    out = {k: "" for k in TARGET_QS}

    for form, fdate, rdate in zip(forms, filing_dates, report_dates):
        if form not in ("10-Q", "10-K"):
            continue
        if not rdate:
            continue
        qk = qkey_from_date(rdate)
        if qk in out and out[qk] == "":
            out[qk] = fdate

    return out

def run_batches():
    sp = fetch_sp500()
    n = len(sp)
    batches = [(i, min(i + BATCH_SIZE, n)) for i in range(0, n, BATCH_SIZE)]

    for b, (start, end) in enumerate(batches, 1):
        chunk = sp.iloc[start:end].copy()
        rows = []

        for _, row in chunk.iterrows():
            cik10 = row["CIK"]
            src_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
            try:
                sub, src_url = sec_get_submissions(cik10)
                qdates = extract_quarter_filing_dates(sub)
            except Exception:
                qdates = {k: "" for k in TARGET_QS}

            rec = {
                "Ticker": row["Ticker"],
                "Company": row["Company"],
                "CIK": cik10,
                **{k: qdates[k] for k in TARGET_QS},
                "Source (SEC Submissions JSON)": src_url,
            }
            rows.append(rec)
            time.sleep(SLEEP_SEC)

        outdf = pd.DataFrame(rows)
        cols = ["Ticker", "Company", "CIK"] + TARGET_QS + ["Source (SEC Submissions JSON)"]
        outdf = outdf[cols]

        outname = f"sp500_quarterly_dates_batch_{b:02d}.xlsx"
        outdf.to_excel(outname, index=False)
        print(f"Wrote {outname} ({start+1}-{end} of {n})")

if __name__ == "__main__":
    run_batches()
