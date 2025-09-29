import os, io, time
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yaml

st.set_page_config(page_title="ETF Macro Agent Dashboard", layout="wide")

# =========================================================
# 1) Load config
# =========================================================
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)
TH = CFG["tiles"]

# =========================================================
# 2) Robust data fetchers
# =========================================================
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """
    Robust fetch for a single FRED series.
    Tries 'downloaddata' (full history) then 'fredgraph.csv'.
    Adds a browser-like UA and retries to survive redirects/timeouts.
    Returns columns: Date, Name, Value
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    urls = [
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
    ]
    last_err = None
    for url in urls:
        for attempt in range(3):
            try:
                r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))

                # Normalize likely column names
                if "DATE" in df.columns:
                    date_col = "DATE"
                elif "observation_date" in df.columns:
                    date_col = "observation_date"
                else:
                    date_col = df.columns[0]

                # Value column is often the series id; otherwise use 2nd column
                val_col = series_id if series_id in df.columns else df.columns[1]

                out = df[[date_col, val_col]].copy()
                out.columns = ["Date", "Value"]
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
                out["Name"]  = series_id
                return out.dropna(subset=["Date"])
            except Exception as e:
                last_err = e
                time.sleep(1.5)
                continue
    raise RuntimeError(f"Failed to fetch {series_id}: {last_err}")

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_tp10() -> pd.DataFrame:
    """
    Robust NY Fed term premium fetch (no API).
    Tries several official URLs, follows redirects, and strips any preamble
    before the actual CSV header. Returns Date/Value for TP10.
    """
    import requests, io, pandas as pd, numpy as np

    url_candidates = [
        # Primary (NY Fed)
        "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTP.csv",
        "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTP.csv?download=true",
        "https://www.newyorkfed.org/research/data_indicators/ACMTP.csv",
        # CDN mirror occasionally used by the site
        "https://nyfed.org/medialibrary/media/research/data_indicators/ACMTP.csv",
    ]
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
    }
    last_err = None
    for url in url_candidates:
        try:
            r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            r.raise_for_status()
            text = r.text

            # Remove any preamble; find the header row containing both Date and TP10
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            hdr_idx = next(i for i, ln in enumerate(lines)
                           if ("date" in ln.lower() and "tp10" in ln.lower()))
            csv_text = "\n".join(lines[hdr_idx:])

            df = pd.read_csv(io.StringIO(csv_text))
            df = df[["Date", "TP10"]].copy()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["TP10"] = pd.to_numeric(df["TP10"], errors="coerce")
            out = df.rename(columns={"TP10": "Value"})
            out["Name"] = "TP10"
            return out.dropna(subset=["Date"])
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to fetch TP10: {last_err}")


# =========================================================
# 3) Small helpers
# =========================================================
def latest_value(df_all: pd.DataFrame, name: str):
    """Return latest value and date for a series name from the combined data table."""
    sub = df_all[df_all["Name"] == name]
    if sub.empty:
        return np.nan, None
    row = sub.sort_values("Date").iloc[-1]
    return float(row["Value"]), pd.to_datetime(row["Date"]).date()

def color_from_rule(val, green_cond, yellow_cond):
    """Tile background color: green / yellow / red / gray (missing)."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "#EEE"
        if green_cond(val):
            return "#C6EFCE"
        if yellow_cond(val):
            return "#FFEB9C"
        return "#FFC7CE"
    except Exception:
        return "#EEE"

# =========================================================
# 4) Load all data (tolerant)
# =========================================================
with st.spinner("Fetching data..."):
    frames, fetch_errors = [], []
    for sid in CFG["series"]["fred"]:
        try:
            frames.append(fetch_fred_series(sid))
        except Exception as e:
            fetch_errors.append(f"{sid}: {e}")

    if CFG["series"].get("nyfed_tp10", True):
        try:
            frames.append(fetch_tp10())
        except Exception as e:
            fetch_errors.append(f"TP10: {e}")

    if not frames:
        st.error("No data could be loaded. Please refresh or try again.")
        st.stop()

    data = pd.concat(frames, ignore_index=True).dropna(subset=["Value"])

    if fetch_errors:
        st.warning("Some series failed to load:\n- " + "\n- ".join(fetch_errors))

# =========================================================
# 5) Compute macro tiles
# =========================================================
y10, y10_date = latest_value(data, "DGS10")
y2, _  = latest_value(data, "DGS2")
y3m, _ = latest_value(data, "DGS3MO")
tips, _ = latest_value(data, "DFII10")
tp10, _ = latest_value(data, "TP10")
dxy, _  = latest_value(data, "DTWEXBGS")   # broad USD index proxy (stable)
pmi, _  = latest_value(data, "NAPMNOI")    # PMI New Orders (ISM)
oas, _  = latest_value(data, "BAMLH0A0HYM2")
wti, _  = latest_value(data, "DCOILWTICO")

def to_decimal(x):
    if pd.isna(x): return np.nan
    return x/100 if x > 1 else x

y10d, y2d, y3md, tipsd, tp10d = map(to_decimal, [y10, y2, y3m, tips, tp10])
curve_bps = (y10d - y2d) * 10000 if (pd.notna(y10d) and pd.notna(y2d)) else np.nan

tiles = [
    ("10Y UST (%)", y10d*100 if pd.notna(y10d) else np.nan, lambda v: v/100 < TH["y10_green_max"], lambda v: v/100 < TH["y10_yellow_max"]),
    ("2Y UST (%)",  y2d*100  if pd.notna(y2d)  else np.nan, lambda v: True, lambda v: True),
    ("3M Bill (%)", y3md*100 if pd.notna(y3md) else np.nan, lambda v: True, lambda v: True),
    ("10Y TIPS (%)",tipsd*100 if pd.notna(tipsd) else np.nan, lambda v: v/100 < TH["tips_green_max"], lambda v: v/100 < TH["tips_yellow_max"]),
    ("Term Prem (%)",tp10d*100 if pd.notna(tp10d) else np.nan, lambda v: v/100 < TH["tp10_green_max"], lambda v: v/100 < TH["tp10_yellow_max"]),
    ("Broad $ Index", dxy, lambda v: v <= TH["dxy_green_max"], lambda v: v <= TH["dxy_yellow_max"]),
    ("PMI New Orders", pmi, lambda v: v >= TH["pmi_green_min"], lambda v: v >= TH["pmi_yellow_min"]),
    ("HY OAS (bps)", oas*100 if pd.notna(oas) else np.nan, lambda v: v <= TH["hyoas_green_max_bps"], lambda v: v <= TH["hyoas_yellow_max_bps"]),
    ("10s–2s (bps)", curve_bps, lambda v: v >= TH["curve_green_min_bps"], lambda v: v >= TH["curve_yellow_min_bps"]),
    ("WTI ($)", wti, lambda v: v >= TH["wti_green_min"], lambda v: v >= TH["wti_yellow_min"]),
]

st.title("ETF Macro Agent Dashboard")
st.caption("Data: FRED & NY Fed. USD tile uses FRED Broad Dollar Index (DTWEXBGS). Thresholds in config.yaml.")

cols = st.columns(len(tiles))
for i, (label, val, green_fn, yellow_fn) in enumerate(tiles):
    bg = color_from_rule(val, green_fn, yellow_fn)
    with cols[i]:
        st.markdown(
            f"""
            <div style="background-color:{bg}; padding:12px; border-radius:8px; text-align:center; border:1px solid #ddd;">
                <div style="font-size:12px; color:#333;">{label}</div>
                <div style="font-size:24px; font-weight:700; color:#111;">{'' if pd.isna(val) else (f"{val:,.2f}")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
st.write(f"**As of:** {y10_date or '—'}")

# =========================================================
# 6) ETF mapping & signals
# =========================================================
ETF_ROWS = [
    ("XLE","Energy: PMI expansion + softer USD support commodities; oil price is direct driver."),
    ("XLB","Materials: manufacturing + weaker USD + tight credit help."),
    ("XLI","Industrials: orders/capex improve with PMI; less inversion supports financing."),
    ("XLF","Financials: steeper curve lifts NIM; tighter spreads reduce credit risk."),
    ("XLK","Tech: lower real yields (duration) + stable demand aid valuations."),
    ("XLV","Health Care: defensive when growth cools or spreads widen."),
    ("XLY","Discretionary: demand + easy credit + moderate long rates."),
    ("XLU","Utilities: bond-proxy; lower yields and defensiveness help."),
    ("XLP","Staples: defensive; relative strength when growth slows."),
    ("XLC","Comm Services: growth/advertising tilt; lower real yields supportive."),
    ("XLRE","REITs: long rates/term premium drive cap rates; credit spreads matter."),
    ("GLD","Gold: inverse to real yields and USD."),
    ("RUT","Small caps: growth + steepening + tight spreads."),
    ("YINN","China proxy (3×): global cycle + softer USD aid exports/commodities."),
    ("SOXL","Semis: cyclical + duration; lower real yields & tight spreads help."),
    ("FAS","3× Financials: magnified curve/credit sensitivity."),
    ("BNKU","3× Big Banks: steeper curve improves NIM; tight spreads support credit."),
    ("DRN","3× Real Estate: long duration/term premium sensitive; credit helps."),
    ("NAIL","3× Homebuilders: lower mortgage/long rates + expanding orders."),
    ("RETL","3× Retail: demand + easy credit; very high gas can weigh."),
    ("TMF","3× Long Treasuries: fall in yields/term prem/real yields."),
    ("BITX","2× Bitcoin proxy: easier liquidity (lower real yields/USD)."),
    ("ETHU","2× Ether proxy: similar macro to BTC."),
    ("DFEN","3× Defense/Aero: funding/geopolitics support."),
    ("EUAD","Europe A&D: budgets/orders tailwinds."),
]

def signal_for(tk: str) -> str:
    if tk=="XLE":
        return "GREEN" if (pmi>=TH["pmi_green_min"] and dxy<=TH["dxy_green_max"] and wti>=TH["wti_green_min"]) else ("YELLOW" if (pmi>=TH["pmi_yellow_min"]) else "RED")
    if tk=="XLB":
        return "GREEN" if (pmi>=TH["pmi_green_min"] and dxy<=TH["dxy_green_max"] and (oas*100)<TH["hyoas_yellow_max_bps"]) else ("YELLOW" if pmi>=TH["pmi_yellow_min"] else "RED")
    if tk=="XLI":
        return "GREEN" if (pmi>=TH["pmi_green_min"] and (curve_bps)>=TH["curve_yellow_min_bps"]) else ("YELLOW" if pmi>=TH["pmi_yellow_min"] else "RED")
    if tk in ("XLF","FAS","BNKU"):
        return "GREEN" if ((curve_bps)>=TH["curve_green_min_bps"] and (oas*100)<=TH["hyoas_yellow_max_bps"]) else ("YELLOW" if ((curve_bps)>=TH["curve_yellow_min_bps"] and (oas*100)<=TH["hyoas_yellow_max_bps"]) else "RED")
    if tk in ("XLK","XLC","SOXL"):
        return "GREEN" if (tipsd<TH["tips_green_max"] and pmi>=TH["pmi_yellow_min"]) else ("YELLOW" if tipsd<TH["tips_yellow_max"] else "RED")
    if tk in ("XLV","XLP"):
        return "GREEN" if (pmi<TH["pmi_yellow_min"] or (oas*100)>=TH["hyoas_yellow_max_bps"]) else "YELLOW"
    if tk=="XLY" or tk=="RETL":
        return "GREEN" if (pmi>=TH["pmi_green_min"] and (oas*100)<TH["hyoas_green_max_bps"] and y10d<TH["y10_green_max"]) else ("YELLOW" if pmi>=TH["pmi_yellow_min"] else "RED")
    if tk=="XLU":
        return "GREEN" if (y10d<TH["y10_green_max"] or (oas*100)>=TH["hyoas_yellow_max_bps"]) else "YELLOW"
    if tk=="XLRE" or tk=="DRN":
        return "GREEN" if (y10d<TH["y10_green_max"] and tp10d<TH["tp10_green_max"] and (oas*100)<TH["hyoas_yellow_max_bps"]) else ("YELLOW" if (y10d<TH["y10_yellow_max"] and (oas*100)<(TH["hyoas_yellow_max_bps"]+50)) else "RED")
    if tk=="GLD":
        return "GREEN" if (tipsd<0.016 and dxy<=TH["dxy_green_max"]) else ("YELLOW" if (tipsd<TH["tips_yellow_max"] or dxy<=TH["dxy_yellow_max"]) else "RED")
    if tk=="RUT":
        return "GREEN" if (pmi>=TH["pmi_green_min"] and (curve_bps)>=TH["curve_green_min_bps"] and (oas*100)<TH["hyoas_yellow_max_bps"]) else ("YELLOW" if (pmi>=TH["pmi_yellow_min"] and (oas*100)<TH["hyoas_yellow_max_bps"]) else "RED")
    if tk in ("NAIL",):
        return "GREEN" if (y10d<TH["y10_green_max"] and pmi>=TH["pmi_green_min"]) else ("YELLOW" if y10d<TH["y10_yellow_max"] else "RED")
    if tk=="TMF":
        return "GREEN" if (y10d<TH["y10_green_max"] and tp10d<TH["tp10_green_max"] and tipsd<TH["tips_green_max"]) else ("YELLOW" if (y10d<TH["y10_yellow_max"] or tp10d<TH["tp10_yellow_max"]) else "RED")
    if tk in ("BITX","ETHU"):
        return "GREEN" if (tipsd<0.020 and dxy<=TH["dxy_yellow_max"]) else "YELLOW"
    if tk in ("DFEN","EUAD"):
        return "GREEN" if (pmi>=TH["pmi_yellow_min"] or (oas*100)<550) else "YELLOW"
    if tk=="YINN":
        return "GREEN" if (pmi>=TH["pmi_green_min"] and dxy<=TH["dxy_green_max"]) else ("YELLOW" if pmi>=TH["pmi_yellow_min"] else "RED")
    return "SETUP"

rows = [(tk, signal_for(tk), note) for tk, note in ETF_ROWS]
etf_df = pd.DataFrame(rows, columns=["Ticker","Signal","Why it moves"])

def style_signals(df):
    def colorize(val):
        if val=="GREEN": return "background-color:#C6EFCE; font-weight:600;"
        if val=="YELLOW": return "background-color:#FFEB9C; font-weight:600;"
        if val=="RED": return "background-color:#FFC7CE; font-weight:600;"
        return ""
    return df.style.applymap(colorize, subset=["Signal"])

st.subheader("Signals")
st.dataframe(style_signals(etf_df), use_container_width=True, height=600)

# =========================================================
# 7) Export to Excel
# =========================================================
def make_export():
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xls:
        etf_df.to_excel(xls, sheet_name="Signals", index=False)
        latest_tbl = pd.DataFrame({
            "Indicator":["Y10","Y2","Y3M","TIPS","TP10","BROAD$","PMI NO","HY OAS (bps)","10s-2s (bps)","WTI"],
            "Value":[y10d*100 if pd.notna(y10d) else np.nan,
                     y2d*100 if pd.notna(y2d) else np.nan,
                     y3md*100 if pd.notna(y3md) else np.nan,
                     tipsd*100 if pd.notna(tipsd) else np.nan,
                     tp10d*100 if pd.notna(tp10d) else np.nan,
                     dxy, pmi, oas*100 if pd.notna(oas) else np.nan, curve_bps, wti]
        })
        latest_tbl.to_excel(xls, sheet_name="MacroTiles", index=False)
    return out.getvalue()

st.download_button("Download Excel snapshot", data=make_export(),
                   file_name="ETF_Macro_Signals.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Edit thresholds in config.yaml, then rerun.")
