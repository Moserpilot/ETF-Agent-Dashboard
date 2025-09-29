import os, io
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import streamlit as st
from dateutil import tz
import plotly.graph_objects as go
import yaml

st.set_page_config(page_title="ETF Macro Agent Dashboard", layout="wide")

# ---------- Load config ----------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

TH = CFG["tiles"]

# ---------- Helpers ----------
@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """
    Downloads a FRED series from the stable 'downloaddata' endpoint.
    Returns columns: Date, Name, Value
    """
    url = f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Normalize columns
    date_col = df.columns[0]
    val_col = df.columns[1]
    df = df[[date_col, val_col]].copy()
    df.columns = ["Date", "Value"]
    # Convert
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Some FRED series have '.' for missing values
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Name"] = series_id
    return df.dropna(subset=["Date"])

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_tp10() -> pd.DataFrame:
    url = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTP.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df[["Date", "TP10"]].copy()
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Name"] = "TP10"
    return df.dropna(subset=["Date"])

def latest_value(df_all: pd.DataFrame, name: str):
    sub = df_all[df_all["Name"] == name]
    if sub.empty:
        return np.nan, None
    row = sub.sort_values("Date").iloc[-1]
    return float(row["Value"]), pd.to_datetime(row["Date"]).date()

def color_from_rule(val, green_cond, yellow_cond):
    """Return CSS color for tile backgrounds: green/yellow/red"""
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

# ---------- Load all data ----------
with st.spinner("Fetching data..."):
    frames = []
    for sid in CFG["series"]["fred"]:
        frames.append(fetch_fred_series(sid))
    if CFG["series"].get("nyfed_tp10", True):
        frames.append(fetch_tp10())
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["Value"])

# ---------- Compute macro tiles ----------
y10, y10_date = latest_value(data, "DGS10")
y2, _  = latest_value(data, "DGS2")
y3m, _ = latest_value(data, "DGS3MO")
tips, _ = latest_value(data, "DFII10")
tp10, _ = latest_value(data, "TP10")
dxy, _  = latest_value(data, "DTWEXBGS")  # broad USD index
pmi, _  = latest_value(data, "NAPMNO")
oas, _  = latest_value(data, "BAMLH0A0HYM2")
wti, _  = latest_value(data, "DCOILWTICO")

# Convert to decimals if needed (FRED yields sometimes look like percentages)
def to_decimal(x):
    if pd.isna(x): return np.nan
    return x/100 if x>1 else x

y10d = to_decimal(y10)
y2d  = to_decimal(y2)
y3md = to_decimal(y3m)
tipsd= to_decimal(tips)
tp10d= to_decimal(tp10)

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

# Top tiles
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

# ---------- ETF mapping & signals ----------
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

rows = []
for tk, note in ETF_ROWS:
    rows.append((tk, signal_for(tk), note))

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

# Export
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
