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
                out
