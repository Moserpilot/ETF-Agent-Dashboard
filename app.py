1  import os, io, time
2  from datetime import datetime
3  import pandas as pd
4  import numpy as np
5  import requests
6  import streamlit as st
7  import yaml
8
9  st.set_page_config(page_title="ETF Macro Agent Dashboard", layout="wide")
10
11 # =========================================================
12 # 1) Load config
13 # =========================================================
14 CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
15 with open(CFG_PATH, "r") as f:
16     CFG = yaml.safe_load(f)
17 TH = CFG["tiles"]
18
19 # =========================================================
20 # 2) Robust data fetchers (no API keys)
21 # =========================================================
22 @st.cache_data(ttl=60*15, show_spinner=False)
23 def fetch_fred_series(series_id: str) -> pd.DataFrame:
24     """
25     Generic FRED series fetch WITHOUT API key.
26     Tries the 'downloaddata' CSV first; falls back to fredgraph.csv.
27     Returns columns: Date, Name, Value
28     """
29     headers = {
30         "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
31                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
32     }
33     urls = [
34         f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
35         f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
36     ]
37     last_err = None
38     for url in urls:
39         for _ in range(2):
40             try:
41                 r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
42                 r.raise_for_status()
43                 df = pd.read_csv(io.StringIO(r.text))
44                 # Normalize likely column names
45                 if "DATE" in df.columns:
46                     date_col = "DATE"
47                 elif "observation_date" in df.columns:
48                     date_col = "observation_date"
49                 else:
50                     date_col = df.columns[0]
51                 # Value column is often the series id; otherwise use 2nd column
52                 val_col = series_id if series_id in df.columns else df.columns[1]
53                 out = df[[date_col, val_col]].copy()
54                 out.columns = ["Date", "Value"]
55                 out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
56                 out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
57                 out["Name"]  = series_id
58                 return out.dropna(subset=["Date"])
59             except Exception as e:
60                 last_err = e
61                 time.sleep(1.0)
62                 continue
63     raise RuntimeError(f"Failed to fetch {series_id}: {last_err}")
64
65 @st.cache_data(ttl=60*30, show_spinner=False)
66 def fetch_tp10() -> pd.DataFrame:
67     """
68     Robust NY Fed term premium fetch (no API).
69     Tries several official URLs, follows redirects, and strips any preamble
70     before the actual CSV header. Returns Date/Value for TP10.
71     """
72     url_candidates = [
73         "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTP.csv",
74         "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTP.csv?download=true",
75         "https://www.newyorkfed.org/research/data_indicators/ACMTP.csv",
76         "https://nyfed.org/medialibrary/media/research/data_indicators/ACMTP.csv",
77     ]
78     headers = {
79         "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
80                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
81     }
82     last_err = None
83     for url in url_candidates:
84         try:
85             r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
86             r.raise_for_status()
87             text = r.text
88             # Remove any preamble; find the header row containing both Date and TP10
89             lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
90             hdr_idx = next(i for i, ln in enumerate(lines)
91                            if ("date" in ln.lower() and "tp10" in ln.lower()))
92             csv_text = "\n".join(lines[hdr_idx:])
93             df = pd.read_csv(io.StringIO(csv_text))
94             df = df[["Date", "TP10"]].copy()
95             df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
96             df["TP10"] = pd.to_numeric(df["TP10"], errors="coerce")
97             out = df.rename(columns={"TP10": "Value"})
98             out["Name"] = "TP10"
99             return out.dropna(subset=["Date"])
100        except Exception as e:
101            last_err = e
102            continue
103    raise RuntimeError(f"Failed to fetch TP10: {last_err}")
104
105# =========================================================
106# 3) Small helpers
107# =========================================================
108def latest_value(df_all: pd.DataFrame, name: str):
109    """Return latest value and date for a series name from the combined data table."""
110    sub = df_all[df_all["Name"] == name]
111    if sub.empty:
112        return np.nan, None
113    row = sub.sort_values("Date").iloc[-1]
114    return float(row["Value"]), pd.to_datetime(row["Date"]).date()
115
116def color_from_rule(val, green_cond, yellow_cond):
117    """Tile background color: green / yellow / red / gray (missing)."""
118    try:
119        if val is None or (isinstance(val, float) and np.isnan(val)):
120            return "#EEE"
121        if green_cond(val):
122            return "#C6EFCE"
123        if yellow_cond(val):
124            return "#FFEB9C"
125        return "#FFC7CE"
126    except Exception:
127        return "#EEE"
128
129# =========================================================
130# 4) Load all data (tolerant)
131# =========================================================
132with st.spinner("Fetching data..."):
133    frames, fetch_errors = [], []
134    for sid in CFG["series"]["fred"]:
135        try:
136            frames.append(fetch_fred_series(sid))
137        except Exception as e:
138            fetch_errors.append(f"{sid}: {e}")
139
140    if CFG["series"].get("nyfed_tp10", True):
141        try:
142            frames.append(fetch_tp10())
143        except Exception as e:
144            fetch_errors.append(f"TP10: {e}")
145
146    if not frames:
147        st.error("No data could be loaded. Please refresh or try again.")
148        st.stop()
149
150    data = pd.concat(frames, ignore_index=True).dropna(subset=["Value"])
151
152    if fetch_errors:
153        st.warning("Some series failed to load:\n- " + "\n- ".join(fetch_errors))
154
155# =========================================================
156# 5) Compute macro tiles
157# =========================================================
158y10, y10_date = latest_value(data, "DGS10")
159y2, _  = latest_value(data, "DGS2")
160y3m, _ = latest_value(data, "DGS3MO")
161tips, _ = latest_value(data, "DFII10")
162tp10, _ = latest_value(data, "TP10")
163dxy, _  = latest_value(data, "DTWEXBGS")   # broad USD index proxy (stable)
164pmi, _  = latest_value(data, "NAPM")       # <-- ISM PMI composite proxy (no API key)
165oas, _  = latest_value(data, "BAMLH0A0HYM2")
166wti, _  = latest_value(data, "DCOILWTICO")
167
168def to_decimal(x):
169    if pd.isna(x): return np.nan
170    return x/100 if x > 1 else x
171
172y10d, y2d, y3md, tipsd, tp10d = map(to_decimal, [y10, y2, y3m, tips, tp10])
173curve_bps = (y10d - y2d) * 10000 if (pd.notna(y10d) and pd.notna(y2d)) else np.nan
174
175tiles = [
176    ("10Y UST (%)", y10d*100 if pd.notna(y10d) else np.nan, lambda v: v/100 < TH["y10_green_max"], lambda v: v/100 < TH["y10_yellow_max"]),
177    ("2Y UST (%)",  y2d*100  if pd.notna(y2d)  else np.nan, lambda v: True, lambda v: True),
178    ("3M Bill (%)", y3md*100 if pd.notna(y3md) else np.nan, lambda v: True, lambda v: True),
179    ("10Y TIPS (%)",tipsd*100 if pd.notna(tipsd) else np.nan, lambda v: v/100 < TH["tips_green_max"], lambda v: v/100 < TH["tips_yellow_max"]),
180    ("Term Prem (%)",tp10d*100 if pd.notna(tp10d) else np.nan, lambda v: v/100 < TH["tp10_green_max"], lambda v: v/100 < TH["tp10_yellow_max"]),
181    ("Broad $ Index", dxy, lambda v: v <= TH["dxy_green_max"], lambda v: v <= TH["dxy_yellow_max"]),
182    ("PMI (ISM)", pmi, lambda v: v >= TH["pmi_green_min"], lambda v: v >= TH["pmi_yellow_min"]),
183    ("HY OAS (bps)", oas*100 if pd.notna(oas) else np.nan, lambda v: v <= TH["hyoas_green_max_bps"], lambda v: v <= TH["hyoas_yellow_max_bps"]),
184    ("10s–2s (bps)", curve_bps, lambda v: v >= TH["curve_green_min_bps"], lambda v: v >= TH["curve_yellow_min_bps"]),
185    ("WTI ($)", wti, lambda v: v >= TH["wti_green_min"], lambda v: v >= TH["wti_yellow_min"]),
186]
187
188st.title("ETF Macro Agent Dashboard")
189st.caption("Data: FRED (no API) & NY Fed. USD tile uses FRED Broad Dollar Index (DTWEXBGS). Thresholds in config.yaml.")
190
191cols = st.columns(len(tiles))
192for i, (label, val, green_fn, yellow_fn) in enumerate(tiles):
193    bg = color_from_rule(val, green_fn, yellow_fn)
194    with cols[i]:
195        st.markdown(
196            f"""
197            <div style="background-color:{bg}; padding:12px; border-radius:8px; text-align:center; border:1px solid #ddd;">
198                <div style="font-size:12px; color:#333;">{label}</div>
199                <div style="font-size:24px; font-weight:700; color:#111;">{'' if pd.isna(val) else (f"{val:,.2f}")}</div>
200            </div>
201            """,
202            unsafe_allow_html=True
203
