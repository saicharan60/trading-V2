import os
import io
import gzip
import json
import math
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Market Lens + Upstox Swing Scanner",
    page_icon="📈",
    layout="wide",
)

UPSTOX_BASE = "https://api.upstox.com/v3"

UPSTOX_HIST_URL = (
    UPSTOX_BASE +
    "/historical-candle/{instrument_key}/days/1/{to_date}/{from_date}"
)

UPSTOX_QUOTES_URL = (
    "https://api.upstox.com/v3/market-quote/quotes"
)

UPSTOX_INSTRUMENT_URL = (
    "https://assets.upstox.com/"
    "market-quote/instruments/exchange/NSE.json.gz"
)

# NSE NIFTY 500 list
NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/"
    "ind_nifty500list.csv"
)


# ============================================================
# SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "SwingScanner/1.0",
})


# ============================================================
# TOKEN
# ============================================================

def get_token():

    token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()

    if token:
        return token

    try:
        token = st.secrets["UPSTOX_ACCESS_TOKEN"]
        return str(token).strip()
    except Exception:
        return ""


def upstox_headers():

    token = get_token()

    if not token:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN is not configured."
        )

    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }


# ============================================================
# NIFTY 500
# ============================================================

@st.cache_data(ttl=86400)
def load_nifty500():

    r = requests.get(
        NIFTY500_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv",
        },
        timeout=30,
    )

    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))

    cols = {c.lower(): c for c in df.columns}

    symbol_col = cols.get("symbol")
    company_col = (
        cols.get("company name")
        or cols.get("company")
    )

    if not symbol_col:
        raise RuntimeError(
            "Could not find Symbol column in NIFTY 500 file."
        )

    out = pd.DataFrame()

    out["Symbol"] = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if company_col:
        out["Company"] = (
            df[company_col]
            .astype(str)
            .str.strip()
        )
    else:
        out["Company"] = out["Symbol"]

    return out


# ============================================================
# UPSTOX INSTRUMENT MASTER
# ============================================================

@st.cache_data(ttl=86400)
def load_upstox_instruments():

    r = requests.get(
        UPSTOX_INSTRUMENT_URL,
        timeout=60,
    )

    r.raise_for_status()

    raw = gzip.decompress(r.content)

    records = json.loads(
        raw.decode("utf-8")
    )

    df = pd.DataFrame(records)

    # Equity only
    df = df[
        (df["segment"] == "NSE_EQ") &
        (
            df["instrument_type"]
            .isin(["EQ", "BE"])
        )
    ].copy()

    df["trading_symbol"] = (
        df["trading_symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df[
        [
            "trading_symbol",
            "instrument_key",
            "isin",
            "name",
        ]
    ]


# ============================================================
# MAP SYMBOL -> INSTRUMENT KEY
# ============================================================

@st.cache_data(ttl=86400)
def create_symbol_map():

    inst = load_upstox_instruments()

    return dict(
        zip(
            inst["trading_symbol"],
            inst["instrument_key"],
        )
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

def get_daily_candles(
    instrument_key: str,
    days: int = 365,
):

    today = datetime.now().date()

    to_date = today.strftime("%Y-%m-%d")

    from_date = (
        today - timedelta(days=days)
    ).strftime("%Y-%m-%d")

    url = UPSTOX_HIST_URL.format(
        instrument_key=instrument_key,
        to_date=to_date,
        from_date=from_date,
    )

    r = SESSION.get(
        url,
        headers=upstox_headers(),
        timeout=30,
    )

    if r.status_code != 200:
        return pd.DataFrame()

    payload = r.json()

    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "oi",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oi",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["ema20"] = (
        df["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["ema200"] = (
        df["close"]
        .ewm(span=200, adjust=False)
        .mean()
    )

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )

    avg_loss = (
        loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = (
        100 -
        (100 / (1 + rs))
    )

    # MACD
    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # ATR
    prev_close = df["close"].shift(1)

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        prev_close
    ).abs()

    tr3 = (
        df["low"] -
        prev_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr14"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    # Volume
    df["volume20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume20"]
    )

    # Recent resistance
    df["high20"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    # Recent support
    df["low20"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    return df


# ============================================================
# SWING SIGNAL
# ============================================================

def calculate_signal(df):

    if len(df) < 220:
        return None

    row = df.iloc[-1]

    price = row["close"]
    atr = row["atr14"]

    if pd.isna(atr) or atr <= 0:
        return None

    score = 0
    reasons = []

    # --------------------------------
    # TREND
    # --------------------------------

    if (
        price > row["ema20"] >
        row["ema50"]
    ):
        score += 20
        reasons.append(
            "Price > EMA20 > EMA50"
        )

    if price > row["ema200"]:
        score += 15
        reasons.append(
            "Above EMA200"
        )

    # --------------------------------
    # RSI
    # --------------------------------

    if 50 <= row["rsi"] <= 68:
        score += 10
        reasons.append(
            "Healthy RSI"
        )

    elif 45 <= row["rsi"] < 50:
        score += 5

    # --------------------------------
    # MACD
    # --------------------------------

    if (
        row["macd"] >
        row["macd_signal"]
    ):
        score += 15
        reasons.append(
            "MACD bullish"
        )

    # --------------------------------
    # VOLUME
    # --------------------------------

    if row["volume_ratio"] >= 1.2:
        score += 15
        reasons.append(
            "Volume confirmation"
        )

    elif row["volume_ratio"] >= 1:
        score += 5

    # --------------------------------
    # BREAKOUT
    # --------------------------------

    breakout = False

    if (
        not pd.isna(row["high20"])
        and price > row["high20"]
    ):
        breakout = True
        score += 15
        reasons.append(
            "20-day breakout"
        )

    # --------------------------------
    # PULLBACK
    # --------------------------------

    distance_ema20 = (
        abs(price - row["ema20"])
        / price
    )

    pullback = (
        distance_ema20 <= 0.025
        and price > row["ema20"]
    )

    if pullback:
        score += 10
        reasons.append(
            "EMA20 pullback"
        )

    # --------------------------------
    # ENTRY
    # --------------------------------

    if breakout:

        entry = max(
            price,
            row["high20"] * 1.002
        )

        setup = "BREAKOUT"

    elif pullback:

        entry = price * 1.002

        setup = "PULLBACK"

    else:

        entry = price

        setup = "TREND"

    # --------------------------------
    # STOP LOSS
    # --------------------------------

    atr_sl = entry - (
        1.5 * atr
    )

    swing_sl = row["low20"]

    candidates = [
        atr_sl,
        swing_sl
    ]

    valid_sl = [
        x for x in candidates
        if not pd.isna(x)
        and x < entry
    ]

    if not valid_sl:
        return None

    # Use the higher of ATR SL and
    # recent support to avoid an
    # unnecessarily wide stop.
    stop_loss = max(valid_sl)

    risk = entry - stop_loss

    if risk <= 0:
        return None

    # --------------------------------
    # TARGETS
    # --------------------------------

    target1 = entry + (
        risk * 2
    )

    target2 = entry + (
        risk * 3
    )

    # --------------------------------
    # RISK / REWARD
    # --------------------------------

    rr1 = (
        target1 - entry
    ) / risk

    rr2 = (
        target2 - entry
    ) / risk

    # --------------------------------
    # SIGNAL
    # --------------------------------

    if score >= 75:
        signal = "BUY"

    elif score >= 60:
        signal = "WATCH"

    else:
        signal = "AVOID"

    return {
        "signal": signal,
        "setup": setup,
        "score": score,
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "risk": risk,
        "rr1": rr1,
        "rr2": rr2,
        "rsi": row["rsi"],
        "atr": atr,
        "volume_ratio": row["volume_ratio"],
        "ema20": row["ema20"],
        "ema50": row["ema50"],
        "ema200": row["ema200"],
        "reasons": reasons,
    }


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_quantity(
    capital,
    risk_percent,
    entry,
    stop_loss,
):

    max_risk = (
        capital *
        risk_percent /
        100
    )

    risk_per_share = (
        entry - stop_loss
    )

    if risk_per_share <= 0:
        return 0

    quantity = math.floor(
        max_risk /
        risk_per_share
    )

    return max(
        quantity,
        0
    )


# ============================================================
# SCAN ONE STOCK
# ============================================================

def scan_stock(
    symbol,
    instrument_key,
    company,
):

    df = get_daily_candles(
        instrument_key,
        days=400
    )

    if df.empty:
        return None

    df = calculate_indicators(df)

    signal = calculate_signal(df)

    if not signal:
        return None

    return {
        "Symbol": symbol,
        "Company": company,
        **signal,
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "📈 Market Lens + Upstox Swing Trading Scanner"
)

st.caption(
    "Daily trend + breakout/pullback scanner "
    "with Entry, Stop Loss and Targets"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Scanner Settings")

    universe = st.selectbox(
        "Universe",
        [
            "NIFTY 500",
            "Custom symbols",
        ]
    )

    capital = st.number_input(
        "Trading Capital",
        min_value=10000.0,
        value=500000.0,
        step=10000.0,
    )

    risk_percent = st.number_input(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

    min_score = st.slider(
        "Minimum Score",
        0,
        100,
        60
    )

    max_stocks = st.number_input(
        "Maximum stocks to scan",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

    run_scan = st.button(
        "🚀 Run Swing Scan",
        type="primary",
    )


# ============================================================
# TOKEN CHECK
# ============================================================

token = get_token()

if not token:

    st.warning(
        "Configure UPSTOX_ACCESS_TOKEN first."
    )

    st.code(
        """
# Linux/macOS
export UPSTOX_ACCESS_TOKEN="YOUR_ANALYTICS_TOKEN"

# Windows PowerShell
$env:UPSTOX_ACCESS_TOKEN="YOUR_ANALYTICS_TOKEN"
        """
    )

    st.stop()


# ============================================================
# MAIN SCAN
# ============================================================

if run_scan:

    progress = st.progress(0)

    status = st.empty()

    try:

        # ---------------------------------------------
        # Load universe
        # ---------------------------------------------

        if universe == "NIFTY 500":

            universe_df = load_nifty500()

        else:

            text = st.text_area(
                "Enter symbols",
                "RELIANCE,TCS,INFY,ICICIBANK,HDFCBANK"
            )

            symbols = [
                x.strip().upper()
                for x in text.split(",")
                if x.strip()
            ]

            universe_df = pd.DataFrame({
                "Symbol": symbols,
                "Company": symbols,
            })

        universe_df = (
            universe_df
            .head(int(max_stocks))
        )

        # ---------------------------------------------
        # Instrument mapping
        # ---------------------------------------------

        status.write(
            "Loading Upstox instrument master..."
        )

        symbol_map = create_symbol_map()

        universe_df["InstrumentKey"] = (
            universe_df["Symbol"]
            .map(symbol_map)
        )

        universe_df = (
            universe_df
            .dropna(
                subset=["InstrumentKey"]
            )
            .reset_index(drop=True)
        )

        total = len(universe_df)

        if total == 0:

            st.error(
                "No NSE instruments matched."
            )

            st.stop()

        # ---------------------------------------------
        # Scan
        # ---------------------------------------------

        results = []

        for i, row in universe_df.iterrows():

            symbol = row["Symbol"]

            status.write(
                f"Scanning {symbol} "
                f"({i + 1}/{total})"
            )

            try:

                result = scan_stock(
                    symbol,
                    row["InstrumentKey"],
                    row["Company"],
                )

                if result:
                    results.append(result)

            except Exception as e:

                print(
                    f"{symbol}: {e}"
                )

            progress.progress(
                int(
                    ((i + 1) / total)
                    * 100
                )
            )

            # avoid hammering API
            time.sleep(0.05)

        # ---------------------------------------------
        # Results
        # ---------------------------------------------

        if not results:

            st.warning(
                "No valid setups found."
            )

            st.stop()

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df[
            results_df["score"]
            >= min_score
        ].copy()

        results_df = results_df.sort_values(
            "score",
            ascending=False
        )

        # ---------------------------------------------
        # Position size
        # ---------------------------------------------

        results_df["Qty"] = results_df.apply(
            lambda x:
            calculate_quantity(
                capital,
                risk_percent,
                x["entry"],
                x["stop_loss"],
            ),
            axis=1,
        )

        # ---------------------------------------------
        # Format display
        # ---------------------------------------------

        display_cols = [
            "Symbol",
            "signal",
            "setup",
            "score",
            "entry",
            "stop_loss",
            "target1",
            "target2",
            "rr1",
            "rr2",
            "rsi",
            "volume_ratio",
            "Qty",
        ]

        st.subheader(
            "🎯 Swing Trading Opportunities"
        )

        st.dataframe(
            results_df[
                display_cols
            ].rename(
                columns={
                    "Symbol": "Stock",
                    "signal": "Signal",
                    "setup": "Setup",
                    "score": "Score",
                    "entry": "Buy Price",
                    "stop_loss": "SL",
                    "target1": "Target 1",
                    "target2": "Target 2",
                    "rr1": "R:R T1",
                    "rr2": "R:R T2",
                    "rsi": "RSI",
                    "volume_ratio": "Volume Ratio",
                    "Qty": "Qty",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------
        # BUY setups
        # ---------------------------------------------

        buy_df = results_df[
            results_df["signal"]
            == "BUY"
        ].copy()

        st.subheader(
            f"🟢 BUY Setups ({len(buy_df)})"
        )

        if buy_df.empty:

            st.info(
                "No BUY setup meets the score threshold."
            )

        else:

            for _, r in buy_df.head(10).iterrows():

                with st.container():

                    c1, c2, c3, c4, c5 = (
                        st.columns(5)
                    )

                    c1.metric(
                        "Stock",
                        r["Symbol"]
                    )

                    c2.metric(
                        "Buy",
                        f"₹{r['entry']:.2f}"
                    )

                    c3.metric(
                        "SL",
                        f"₹{r['stop_loss']:.2f}"
                    )

                    c4.metric(
                        "Target 1",
                        f"₹{r['target1']:.2f}"
                    )

                    c5.metric(
                        "Target 2",
                        f"₹{r['target2']:.2f}"
                    )

                    st.write(
                        f"**Setup:** {r['setup']} | "
                        f"**Score:** {r['score']}/100 | "
                        f"**RSI:** {r['rsi']:.1f} | "
                        f"**Volume:** "
                        f"{r['volume_ratio']:.2f}x | "
                        f"**Qty:** {int(r['Qty'])}"
                    )

                    st.write(
                        "**Reasons:** "
                        + ", ".join(
                            r["reasons"]
                        )
                    )

                    st.divider()

        # ---------------------------------------------
        # DOWNLOAD
        # ---------------------------------------------

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Scan Results",
            csv,
            file_name=(
                "swing_scan_results.csv"
            ),
            mime="text/csv",
        )

        status.success(
            f"Scan completed. "
            f"{len(results_df)} setups found."
        )

    except Exception as e:

        st.error(
            f"Scanner error: {e}"
        )

else:

    st.info(
        "Configure the token and click "
        "**Run Swing Scan**."
    )
