import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="GOLDEN SMART MONEY AI | Scalping", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 GOLDEN SMART MONEY AI")
st.caption("Scalping BEI • Intraday 5m/15m • VWAP • EMA • MACD • RSI • Risk-based lot sizing")
st.warning(
    "Alat bantu analisis, bukan robot auto-trading. Data Yahoo Finance dapat terlambat/tidak lengkap. "
    "Sinyal hanya kandidat; periksa order book, spread, antrean, dan kondisi pasar sebelum transaksi."
)

DEFAULT_CAPITAL = 10_000_000
DEFAULT_RISK = 2.0
DEFAULT_POSITIONS = 2

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr(df, period=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def normalize_validation_file(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    # Format file validasi yang diunggah pengguna
    if "ticker yahoo" in frame.columns:
        frame["ticker"] = frame["ticker yahoo"].astype(str).str.strip().str.upper()
    if "status data" in frame.columns:
        frame = frame[frame["status data"].astype(str).str.upper().eq("TERSEDIA")]
    if "kode" not in frame.columns and "kode " in frame.columns:
        frame["kode"] = frame["kode "]
    if "ticker" not in frame.columns:
        raise ValueError("CSV harus memiliki kolom ticker atau Ticker Yahoo.")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["ticker"].str.endswith(".JK")]
    if "kode" not in frame.columns:
        frame["kode"] = frame["ticker"].str.replace(".JK", "", regex=False)
    if "last_data" not in frame.columns:
        frame["last_data"] = frame.get("tanggal terakhir", "")
    frame = frame.drop_duplicates("ticker")
    return frame[["ticker", "kode", "last_data"]].reset_index(drop=True)

def extract_symbol_frame(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance biasanya mengembalikan (Price, Ticker)
        if ticker in data.columns.get_level_values(-1):
            out = data.xs(ticker, axis=1, level=-1, drop_level=True)
        elif ticker in data.columns.get_level_values(0):
            out = data.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            return pd.DataFrame()
    else:
        out = data.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame()
    return out[required].dropna(subset=["Close"])

@st.cache_data(ttl=300, show_spinner=False)
def download_daily(tickers):
    result = {}
    # Batch agar tidak membuat ratusan koneksi terpisah
    for start in range(0, len(tickers), 40):
        batch = tickers[start:start+40]
        try:
            data = yf.download(
                tickers=batch, period="3mo", interval="1d",
                auto_adjust=True, group_by="ticker", progress=False,
                threads=True, timeout=25
            )
            for ticker in batch:
                frame = extract_symbol_frame(data, ticker)
                if len(frame) >= 30:
                    result[ticker] = frame
        except Exception:
            continue
    return result

@st.cache_data(ttl=90, show_spinner=False)
def download_intraday(tickers, interval):
    result = {}
    for start in range(0, len(tickers), 20):
        batch = tickers[start:start+20]
        try:
            data = yf.download(
                tickers=batch, period="5d", interval=interval,
                auto_adjust=True, group_by="ticker", progress=False,
                threads=True, timeout=25, prepost=False
            )
            for ticker in batch:
                frame = extract_symbol_frame(data, ticker)
                if len(frame) >= 25:
                    result[ticker] = frame
        except Exception:
            continue
    return result

def daily_quality(frame):
    close, vol = frame["Close"], frame["Volume"]
    turnover = close * vol
    avg_turnover = safe_float(turnover.tail(20).mean(), 0)
    avg_volume = safe_float(vol.tail(20).mean(), 0)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ma50 = close.rolling(50).mean()
    last = safe_float(close.iloc[-1])
    trend = bool(last > ema20.iloc[-1] and ema20.iloc[-1] > ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else bool(last > ema20.iloc[-1])
    return {
        "last_daily": last,
        "avg_turnover": avg_turnover,
        "avg_volume": avg_volume,
        "daily_trend": trend,
        "daily_change": safe_float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0,
        "daily_date": str(frame.index[-1])[:10]
    }

def analyze_intraday(ticker, frame, daily_info, capital, risk_pct, max_positions):
    frame = frame.copy().dropna()
    if len(frame) < 25:
        return None
    # Gunakan candle terakhir yang telah selesai; candle paling akhir berpotensi masih berjalan.
    x = frame.iloc[:-1].copy() if len(frame) > 26 else frame.copy()
    if len(x) < 25:
        return None
    close, high, low, volume = x["Close"], x["High"], x["Low"], x["Volume"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    rsi_now = safe_float(rsi(close).iloc[-1], 50)
    typical = (high + low + close) / 3
    # VWAP dihitung ulang setiap tanggal perdagangan.
    dates = pd.Index(x.index.date)
    pv = typical * volume
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum().replace(0, np.nan)
    vwap_series = cum_pv / cum_vol
    vwap_now = safe_float(vwap_series.iloc[-1])
    price = safe_float(close.iloc[-1])
    vol_avg = safe_float(volume.iloc[-21:-1].mean(), 0)
    vol_ratio = safe_float(volume.iloc[-1] / vol_avg, 0) if vol_avg > 0 else 0
    atr_now = safe_float(atr(x, 14).iloc[-1], price * 0.005)
    support = safe_float(low.tail(8).min(), price * 0.98)
    ema9_now, ema20_now = safe_float(ema9.iloc[-1]), safe_float(ema20.iloc[-1])
    macd_bull = bool(macd.iloc[-1] > macd_sig.iloc[-1])
    macd_rising = bool(macd.iloc[-1] > macd.iloc[-2])
    above_vwap = bool(price > vwap_now) if np.isfinite(vwap_now) else False
    intraday_up = price > ema9_now > ema20_now
    recent_high = safe_float(high.iloc[-7:-1].max(), price)
    breakout = price > recent_high
    # Likuiditas rata-rata nilai transaksi 20 hari; default minimal Rp1 miliar.
    avg_turnover = daily_info["avg_turnover"]
    liquidity_ok = avg_turnover >= 1_000_000_000
    # Skor adalah ranking teknikal, bukan probabilitas profit.
    score = 0
    score += 20 if daily_info["daily_trend"] else 0
    score += 20 if intraday_up else 0
    score += 15 if above_vwap else 0
    score += 10 if macd_bull else 0
    score += 5 if macd_rising else 0
    score += 10 if 50 <= rsi_now <= 68 else (5 if 45 <= rsi_now < 50 or 68 < rsi_now <= 72 else 0)
    score += 10 if vol_ratio >= 1.5 else (5 if vol_ratio >= 1.2 else 0)
    score += 5 if breakout else 0
    score += 5 if liquidity_ok else 0

    # Entry hanya kandidat bila harga dekat VWAP atau baru breakout, tidak mengejar terlalu jauh.
    distance_vwap = abs(price / vwap_now - 1) * 100 if vwap_now and np.isfinite(vwap_now) else 99
    entry_ok = above_vwap and (distance_vwap <= 0.6 or breakout) and rsi_now <= 72
    if score >= 75 and entry_ok and liquidity_ok:
        signal = "BUY CANDIDATE"
    elif score >= 60 and daily_info["daily_trend"] and above_vwap:
        signal = "WATCH / WAIT"
    else:
        signal = "NO TRADE"

    # SL di bawah support atau 1.2 ATR; ambil level yang lebih dekat tetapi tetap di bawah harga.
    atr_stop = price - 1.2 * atr_now if atr_now > 0 else price * 0.99
    support_stop = support * 0.998
    stop = max(atr_stop, support_stop)
    if stop >= price or stop <= 0:
        stop = price - max(price * 0.008, atr_now)
    risk_per_share = price - stop
    # Hindari stop terlalu lebar untuk scalping; tandai jika >2.5% dari entry.
    stop_pct = risk_per_share / price * 100 if price > 0 else 99
    tp1, tp2 = price + risk_per_share, price + 2 * risk_per_share
    risk_cash = capital * risk_pct / 100
    cash_per_position = capital / max_positions
    lot_by_risk = int(risk_cash / (risk_per_share * 100)) if risk_per_share > 0 else 0
    lot_by_cash = int(cash_per_position / (price * 100)) if price > 0 else 0
    lots = max(0, min(lot_by_risk, lot_by_cash))
    estimated_risk = lots * 100 * risk_per_share
    estimated_value = lots * 100 * price
    warnings = []
    if not liquidity_ok:
        warnings.append("Likuiditas harian rendah")
    if stop_pct > 2.5:
        warnings.append("Jarak SL >2,5%")
    if rsi_now > 72:
        warnings.append("RSI panas")
    if not entry_ok:
        warnings.append("Belum ada trigger entry")
    if lots == 0:
        warnings.append("Lot 0: dana/risiko tidak mencukupi")
    return {
        "Kode": ticker.replace(".JK", ""),
        "Harga entry indikatif": price,
        "VWAP": vwap_now,
        "EMA9": ema9_now,
        "EMA20 intraday": ema20_now,
        "RSI intraday": rsi_now,
        "Volume x 20 candle": vol_ratio,
        "Avg nilai transaksi 20D (Rp)": avg_turnover,
        "Skor": int(score),
        "Sinyal": signal,
        "Trend harian": "Uptrend" if daily_info["daily_trend"] else "Belum uptrend",
        "MACD bullish": "Ya" if macd_bull else "Tidak",
        "Breakout 6 candle": "Ya" if breakout else "Tidak",
        "Stop loss": stop,
        "SL (%)": stop_pct,
        "TP1 (1R)": tp1,
        "TP2 (2R)": tp2,
        "Lot estimasi": lots,
        "Nilai posisi (Rp)": estimated_value,
        "Risiko estimasi (Rp)": estimated_risk,
        "Catatan": "; ".join(warnings) if warnings else "Trigger teknikal terpenuhi",
        "Interval": "intraday"
    }

with st.sidebar:
    st.header("⚙️ Pengaturan Scalping")
    capital = st.number_input("Modal trading (Rp)", min_value=1_000_000, value=DEFAULT_CAPITAL, step=500_000, format="%d")
    risk_pct = st.slider("Risiko per transaksi (%)", 0.5, 2.0, DEFAULT_RISK, 0.25)
    max_positions = st.selectbox("Maksimal posisi bersamaan", [1, 2, 3], index=1)
    min_turnover = st.number_input("Minimal rata-rata nilai transaksi 20D (Rp)", min_value=0, value=1_000_000_000, step=500_000_000, format="%d")
    shortlist_size = st.selectbox("Maksimal saham diperiksa intraday", [10, 20, 30, 40, 50], index=2)
    interval_choice = st.selectbox("Timeframe intraday", ["5 menit (utama)", "15 menit (alternatif)"], index=0)
    uploaded = st.file_uploader("Upload CSV hasil validasi Yahoo", type=["csv"])
    scan = st.button("🚀 SCAN SCALPING", type="primary", use_container_width=True)
    st.caption("Data intraday Yahoo bukan feed broker real-time. Verifikasi harga di aplikasi sekuritas.")

# Load the user's validated universe file from the same repository if present.
try:
    if uploaded is not None:
        uploaded.seek(0)
        universe = normalize_validation_file(pd.read_csv(uploaded))
        source_label = "CSV hasil validasi yang diunggah"
    elif Path("universe_validasi_scalping.csv").exists():
        universe = normalize_validation_file(pd.read_csv("universe_validasi_scalping.csv"))
        source_label = "universe_validasi_scalping.csv di repository"
    else:
        universe = pd.DataFrame(columns=["ticker", "kode", "last_data"])
        source_label = "Belum ada CSV"
except Exception as exc:
    st.error(f"Gagal membaca universe: {exc}")
    st.stop()

st.subheader("Konfigurasi dan sumber data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker tersedia", f"{len(universe):,}")
c2.metric("Modal", f"Rp{capital:,.0f}")
c3.metric("Risiko per transaksi", f"Rp{capital*risk_pct/100:,.0f}")
c4.metric("Batas risiko 2 posisi", f"Rp{capital*risk_pct/100*max_positions:,.0f}")
st.caption(f"Sumber universe: {source_label}. Ticker hanya berasal dari baris berstatus TERSEDIA pada file validasi.")
st.info(
    "Mode scalping memakai filter harian untuk likuiditas/tren, lalu memeriksa intraday hanya pada shortlist "
    "saham paling likuid. Akumulasi bandar/broker flow tidak tersedia dari data OHLCV ini."
)

if scan:
    if universe.empty:
        st.error("Universe kosong. Upload file validasi atau letakkan universe_validasi_scalping.csv di repository.")
    else:
        tickers = universe["ticker"].dropna().unique().tolist()
        progress = st.progress(0)
        status = st.empty()
        status.write(f"Mengambil data harian untuk {len(tickers)} ticker...")
        daily_map = download_daily(tuple(tickers))
        daily_rows = []
        for ticker, frame in daily_map.items():
            try:
                info = daily_quality(frame)
                info["ticker"] = ticker
                daily_rows.append(info)
            except Exception:
                continue
        daily_df = pd.DataFrame(daily_rows)
        if daily_df.empty:
            progress.empty()
            status.empty()
            st.error("Tidak ada data harian yang berhasil diambil. Coba ulangi beberapa menit lagi.")
        else:
            daily_df = daily_df[daily_df["avg_turnover"] >= min_turnover]
            daily_df = daily_df.sort_values(["daily_trend", "avg_turnover"], ascending=[False, False])
            shortlist = daily_df.head(shortlist_size)["ticker"].tolist()
            progress.progress(0.55)
            status.write(f"Mengambil data intraday {interval_choice} untuk {len(shortlist)} kandidat...")
            interval = "5m" if interval_choice.startswith("5") else "15m"
            intra_map = download_intraday(tuple(shortlist), interval)
            results = []
            daily_lookup = daily_df.set_index("ticker").to_dict("index")
            for i, ticker in enumerate(shortlist):
                frame = intra_map.get(ticker)
                if frame is not None and not frame.empty:
                    try:
                        row = analyze_intraday(ticker, frame, daily_lookup[ticker], capital, risk_pct, max_positions)
                        if row:
                            row["Interval"] = interval
                            results.append(row)
                    except Exception:
                        pass
                progress.progress(0.55 + 0.45 * ((i+1)/max(1, len(shortlist))))
            progress.empty()
            status.empty()
            if results:
                out = pd.DataFrame(results).sort_values(["Skor", "Avg nilai transaksi 20D (Rp)"], ascending=[False, False])
                st.session_state["scalp_results"] = out
                st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["scan_interval"] = interval
                st.success(f"Pemindaian selesai: {len(out)} saham mendapat data intraday dari {len(shortlist)} kandidat shortlist.")
            else:
                st.warning("Tidak ada data intraday yang berhasil diproses. Ini bukan berarti saham tidak aktif; Yahoo mungkin tidak menyediakan candle intraday untuk ticker tersebut.")

if "scalp_results" in st.session_state:
    df = st.session_state["scalp_results"].copy()
    st.subheader("🏆 Ranking kandidat scalping")
    st.caption(f"Waktu scan: {st.session_state.get('scan_time','-')} • Interval: {st.session_state.get('scan_interval','-')}")
    candidates = df[df["Sinyal"] == "BUY CANDIDATE"]
    a, b, c = st.columns(3)
    a.metric("BUY Candidate", len(candidates))
    b.metric("Watch / Wait", int((df["Sinyal"] == "WATCH / WAIT").sum()))
    c.metric("No Trade", int((df["Sinyal"] == "NO TRADE").sum()))
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga entry indikatif": st.column_config.NumberColumn(format="Rp %.2f"),
            "VWAP": st.column_config.NumberColumn(format="Rp %.2f"),
            "Stop loss": st.column_config.NumberColumn(format="Rp %.2f"),
            "TP1 (1R)": st.column_config.NumberColumn(format="Rp %.2f"),
            "TP2 (2R)": st.column_config.NumberColumn(format="Rp %.2f"),
            "Skor": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Volume x 20 candle": st.column_config.NumberColumn(format="%.2fx"),
            "Avg nilai transaksi 20D (Rp)": st.column_config.NumberColumn(format="Rp %.0f"),
            "Nilai posisi (Rp)": st.column_config.NumberColumn(format="Rp %.0f"),
            "Risiko estimasi (Rp)": st.column_config.NumberColumn(format="Rp %.0f"),
        }
    )
    st.download_button(
        "⬇️ Download hasil scan scalping CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_scan_scalping.csv",
        mime="text/csv"
    )
    st.subheader("📋 Rencana transaksi kandidat")
    selectable = df[df["Sinyal"].isin(["BUY CANDIDATE", "WATCH / WAIT"])]
    if not selectable.empty:
        choice = st.selectbox("Pilih saham", selectable["Kode"].tolist())
        row = selectable[selectable["Kode"] == choice].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Entry indikatif", f"Rp{row['Harga entry indikatif']:,.2f}")
        k2.metric("Stop loss", f"Rp{row['Stop loss']:,.2f}")
        k3.metric("TP1 (1R)", f"Rp{row['TP1 (1R)']:,.2f}")
        k4.metric("TP2 (2R)", f"Rp{row['TP2 (2R)']:,.2f}")
        st.write(
            f"**Estimasi {int(row['Lot estimasi']):,} lot** • Nilai posisi Rp{row['Nilai posisi (Rp)']:,.0f} "
            f"• Risiko terukur Rp{row['Risiko estimasi (Rp)']:,.0f}"
        )
        st.write(f"Catatan: {row['Catatan']}")
    else:
        st.info("Belum ada kandidat BUY atau WATCH. Jangan memaksakan transaksi; tunggu setup yang memenuhi filter.")

st.divider()
st.caption(
    "GOLDEN SMART MONEY AI • Scalping prototype. Risiko 2% per transaksi berarti Rp200.000 dari modal Rp10 juta. "
    "Perhitungan lot memakai 1 lot = 100 saham, dibatasi risiko dan alokasi modal per posisi; belum termasuk fee/slippage."
)
