"""
data_loader.py — Download and load stock data + external market data

Usage:
    df = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
    ext = download_external_data("2020-01-01", "2024-12-31", ticker="AAPL")

Features:
    - Stock OHLCV data via yfinance
    - External data: VIX, market benchmark, and sector ETF proxy
    - Streamlit cache support
    - Error handling
"""

import pandas as pd
import yfinance as yf
import streamlit as st


# ─── Ticker-to-market-index mapping ───
# Used to automatically select the right market benchmark
MARKET_INDEX_MAP = {
    ".IS": "XU100.IS",     # Borsa Istanbul
    ".DE": "^GDAXI",       # Germany (DAX)
    ".L": "^FTSE",          # London
    ".PA": "^FCHI",        # France (CAC 40)
    ".AS": "^AEX",         # Netherlands (AEX)
    ".MI": "FTSEMIB.MI",   # Italy (FTSE MIB)
    ".SW": "^SSMI",        # Switzerland (SMI)
    ".ST": "^OMX",         # Sweden (OMX Stockholm 30)
    ".HK": "^HSI",          # Hong Kong
    ".TO": "^GSPTSE",       # Toronto
}
DEFAULT_MARKET_INDEX = "^GSPC"  # S&P 500 as default
SECTOR_ETF_MAP = {
    "technology": "XLK",
    "communication services": "XLC",
    "consumer cyclical": "XLY",
    "consumer discretionary": "XLY",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "healthcare": "XLV",
    "financial services": "XLF",
    "financial": "XLF",
    "industrials": "XLI",
    "industrial goods": "XLI",
    "energy": "XLE",
    "basic materials": "XLB",
    "materials": "XLB",
    "utilities": "XLU",
    "real estate": "XLRE",
}


def _detect_market_index(ticker: str) -> str:
    """Auto-detect the appropriate market index based on ticker suffix."""
    for suffix, index in MARKET_INDEX_MAP.items():
        if ticker.upper().endswith(suffix):
            return index
    return DEFAULT_MARKET_INDEX


def _normalize_sector(sector: str) -> str:
    """Normalize sector labels for ETF mapping."""
    return str(sector or "").strip().lower()


def _detect_sector_etf(sector: str = "") -> str | None:
    """Map a stock sector to a liquid ETF proxy when possible."""
    normalized_sector = _normalize_sector(sector)
    if not normalized_sector:
        return None
    return SECTOR_ETF_MAP.get(normalized_sector)


def _make_inclusive_end(end) -> str:
    """Convert a user-selected end date into the inclusive form yfinance expects."""
    inclusive_end = pd.Timestamp(end) + pd.Timedelta(days=1)
    return inclusive_end.strftime("%Y-%m-%d")


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download stock data via yfinance.
    Streamlit cache ensures the same parameters don't trigger a re-download.

    Args:
        ticker: Stock symbol (e.g. "AAPL", "MSFT", "THYAO.IS")
        start:  Start date (YYYY-MM-DD)
        end:    Inclusive end date (YYYY-MM-DD)

    Returns:
        DataFrame: Date-indexed OHLCV data

    Raises:
        ValueError: If no data is found
    """
    try:
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError("Invalid date range: start date must be on or before end date.")

        df = yf.download(
            ticker,
            start=start,
            end=_make_inclusive_end(end),
            progress=False,
        )

        # yfinance sometimes returns MultiIndex (when querying multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError(
                f"No data found for '{ticker}'.\n"
                "Please check the ticker symbol."
            )

        # Standardize column names
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        return df

    except Exception as e:
        raise ValueError(f"Data download error: {str(e)}")


@st.cache_data(ttl=3600, show_spinner=False)
def download_external_data(
    start: str,
    end: str,
    ticker: str = "AAPL",
    sector: str = "",
    include_vix: bool = True,
    include_market: bool = True,
    include_sector: bool = True,
) -> pd.DataFrame:
    """
    Download selected external factors: VIX, market benchmark, and sector ETF proxy.

    VIX (CBOE Volatility Index):
        - Measures market fear / expected volatility
        - High VIX → high uncertainty → harder to predict
        - Provides global market context

    Market Index (e.g. S&P 500):
        - Market-level trend context
        - Helps model separate stock-specific vs market-wide movements

    Args:
        start: Start date (YYYY-MM-DD)
        end:   Inclusive end date (YYYY-MM-DD)
        ticker: Stock ticker (used to auto-detect market index)
        sector: Stock sector (used to detect a sector ETF proxy)
        include_vix: Whether to include VIX features
        include_market: Whether to include market benchmark features
        include_sector: Whether to include sector ETF proxy features

    Returns:
        DataFrame with the selected external feature groups when available
    """
    result = pd.DataFrame()
    download_end = _make_inclusive_end(end)

    # 1. VIX — Fear Index
    if include_vix:
        try:
            vix = yf.download("^VIX", start=start, end=download_end, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            if not vix.empty:
                result["VIX_Close"] = vix["Close"]
                result["VIX_Change"] = vix["Close"].pct_change()
                result["VIX_Change_5d"] = vix["Close"].pct_change(5)
                result["VIX_Regime_20"] = vix["Close"] / vix["Close"].rolling(20).mean()
                result["VIX_Close_Zscore_20"] = (
                    (vix["Close"] - vix["Close"].rolling(20).mean()) /
                    vix["Close"].rolling(20).std()
                )
                result["VIX_Percentile"] = (
                    vix["Close"].rolling(60).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                        raw=False,
                    )
                )
        except Exception:
            pass  # VIX not available — continue without it

    # 2. Market Index
    if include_market:
        try:
            market_ticker = _detect_market_index(ticker)
            mkt = yf.download(market_ticker, start=start, end=download_end, progress=False)
            if isinstance(mkt.columns, pd.MultiIndex):
                mkt.columns = mkt.columns.get_level_values(0)
            if not mkt.empty:
                result["Market_Return"] = mkt["Close"].pct_change()
                result["Market_Return_5d"] = mkt["Close"].pct_change(5)
                result["Market_Return_10d"] = mkt["Close"].pct_change(10)
                result["Market_Return_20d"] = mkt["Close"].pct_change(20)
                result["Market_Volatility"] = mkt["Close"].pct_change().rolling(20).std()
                result["Market_Momentum"] = mkt["Close"].pct_change(20)
                result["Market_Price_SMA20_Ratio"] = (
                    mkt["Close"] / mkt["Close"].rolling(20).mean()
                )
                result["Market_Price_SMA50_Ratio"] = (
                    mkt["Close"] / mkt["Close"].rolling(50).mean()
                )
        except Exception:
            pass  # Market index not available — continue without it

    # 3. Sector ETF Proxy
    if include_sector:
        try:
            sector_ticker = _detect_sector_etf(sector)
            if sector_ticker:
                sector_df = yf.download(sector_ticker, start=start, end=download_end, progress=False)
                if isinstance(sector_df.columns, pd.MultiIndex):
                    sector_df.columns = sector_df.columns.get_level_values(0)
                if not sector_df.empty:
                    result["Sector_Return"] = sector_df["Close"].pct_change()
                    result["Sector_Return_5d"] = sector_df["Close"].pct_change(5)
                    result["Sector_Return_10d"] = sector_df["Close"].pct_change(10)
                    result["Sector_Return_20d"] = sector_df["Close"].pct_change(20)
                    result["Sector_Volatility"] = sector_df["Close"].pct_change().rolling(20).std()
                    result["Sector_Momentum"] = sector_df["Close"].pct_change(20)
                    result["Sector_Price_SMA20_Ratio"] = (
                        sector_df["Close"] / sector_df["Close"].rolling(20).mean()
                    )
                    result["Sector_Price_SMA50_Ratio"] = (
                        sector_df["Close"] / sector_df["Close"].rolling(50).mean()
                    )
        except Exception:
            pass  # Sector ETF not available — continue without it

    return result.sort_index()


def get_stock_info(ticker: str) -> dict:
    """Return basic information about the stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "currency": info.get("currency", ""),
            "exchange": info.get("exchange", ""),
            "sector": info.get("sector", ""),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {
            "name": ticker,
            "currency": "",
            "exchange": "",
            "sector": "",
            "market_cap": 0,
        }
