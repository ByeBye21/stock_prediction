"""
features.py - Feature Engineering for Direction Classification

Features for prediction:
  - Moving averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
  - OBV (On-Balance Volume)
  - Stochastic Oscillator (%K, %D)
  - Lag features (past prices)
  - Volatility (rolling std)
  - Volume pressure, gap/range, and trend-regime factors
  - Relative-strength features vs the benchmark market
  - Calendar features (day, month, quarter)
  - Log-transform features
  - External features (VIX, market index) - optional

Target:
  - Classification: 1 if the N-day forward return is positive, else 0

Data leakage prevention:
  - All features are computed from past data only
  - The target uses shift(-horizon) to represent future movement
  - Last rows become NaN and never enter training
"""

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA and EMA at multiple windows."""
    for period in [7, 21, 50]:
        df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
    for period in [12, 26]:
        df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI ranges 0-100. >70 overbought, <30 oversold."""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD trend-following momentum indicator."""
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Bollinger Bands price channel with 2 standard deviations."""
    sma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    df["BB_Upper"] = sma + 2 * std
    df["BB_Lower"] = sma - 2 * std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR measures volatility using true range."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """OBV cumulative volume indicator."""
    obv = (df["Volume"] * np.sign(df["Close"].diff())).cumsum()
    df["OBV"] = obv
    df["OBV_Zscore"] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
    return df


def add_stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D) momentum indicator."""
    low_min = df["Low"].rolling(window=period).min()
    high_max = df["High"].rolling(window=period).max()
    df["Stoch_K"] = ((df["Close"] - low_min) / (high_max - low_min)) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based features (seasonality, day-of-week effects)."""
    idx = df.index
    df["DayOfWeek"] = idx.dayofweek
    df["Month"] = idx.month
    df["Quarter"] = idx.quarter
    df["IsMonthEnd"] = idx.is_month_end.astype(int)
    df["IsQuarterEnd"] = idx.is_quarter_end.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list | None = None) -> pd.DataFrame:
    """Lag features use past close prices as autoregressive signals."""
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f"Close_Lag{lag}"] = df["Close"].shift(lag)
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling volatility and volume change features."""
    df["Volatility_10"] = df["Close"].pct_change().rolling(10).std()
    df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]
    df["Daily_Return"] = df["Close"].pct_change()
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price-to-SMA ratios, log-transforms, and multi-window returns."""
    if "SMA_21" in df.columns:
        df["Price_SMA21_Ratio"] = df["Close"] / df["SMA_21"]
    if "SMA_50" in df.columns:
        df["Price_SMA50_Ratio"] = df["Close"] / df["SMA_50"]

    df["Log_Close"] = np.log(df["Close"])
    df["Log_Volume"] = np.log(df["Volume"] + 1)

    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)
    df["Return_21d"] = df["Close"].pct_change(21)
    return df


def add_range_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap, intraday, and rolling range features."""
    prev_close = df["Close"].shift(1)
    daily_range = (df["High"] - df["Low"]) / df["Close"]
    high_low_range = (df["High"] - df["Low"]).replace(0, np.nan)

    df["Gap_Return"] = (df["Open"] / prev_close) - 1
    df["Intraday_Return"] = (df["Close"] / df["Open"]) - 1
    df["High_Low_Range"] = daily_range
    df["Range_Mean_5"] = daily_range.rolling(5).mean()
    df["Range_Mean_20"] = daily_range.rolling(20).mean()
    df["Close_Location"] = (df["Close"] - df["Low"]) / high_low_range

    if "ATR" in df.columns:
        df["ATR_Ratio"] = df["ATR"] / df["Close"]

    return df


def add_volume_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume pressure and money-flow style features."""
    volume_std_20 = df["Volume"].rolling(20).std()
    signed_volume = df["Volume"] * np.sign(df["Daily_Return"].fillna(0.0))
    high_low_range = (df["High"] - df["Low"]).replace(0, np.nan)
    money_flow_multiplier = (
        ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_range
    ).fillna(0.0)
    money_flow_volume = money_flow_multiplier * df["Volume"]

    df["Volume_Zscore_20"] = (df["Volume"] - df["Volume_SMA20"]) / volume_std_20
    df["Signed_Volume_Ratio_20"] = (
        signed_volume.rolling(20).sum() / df["Volume"].rolling(20).sum()
    )
    df["CMF_20"] = (
        money_flow_volume.rolling(20).sum() / df["Volume"].rolling(20).sum()
    )
    return df


def add_trend_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-strength and regime features from rolling structure."""
    rolling_high_20 = df["Close"].rolling(20).max()
    rolling_low_20 = df["Close"].rolling(20).min()
    rolling_high_50 = df["Close"].rolling(50).max()
    rolling_low_50 = df["Close"].rolling(50).min()

    if {"SMA_7", "SMA_21"}.issubset(df.columns):
        df["SMA_7_21_Spread"] = (df["SMA_7"] / df["SMA_21"]) - 1
    if {"SMA_21", "SMA_50"}.issubset(df.columns):
        df["SMA_21_50_Spread"] = (df["SMA_21"] / df["SMA_50"]) - 1
    if {"EMA_12", "EMA_26"}.issubset(df.columns):
        df["EMA_12_26_Spread"] = (df["EMA_12"] / df["EMA_26"]) - 1

    df["Dist_High_20"] = (df["Close"] / rolling_high_20) - 1
    df["Dist_Low_20"] = (df["Close"] / rolling_low_20) - 1
    df["Dist_High_50"] = (df["Close"] / rolling_high_50) - 1
    df["Dist_Low_50"] = (df["Close"] / rolling_low_50) - 1

    if {"Volatility_10", "Volatility_20"}.issubset(df.columns):
        df["Volatility_Regime"] = df["Volatility_10"] / df["Volatility_20"]

    return df


def add_external_features(
    df: pd.DataFrame,
    external_df: pd.DataFrame,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Merge external market features (VIX, market index) into the main frame.

    External features provide market-wide context that helps models
    distinguish between stock-specific and macro-driven movements.
    """
    if external_df is None or external_df.empty:
        return df

    aligned_external = external_df.reindex(df.index).ffill()
    if lag_days > 0:
        aligned_external = aligned_external.shift(lag_days)

    for col in aligned_external.columns:
        if col not in df.columns:
            df[col] = aligned_external[col]

    return df


def add_market_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark-relative strength and regime features when market data exists."""
    new_features = {}

    if "Market_Return" in df.columns and "Daily_Return" in df.columns:
        new_features["Relative_Return_1d"] = df["Daily_Return"] - df["Market_Return"]
        market_var_20 = df["Market_Return"].rolling(20).var().replace(0, np.nan)
        market_var_60 = df["Market_Return"].rolling(60).var().replace(0, np.nan)
        new_features["Rolling_Corr_Market_20"] = df["Daily_Return"].rolling(20).corr(df["Market_Return"])
        new_features["Rolling_Corr_Market_60"] = df["Daily_Return"].rolling(60).corr(df["Market_Return"])
        new_features["Beta_Market_20"] = df["Daily_Return"].rolling(20).cov(df["Market_Return"]) / market_var_20
        new_features["Beta_Market_60"] = df["Daily_Return"].rolling(60).cov(df["Market_Return"]) / market_var_60
    if "Market_Return_5d" in df.columns and "Return_5d" in df.columns:
        new_features["Relative_Return_5d"] = df["Return_5d"] - df["Market_Return_5d"]
    if "Market_Return_10d" in df.columns and "Return_10d" in df.columns:
        new_features["Relative_Return_10d"] = df["Return_10d"] - df["Market_Return_10d"]
    if "Market_Return_20d" in df.columns and "Return_20d" in df.columns:
        new_features["Relative_Return_20d"] = df["Return_20d"] - df["Market_Return_20d"]
    if "Market_Volatility" in df.columns and "Volatility_20" in df.columns:
        new_features["Relative_Volatility_20"] = df["Volatility_20"] / df["Market_Volatility"]
    if "Market_Price_SMA20_Ratio" in df.columns and "Price_SMA21_Ratio" in df.columns:
        new_features["Trend_Strength_vs_Market_20"] = (
            df["Price_SMA21_Ratio"] - df["Market_Price_SMA20_Ratio"]
        )
    if "Market_Price_SMA50_Ratio" in df.columns and "Price_SMA50_Ratio" in df.columns:
        new_features["Trend_Strength_vs_Market_50"] = (
            df["Price_SMA50_Ratio"] - df["Market_Price_SMA50_Ratio"]
        )
    if "Sector_Return" in df.columns and "Daily_Return" in df.columns:
        sector_var_20 = df["Sector_Return"].rolling(20).var().replace(0, np.nan)
        sector_var_60 = df["Sector_Return"].rolling(60).var().replace(0, np.nan)
        new_features["Relative_Return_vs_Sector_1d"] = df["Daily_Return"] - df["Sector_Return"]
        new_features["Rolling_Corr_Sector_20"] = df["Daily_Return"].rolling(20).corr(df["Sector_Return"])
        new_features["Rolling_Corr_Sector_60"] = df["Daily_Return"].rolling(60).corr(df["Sector_Return"])
        new_features["Beta_Sector_20"] = df["Daily_Return"].rolling(20).cov(df["Sector_Return"]) / sector_var_20
        new_features["Beta_Sector_60"] = df["Daily_Return"].rolling(60).cov(df["Sector_Return"]) / sector_var_60
    if "Sector_Return_5d" in df.columns and "Return_5d" in df.columns:
        new_features["Relative_Return_vs_Sector_5d"] = df["Return_5d"] - df["Sector_Return_5d"]
    if "Sector_Return_10d" in df.columns and "Return_10d" in df.columns:
        new_features["Relative_Return_vs_Sector_10d"] = df["Return_10d"] - df["Sector_Return_10d"]
    if "Sector_Return_20d" in df.columns and "Return_20d" in df.columns:
        new_features["Relative_Return_vs_Sector_20d"] = df["Return_20d"] - df["Sector_Return_20d"]
    if "Sector_Volatility" in df.columns and "Volatility_20" in df.columns:
        new_features["Relative_Volatility_vs_Sector_20"] = df["Volatility_20"] / df["Sector_Volatility"]
    if "Sector_Price_SMA20_Ratio" in df.columns and "Price_SMA21_Ratio" in df.columns:
        new_features["Trend_Strength_vs_Sector_20"] = (
            df["Price_SMA21_Ratio"] - df["Sector_Price_SMA20_Ratio"]
        )
    if "Sector_Price_SMA50_Ratio" in df.columns and "Price_SMA50_Ratio" in df.columns:
        new_features["Trend_Strength_vs_Sector_50"] = (
            df["Price_SMA50_Ratio"] - df["Sector_Price_SMA50_Ratio"]
        )
    if "Market_Return_20d" in df.columns and "Sector_Return_20d" in df.columns:
        new_features["Sector_vs_Market_20d"] = df["Sector_Return_20d"] - df["Market_Return_20d"]

    return df.assign(**new_features) if new_features else df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create the classification target for the selected horizon."""
    forward_return = df["Close"].pct_change(horizon).shift(-horizon)
    df["Target"] = np.where(
        forward_return.isna(),
        np.nan,
        (forward_return > 0).astype(float),
    )
    return df


def _build_feature_frame(
    df: pd.DataFrame,
    external_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build the predictor frame without adding any future-dependent targets."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_stochastic(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_volatility(df)
    df = add_extra_features(df)
    df = add_range_gap_features(df)
    df = add_volume_pressure_features(df)
    df = add_trend_regime_features(df)

    if external_df is not None:
        df = add_external_features(df, external_df)
        df = add_market_relative_features(df)

    return df.replace([np.inf, -np.inf], np.nan)


def build_feature_matrix(
    df: pd.DataFrame,
    external_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build an inference-ready feature matrix without dropping the newest row
    just because its future target is unavailable.
    """
    df = _build_feature_frame(df, external_df=external_df)
    return df.dropna().copy()


def build_features(
    df: pd.DataFrame,
    horizon: int = 1,
    external_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps and clean NaN rows.

    Args:
        df: Raw OHLCV DataFrame
        horizon: Prediction horizon in days (1, 5, 10, 20)
        external_df: Optional external data (VIX, market index)

    Returns:
        Cleaned DataFrame with all features and target.
    """
    df = _build_feature_frame(df, external_df=external_df)
    df = add_target(df, horizon=horizon)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (everything except raw OHLCV and targets)."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "Target", "OBV",
    }
    return [col for col in df.columns if col not in exclude]
