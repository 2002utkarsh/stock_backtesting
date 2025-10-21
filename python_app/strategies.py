# strategies.py
"""
Advanced strategy implementations for backtesting.

Each strategy function accepts a pandas DataFrame `df` (must contain a 'close' column;
other OHLC/volume columns are used when present) and returns a pd.Series of integer
signals aligned to df.index with values:
    1  -> Buy (enter long)
   -1  -> Sell (exit long)
    0  -> Hold / no action

Strategies:
 - rsi_macd_hybrid(...)
 - bollinger_mean_reversion(...)
 - ml_short_term(...)

These are designed to plug directly into the BacktestManager pipeline.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# Strategy registry — maps names in config.yaml to functions
# ------------------------------------------------


# -------------------------
# Indicator helpers
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Return (macd_line, signal_line)."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Classic RSI implementation returning values in [0,100]."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # avoid division by zero using small epsilon
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50)


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (middle_band, upper_band, lower_band)
    """
    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0)
    upper = middle + n_std * std
    lower = middle - n_std * std
    return middle, upper, lower


# -------------------------
# Strategy 1: RSI + MACD Hybrid
# -------------------------
def rsi_macd_hybrid(df: pd.DataFrame,
                    rsi_period: int = 14,
                    rsi_overbought: int = 70,
                    rsi_oversold: int = 30,
                    macd_fast: int = 12,
                    macd_slow: int = 26,
                    macd_signal: int = 9) -> pd.Series:
    """
    Combines RSI and MACD:
      - Buy when RSI < oversold AND MACD crosses above its signal line.
      - Sell when RSI > overbought AND MACD crosses below its signal line.
    Returns entry/exit signals (position diffs).
    """
    close = df["close"].astype(float)
    macd_line, signal_line = macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    rsi_vals = rsi(close, period=rsi_period)

    # Create boolean conditions
    macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    buy_cond = (rsi_vals < rsi_oversold) & macd_cross_up
    sell_cond = (rsi_vals > rsi_overbought) & macd_cross_down

    # Build a position series: +1 when buy, 0 when hold, -1 when sell — using diff of "in/out"
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[buy_cond.fillna(False)] = 1
    raw[sell_cond.fillna(False)] = -1

    # Convert raw discrete signals into entry/exit signals via position tracking
    # We prefer to use the discrete raw events as immediate trade signals (1 / -1).
    trade_signals = raw.astype(int).reindex(df.index).fillna(0).astype(int)
    return trade_signals


# -------------------------
# Strategy 2: Bollinger Bands Mean Reversion
# -------------------------
def bollinger_mean_reversion(df: pd.DataFrame,
                             window: int = 20,
                             n_std: float = 2.0,
                             exit_band_pct: float = 0.05) -> pd.Series:
    """
    Mean reversion on Bollinger Bands:
      - Buy when price crosses below lower band.
      - Sell when price crosses above upper band.
      - Exit long when price returns to within `exit_band_pct` of the middle band.
    """
    close = df["close"].astype(float)
    middle, upper, lower = bollinger_bands(close, window=window, n_std=n_std)

    cross_below = (close < lower) & (close.shift(1) >= lower.shift(1))
    cross_above = (close > upper) & (close.shift(1) <= upper.shift(1))

    # Exit condition when price is close to middle band (within a percentage)
    within_exit = (np.abs(close - middle) / (middle + 1e-9)) < exit_band_pct

    # We'll produce immediate trade signals on events:
    signals = pd.Series(0, index=df.index, dtype=int)
    signals[cross_below.fillna(False)] = 1   # buy
    signals[cross_above.fillna(False)] = -1  # sell

    # Optionally: if within_exit after buy, emit a sell to close — but to avoid double-selling,
    # we use position tracking externally (C++ engine does 1-share per signal).
    # Keep outputs as immediate signals only.
    return signals


# -------------------------
# Strategy 3: ML-based short-term predictor (RandomForest)
# -------------------------
def ml_short_term(df: pd.DataFrame,
                  lookahead: int = 5,
                  min_train_size: int = 200,
                  prob_threshold: float = 0.55,
                  random_state: int = 42,
                  n_estimators: int = 200) -> pd.Series:
    """
    Train a RandomForest to predict whether price will be higher after `lookahead` periods.
    Produces entry signals when predicted class changes from 0->1 and exit when 1->0.

    Implementation details:
      - Features: recent returns, simple moving averages, volatility, MACD, RSI
      - Target: 1 if close.shift(-lookahead) > close else 0
      - If not enough training rows, returns zeros.
    """
    df = df.copy()
    close = df["close"].astype(float)

    # Feature engineering
    df["ret_1"] = close.pct_change(1).fillna(0)
    df["ret_3"] = close.pct_change(3).fillna(0)
    df["ret_5"] = close.pct_change(5).fillna(0)
    df["sma_5"] = close.rolling(window=5, min_periods=1).mean()
    df["sma_10"] = close.rolling(window=10, min_periods=1).mean()
    df["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    df["vol_10"] = df["ret_1"].rolling(window=10, min_periods=1).std().fillna(0)
    macd_line, macd_sig = macd(close)
    df["macd"] = macd_line
    df["macd_sig"] = macd_sig
    df["rsi_14"] = rsi(close, period=14)

    # Target (future direction)
    df["target"] = (close.shift(-lookahead) > close).astype(int)

    # Drop last lookahead rows where target is NaN
    df.dropna(inplace=True)
    if len(df) < min_train_size:
        # Not enough data to train — return zeros
        return pd.Series(0, index=df.index).reindex(close.index, fill_value=0).astype(int)

    feature_cols = ["ret_1", "ret_3", "ret_5", "sma_5", "sma_10", "sma_20", "vol_10", "macd", "macd_sig", "rsi_14"]
    X = df[feature_cols].fillna(0)
    y = df["target"]

    # Train / test split maintaining chronological order: train on first 80%
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict probabilities for the entire dataset X
    probs = pd.Series(model.predict_proba(X)[:, 1], index=X.index)

    # Construct signals: entry when prob crosses above threshold; exit when crosses below threshold
    above = probs > prob_threshold
    cross_up = (above & (~above.shift(1).fillna(False)))
    cross_down = ((~above) & (above.shift(1).fillna(False)))

    signals = pd.Series(0, index=close.index, dtype=int)
    # Note: df.index is truncated (dropped tail rows), align indices carefully
    # Fill signals at appropriate timestamps where we have prob info
    signals.loc[probs.index[cross_up]] = 1
    signals.loc[probs.index[cross_down]] = -1

    # Reindex to full original index (fill missing zeros)
    return signals.reindex(close.index, fill_value=0).astype(int)


# -------------------------
# Backwards-compatible default wrappers
# -------------------------
def moving_average_crossover(df: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """
    Existing simple MA crossover strategy kept for compatibility.
    Returns position diffs as signals (-1,0,1).
    """
    signals = pd.DataFrame(index=df.index)
    signals["short_mavg"] = df["close"].rolling(window=short_window, min_periods=1).mean()
    signals["long_mavg"] = df["close"].rolling(window=long_window, min_periods=1).mean()
    signals["raw"] = np.where(signals["short_mavg"] > signals["long_mavg"], 1, 0)
    signals["positions"] = signals["raw"].diff().fillna(0)
    trade_signals = signals["positions"].astype(int)
    return trade_signals.reindex(df.index).fillna(0).astype(int)


def ml_strategy(df: pd.DataFrame):
    """
    Backwards-compatible lightweight logistic regression strategy.
    Kept as-is for reproducibility; prefer ml_short_term for better results.
    """
    df = df.copy()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(20).std()
    df.dropna(inplace=True)
    if df.empty:
        return pd.Series(0, index=df.index)
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
    features = ["sma50", "sma200", "volatility"]
    X = df[features].fillna(0)
    y = df["target"]
    if len(X) < 10:
        return pd.Series(0, index=df.index)
    split_idx = int(len(X) * 0.8)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    preds = pd.Series(model.predict(X), index=X.index)
    trade_signals = preds.diff().fillna(0).astype(int)
    trade_signals = trade_signals.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    trade_signals.index = df.index
    return trade_signals.reindex(df.index).fillna(0).astype(int)

STRATEGIES = {
    "moving_average_crossover": moving_average_crossover,
    "rsi_macd_hybrid": rsi_macd_hybrid,
    "bollinger_mean_reversion": bollinger_mean_reversion,
    "ml_short_term": ml_short_term,
    "ml_strategy": ml_strategy,
}