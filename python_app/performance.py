"""
PerformanceAnalyzer
- Given a portfolio equity series and optionally a price series (benchmark),
  compute:
  - cumulative returns, CAGR, annualized volatility, Sharpe ratio,
    max drawdown, drawdown duration, total trades, win rate (if trade list provided)
"""

import pandas as pd
import numpy as np


class PerformanceAnalyzer:
    def __init__(self, equity_series: pd.Series, price_series: pd.Series = None):
        if not isinstance(equity_series, pd.Series):
            raise TypeError("equity_series must be pandas.Series")
        self.equity = equity_series.sort_index()
        self.price = price_series.sort_index() if price_series is not None else None

        # compute simple returns on equity (period returns)
        self.returns = self.equity.pct_change().fillna(0)

    def cagr(self):
        start_val = self.equity.iloc[0]
        end_val = self.equity.iloc[-1]
        days = (self.equity.index[-1] - self.equity.index[0]).days
        if days <= 0:
            return np.nan
        years = days / 365.25
        return (end_val / start_val) ** (1 / years) - 1

    def annualized_volatility(self):
        # daily returns assumed; scale by sqrt(252)
        return self.returns.std(ddof=1) * np.sqrt(252)

    def sharpe_ratio(self, risk_free_rate=0.0):
        ann_ret = self.cagr()
        ann_vol = self.annualized_volatility()
        if np.isnan(ann_ret) or ann_vol == 0:
            return np.nan
        return (ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(self):
        cum = self.equity
        high_water = cum.cummax()
        drawdown = (cum - high_water) / high_water
        max_dd = drawdown.min()
        # find drawdown period
        end_idx = drawdown.idxmin()
        # find start as previous peak
        try:
            start_idx = cum.loc[:end_idx].idxmax()
        except Exception:
            start_idx = cum.index[0]
        return float(max_dd), start_idx, end_idx

    def summary(self):
        cagr = self.cagr()
        vol = self.annualized_volatility()
        sharpe = self.sharpe_ratio()
        maxdd, dd_start, dd_end = self.max_drawdown()

        metrics = {
            "CAGR": cagr,
            "Annualized Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": maxdd,
            "Drawdown Start": dd_start,
            "Drawdown End": dd_end,
            "Start Value": float(self.equity.iloc[0]),
            "End Value": float(self.equity.iloc[-1]),
            "Total Return": float(self.equity.iloc[-1] / self.equity.iloc[0] - 1),
        }
        return metrics

    def to_dataframe(self):
        s = self.summary()
        df = pd.DataFrame.from_dict(s, orient="index", columns=["Value"])
        return df
