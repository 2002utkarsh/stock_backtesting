"""
BacktestManager
- Loads the C++ backtester shared library (.so/.dll)
- Converts pandas DataFrame -> C struct array
- Calls perform_backtest and returns pandas.Series of portfolio values
"""

import ctypes
import os
import sys
from typing import Optional, Sequence
import pandas as pd
import numpy as np
from pathlib import Path

# Define StockTick structure to match C++ header exactly
class StockTick(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_longlong),
        ("open", ctypes.c_double),
        ("high", ctypes.c_double),
        ("low", ctypes.c_double),
        ("close", ctypes.c_double),
        ("volume", ctypes.c_int),
    ]


class BacktestManager:
    def __init__(self, lib_path: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.lib = None
        if lib_path:
            self.lib = self._try_load(lib_path)
        else:
            self.lib = self._auto_find_and_load()

        if self.lib is None:
            raise RuntimeError("Could not load backtester shared library. "
                               "Provide path or place lib in project folders.")

        # set argtypes for safety
        self.lib.perform_backtest.argtypes = [
            ctypes.POINTER(StockTick),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.perform_backtest.restype = None

    # -----------------------
    # Library loading helpers
    # -----------------------
    def _try_load(self, path: str):
        try:
            if self.verbose:
                print(f"[BacktestManager] Trying to load: {path}")
            return ctypes.CDLL(path)
        except OSError:
            if self.verbose:
                print(f"[BacktestManager] Failed to load: {path}")
            return None

    def _auto_find_and_load(self):
        """
        Search likely places in project for backtester shared library (.so/.dll)
        Walks up to 3 levels and checks sibling folders like cpp_engine, build, /mnt/data.
        """
        cand_names = ["backtester.dll", "libbacktester.so", "libbacktester.dylib"]
        search_roots = [
            Path.cwd(),
            Path.cwd().parent,
            Path.cwd().parent.parent,
            Path("/mnt/data")
        ]

        # search within project tree for candidate filenames
        for root in search_roots:
            if not root.exists():
                continue
            for cand in cand_names:
                p = root / cand
                if p.exists():
                    lib = self._try_load(str(p))
                    if lib:
                        if self.verbose:
                            print(f"[BacktestManager] Loaded library from {p}")
                        return lib

        # Walk directory structure (limited depth) for candidate names
        project_root = Path.cwd().parent if (Path.cwd() / "python_app").exists() else Path.cwd()
        max_depth = 3
        for root, dirs, files in os.walk(project_root):
            depth = len(Path(root).relative_to(project_root).parts)
            if depth > max_depth:
                continue
            for cand in cand_names:
                if cand in files:
                    p = Path(root) / cand
                    lib = self._try_load(str(p))
                    if lib:
                        if self.verbose:
                            print(f"[BacktestManager] Loaded library from {p}")
                        return lib
        if self.verbose:
            print("[BacktestManager] No library found in auto-search.")
        return None

    # -----------------------
    # Conversion helpers
    # -----------------------
    def _ts_to_int(self, ts) -> int:
        # Accept pandas Timestamp or python datetime
        try:
            # pd.Timestamp has timestamp() method; handle tz-aware
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                # convert to UTC
                return int(pd.to_datetime(ts).tz_convert("UTC").timestamp())
            return int(pd.to_datetime(ts).timestamp())
        except Exception:
            # fallback to ns -> seconds
            try:
                return int(pd.to_datetime(ts).value // 1_000_000_000)
            except Exception:
                raise ValueError(f"Unable to convert timestamp {ts}")

    def _validate_df(self, df: pd.DataFrame):
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    # -----------------------
    # Main entrypoint
    # -----------------------
    def run_backtest(self, df: pd.DataFrame, signals: pd.Series, initial_cash: float = 10000.0) -> pd.Series:
        """
        df: pandas DataFrame indexed by tz-aware or naive Timestamp, containing open, high, low, close, volume
        signals: pandas Series aligned with df.index with values {-1, 0, 1}
        returns: pandas Series of portfolio value across same index
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        # normalize column names to lower-case
        df = df.copy()
        df.columns = df.columns.str.lower()
        self._validate_df(df)

        # Align signals to df index
        signals = signals.reindex(df.index).fillna(0).astype(int)

        num_ticks = len(df)
        if num_ticks == 0:
            raise ValueError("Empty DataFrame provided to backtest.")

        # Build C array of StockTick
        ticks_array = (StockTick * num_ticks)()
        for i, (idx, row) in enumerate(df.iterrows()):
            ts_int = self._ts_to_int(idx)
            ticks_array[i] = StockTick(
                ts_int,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            )

        # Prepare signal array
        signals_array = (ctypes.c_int * num_ticks)(*signals.astype(int).values.tolist())

        # Prepare output buffer
        output_array = (ctypes.c_double * num_ticks)()

        # call the C++ engine
        self.lib.perform_backtest(ticks_array, ctypes.c_int(num_ticks), signals_array, output_array)

        # convert back to pandas Series
        py_vals = [float(output_array[i]) for i in range(num_ticks)]
        result_series = pd.Series(py_vals, index=df.index, name="portfolio_value")

        return result_series
