import pandas as pd
from pathlib import Path
from typing import Union


def load_ohlcv_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file into a standardized DataFrame.

    The CSV is expected to contain at least the following columns:
    - open_time: timestamp or datetime string for candle open
    - open
    - high
    - low
    - close
    - volume

    Optional columns such as close_time will be preserved if present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Historical data file not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(missing)}. "
            f"Expected at least: {', '.join(required_cols)}"
        )

    # Parse timestamps
    df["open_time"] = pd.to_datetime(df["open_time"])
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"])

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    return df.sort_values("open_time").reset_index(drop=True)
