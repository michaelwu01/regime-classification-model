from __future__ import annotations

import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def load_bars(
    catalog_path: str,
    bar_type_str: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Load bars from ParquetDataCatalog and return as a DataFrame."""
    catalog = ParquetDataCatalog(catalog_path)

    start_dt = pd.Timestamp(start, tz="UTC") if start else None
    end_dt = pd.Timestamp(end, tz="UTC") if end else None

    bars_list = catalog.bars(bar_types=[bar_type_str], start=start_dt, end=end_dt)
    df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(b.ts_event, unit="ns", tz="UTC"),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
        }
        for b in bars_list
    ])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df):,} bars ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    return df


def load_ticks(
    catalog_path: str,
    instrument_id: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Load trade ticks from ParquetDataCatalog.

    Returns DataFrame with columns: timestamp, price, volume.
    """
    catalog = ParquetDataCatalog(catalog_path)

    start_dt = pd.Timestamp(start, tz="UTC") if start else None
    end_dt = pd.Timestamp(end, tz="UTC") if end else None

    ticks_list = catalog.trade_ticks(
        instrument_ids=[instrument_id],
        start=start_dt,
        end=end_dt,
    )
    if not ticks_list:
        raise ValueError(
            f"No trade ticks found for instrument='{instrument_id}' "
        )

    df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp(t.ts_event, unit="ns", tz="UTC"),
            "price": float(t.price),
            "volume": float(t.size),
        }
        for t in ticks_list
    ])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df):,} trade ticks ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    return df
