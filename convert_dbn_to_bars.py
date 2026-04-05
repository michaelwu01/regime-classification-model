#!/usr/bin/env python3
"""
Convert DBN trade files to 5-minute bars for the regime model.
This script reads .dbn.zst files and converts them to aggregated bars.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import databento as db
from datetime import datetime, timezone
import glob

# Configuration
RAW_DATA_DIR = "raw_data"
OUTPUT_DIR = "expanded_data"
BAR_INTERVAL_MINUTES = 5

# GC contract specifications
GC_TICK_SIZE = 0.1  # $0.10 per troy ounce
GC_MULTIPLIER = 100  # 100 troy ounces per contract

def convert_dbn_to_dataframe(dbn_file_path):
    """
    Read a DBN file and convert to pandas DataFrame.
    """
    print(f"Reading {os.path.basename(dbn_file_path)}...")

    # Read DBN file using databento
    store = db.DBNStore.from_file(dbn_file_path)

    # Convert to DataFrame
    df = store.to_df()

    if df.empty:
        print(f"  WARNING: Empty file {dbn_file_path}")
        return None

    # Ensure timestamp is datetime
    if 'ts_event' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
    elif df.index.name == 'ts_event':
        df['timestamp'] = pd.to_datetime(df.index, unit='ns', utc=True)
        df = df.reset_index()

    print(f"  Loaded {len(df):,} trades")
    return df

def aggregate_to_bars(trades_df, interval_minutes=5):
    """
    Aggregate trades into OHLCV bars.
    """
    if trades_df is None or trades_df.empty:
        return None

    # Set timestamp as index for resampling
    trades_df = trades_df.set_index('timestamp')

    # Resample to bars
    bars = trades_df.resample(f'{interval_minutes}min').agg({
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum'
    })

    # Flatten column names
    bars.columns = ['open', 'high', 'low', 'close', 'volume']

    # Remove bars with no trades
    bars = bars[bars['volume'] > 0].copy()

    # Reset index to get timestamp as column
    bars = bars.reset_index()

    print(f"  Created {len(bars):,} {interval_minutes}-minute bars")
    return bars

def process_all_files(start_year=2020, end_year=2022):
    """
    Process all DBN files for the specified year range.
    """
    print(f"\n{'='*80}")
    print(f"CONVERTING DBN FILES TO {BAR_INTERVAL_MINUTES}-MINUTE BARS")
    print(f"Years: {start_year}-{end_year}")
    print(f"{'='*80}\n")

    # Find all GC trade files
    pattern = f"{RAW_DATA_DIR}/raw_trades/GLBX.MDP3__GC.n.0__continuous__trades__*.dbn.zst"
    all_files = sorted(glob.glob(pattern))

    # Filter by year range
    files_to_process = []
    for f in all_files:
        filename = os.path.basename(f)
        # Extract year from filename (format: ...trades__20200101T000000Z...)
        # Split by __ and find the part starting with 4 digits
        parts = filename.split('__')
        for part in parts:
            if part and part[0].isdigit() and len(part) >= 4:
                year_str = part[:4]
                try:
                    year = int(year_str)
                    if start_year <= year <= end_year:
                        files_to_process.append(f)
                    break
                except ValueError:
                    continue

    print(f"Found {len(files_to_process)} files to process")
    print(f"Date range: {os.path.basename(files_to_process[0])} to {os.path.basename(files_to_process[-1])}\n")

    # Process each file and collect bars
    all_bars = []

    for i, file_path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] Processing {os.path.basename(file_path)}")

        try:
            # Convert DBN to DataFrame
            trades_df = convert_dbn_to_dataframe(file_path)

            if trades_df is not None:
                # Aggregate to bars
                bars_df = aggregate_to_bars(trades_df, BAR_INTERVAL_MINUTES)

                if bars_df is not None and not bars_df.empty:
                    all_bars.append(bars_df)

        except Exception as e:
            print(f"  ERROR processing {file_path}: {e}")
            continue

    if not all_bars:
        print("\nERROR: No bars created!")
        return None

    # Concatenate all bars
    print(f"\n{'-'*80}")
    print("Combining all bars...")
    combined_bars = pd.concat(all_bars, ignore_index=True)
    combined_bars = combined_bars.sort_values('timestamp').reset_index(drop=True)

    # Remove timezone from timestamp for compatibility
    combined_bars['timestamp'] = combined_bars['timestamp'].dt.tz_localize(None)

    print(f"Total bars created: {len(combined_bars):,}")
    print(f"Date range: {combined_bars['timestamp'].min()} to {combined_bars['timestamp'].max()}")
    print(f"\nBar statistics:")
    print(f"  Open:   ${combined_bars['open'].min():.2f} - ${combined_bars['open'].max():.2f}")
    print(f"  High:   ${combined_bars['high'].min():.2f} - ${combined_bars['high'].max():.2f}")
    print(f"  Low:    ${combined_bars['low'].min():.2f} - ${combined_bars['low'].max():.2f}")
    print(f"  Close:  ${combined_bars['close'].min():.2f} - ${combined_bars['close'].max():.2f}")
    print(f"  Volume: {int(combined_bars['volume'].min())} - {int(combined_bars['volume'].max())}")

    # Save to CSV
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/GC_bars_{BAR_INTERVAL_MINUTES}min_{start_year}_{end_year}.csv"
    combined_bars.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    return combined_bars

if __name__ == "__main__":
    # Process 2020-2022 data
    bars_df = process_all_files(start_year=2020, end_year=2022)

    if bars_df is not None:
        print(f"\n{'='*80}")
        print("CONVERSION COMPLETE!")
        print(f"{'='*80}")
    else:
        print("\nConversion failed!")
        sys.exit(1)
