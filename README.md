# Regime Classification Model v2

Machine learning models for classifying market regimes (TREND, TRANSITION, CHOP) in gold futures (GC) using technical indicators.

## Overview

This project trains classification models to identify three market regime types:
- **TREND**: Strong directional movement
- **TRANSITION**: Moderate directional movement
- **CHOP**: Sideways/choppy market with no clear direction

## Key Results

### Expanded Dataset (2020-2022)
- Dataset: 212,389 5-minute bars over 3 years
- Best Model: MLP (64 hidden units)
- **TREND Recall: 46.67%** (solved the 0% recall problem)
- **TRANSITION Recall: 94.17%**
- Macro F1: 0.4819

See [EXPANDED_DATASET_RESULTS.md](EXPANDED_DATASET_RESULTS.md) for detailed performance metrics.

## Project Structure

```
regime_model_v2/
├── regime_sandbox/          # Core library
│   ├── preprocess/         # Feature engineering
│   ├── label/              # Regime labeling logic
│   └── train/              # Model training
├── run_training.py         # Original baseline training script
├── compare_models_weighted.py  # Model comparison script
├── convert_dbn_to_bars.py  # DBN file conversion to OHLCV bars
├── train_expanded_data.py  # Training on expanded 2020-2022 dataset
├── regime_model_pipeline.ipynb  # Original Jupyter notebook
└── EXPANDED_DATASET_RESULTS.md  # Performance report

```

## Features

The model uses 6 technical indicator features:
1. **Delta DI**: Difference between ADX positive and negative directional indicators
2. **Slope**: Moving average slope (trend strength)
3. **Efficiency Ratio**: Price movement efficiency
4. **ATR Z-score**: Normalized volatility
5. **Bollinger Band Width**: Volatility measure
6. **Imbalance**: Order flow imbalance (tick data)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train on Expanded Dataset (Recommended)

```bash
python train_expanded_data.py
```

This trains both Logistic Regression (weighted) and MLP models on 3 years of GC data.

### Convert DBN Files to Bars

If you have additional Databento DBN files:

```bash
python convert_dbn_to_bars.py
```

Edit the script to specify:
- Input directory with .dbn.zst files
- Output CSV path
- Date range (start_year, end_year)

### Run Baseline Training

```bash
python run_training.py
```

### Compare Multiple Models

```bash
python compare_models_weighted.py
```

## Model Files

After training, models are saved to:
- `output/expanded_train_mlp/model.pkl` - Best MLP model
- `output/expanded_train_mlp/scaler.pkl` - Feature scaler
- `output/expanded_train_logistic_weighted/` - Logistic regression model

## Performance History

| Model | Dataset | TREND Recall | Macro F1 |
|-------|---------|--------------|----------|
| Logistic (baseline) | 1.5 months (13K bars) | 0.00% | 0.39 |
| Logistic (weighted) | 1.5 months (13K bars) | 1.13% | 0.41 |
| MLP | 1.5 months (13K bars) | 1.32% | 0.43 |
| **MLP (best)** | **3 years (212K bars)** | **46.67%** | **0.48** |

## Data Sources

- **Primary**: Gold futures (GC) 5-minute bars from Databento
- **Period**: 2020-2022 (36 months)
- **Total trades**: 75+ million
- **Total bars**: 212,389

## Known Issues

1. **CHOP class under-represented** in 2020-2022 data (0.5% vs 50.7% in original)
   - Models cannot reliably identify CHOP regimes
   - Consider adjusting labeling thresholds or adding more data

## Future Improvements

1. Fix CHOP detection by adjusting thresholds in `regime_sandbox/label/config.py`
2. Add NQ/ES data for more diverse market conditions
3. Implement ensemble models
4. Add HMM post-processing for temporal smoothing
5. Expand to 2018-2024 for more historical coverage

## License

MIT

## Authors

Created for gold futures regime classification research.
