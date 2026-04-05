# Data Split Analysis - Train/Validation/Test

## 🎯 Your Important Questions

**Q1: How do you split the train, validate, test dataset?**
**Q2: What is the prediction accuracy?**
**Q3: Does the prediction is on the dataset that never seen?**

Let me answer these questions in detail:

---

## 📊 Data Split Method

### Split Strategy: **Time-Based Sequential Split** (60/20/20)

```python
# From train_optimized_model.py, lines 87-94

n = len(df_merged)  # Total: 212,290 samples
train_end = int(0.6 * n)  # 60% mark
val_end = int(0.8 * n)    # 80% mark

train_df = df_merged.iloc[:train_end]        # First 60%
val_df = df_merged.iloc[train_end:val_end]   # Next 20%
test_df = df_merged.iloc[val_end:]            # Last 20%
```

### Actual Split Sizes

| Dataset | Samples | Percentage | Date Range (Approx) |
|---------|---------|------------|---------------------|
| **Training** | 127,374 | 60% | 2020-01-01 to 2021-03-15 |
| **Validation** | 42,458 | 20% | 2021-03-15 to 2022-01-01 |
| **Test** | 42,458 | 20% | 2022-01-01 to 2022-12-30 |
| **Total** | 212,290 | 100% | 2020-01-01 to 2022-12-30 |

---

## ⚠️ **CRITICAL: Is Test Data "Never Seen"?**

### The Answer: **YES and NO** - Let me explain the nuance

### ✅ YES - Model Never Saw Test Features

**The model NEVER saw the test set during training:**
- Training: Model learns from first 60% (2020-2021 data)
- Validation: Model tunes hyperparameters on next 20% (2021 data)
- **Test: Model evaluates on FINAL 20% (2022 data) - NEVER used in training**

This means:
- ✅ The 93% macro F1 is on **completely unseen data** (2022)
- ✅ Model has NO knowledge of test feature values
- ✅ This is proper machine learning evaluation

### ⚠️ BUT - There's a Potential Issue: **Data Leakage**

**The labels are created using future information!** This is subtle but important:

```python
# Labeling uses H=20 bars FORWARD-LOOKING window
H = 20
close_change = bars['close'].diff(H).abs()  # Looks 20 bars AHEAD
```

**What this means:**
- Label at time `t` uses prices from `t` to `t+20`
- This creates **forward-looking bias** in the labels
- The model learns to predict a label that was created using future data

### 🔍 Detailed Example of the Problem

Suppose we're at bar 100,000 (in test set, year 2022):

```
Bar Index: 100,000
Label Creation:
  - Uses close[100,000] to close[100,020] (next 20 bars)
  - Calculates efficiency_ratio and |R_ATR| using FUTURE prices
  - Assigns label: "TREND" (because future 20 bars show strong trend)

Model Prediction:
  - Uses features at bar 100,000 (current + past data only)
  - Predicts: "TREND"
  - Evaluation: ✅ Correct!

But...
  - In REAL TRADING at bar 100,000, you don't know the next 20 bars yet!
  - The label "TREND" was assigned using information you won't have
```

---

## 🎯 What Does 93% Accuracy Really Mean?

### Two Interpretations

#### 1. **Optimistic View** (93% is Real)

**If you believe the labels represent true regime states:**
- The model achieves 93% accuracy at detecting market regimes
- Test set (2022) was never seen during training
- Performance generalizes to new time periods

**Use case:**
- Nowcasting: "What regime are we CURRENTLY in?" (using past 20 bars)
- This is still useful! Tells you how to trade RIGHT NOW

#### 2. **Realistic View** (93% is Inflated)

**The model is learning a forward-looking labeling function:**
- Labels use future data (H=20 bars ahead)
- Model learns: "Given current features, what will happen in next 20 bars?"
- This is actually **forecasting**, not classification

**The truth:**
- You cannot achieve 93% accuracy in REAL-TIME trading
- Real accuracy would be lower (maybe 60-80%?)
- You'd need to test with **true forward prediction**

---

## 📈 Temporal Nature of Data Split

### Timeline Visualization

```
|<-------- Training (60%) -------->|<-- Val (20%) -->|<-- Test (20%) -->|
2020-01-01                    2021-03-15        2022-01-01      2022-12-30

Train: 2020-2021 (15 months)
Val:   2021 mid-2022 (9 months)
Test:  2022 (12 months)           ← Model NEVER sees this during training
```

### Why Time-Based Split?

**Time-series data requires sequential splits:**
- ❌ **BAD**: Random shuffle (causes future data to leak into training)
- ✅ **GOOD**: Time-based split (respects temporal order)

**Our approach:**
```python
# Sequential split (correct for time series)
train_df = df_merged.iloc[:train_end]
val_df = df_merged.iloc[train_end:val_end]
test_df = df_merged.iloc[val_end:]
```

This ensures:
- Training data comes BEFORE validation data
- Validation data comes BEFORE test data
- No future information leaks into past

---

## 🔬 Model Training Process

### Step 1: Train on Training Set (60%)

```python
# Fit on training data only
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
model.fit(X_train_scaled, y_train)
```

### Step 2: Tune on Validation Set (20%)

```python
# Try different C values, evaluate on validation
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    model = LogisticRegression(C=C, ...)
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    val_f1 = f1_score(y_val, val_pred)  # Select best C based on this
```

**Best C found: 10.0** (based on validation F1 = 0.9269)

### Step 3: Final Training on Train+Val (80%)

```python
# Retrain on combined train+val with best C
X_trainval = np.vstack([X_train_scaled, X_val_scaled])
y_trainval = np.concatenate([y_train, y_val])
final_model.fit(X_trainval, y_trainval)
```

### Step 4: Evaluate ONCE on Test Set (20%)

```python
# Predict on test set (NEVER seen before)
test_pred = final_model.predict(X_test_scaled)
test_f1 = f1_score(y_test, test_pred)  # 0.9293 (93%)
```

**Key point**: Test set used ONLY for final evaluation, never for training or tuning.

---

## 📊 Test Set Performance Breakdown

### Overall Metrics (42,458 test samples)

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.9293** |
| **Accuracy** | **0.9360** |
| **Balanced Accuracy** | **0.9531** |

### Per-Class Performance on Test Set

| Class | Test Samples | Precision | Recall | F1-Score |
|-------|--------------|-----------|--------|----------|
| TREND | 10,540 | 90% | **96%** | 93% |
| TRANSITION | 25,665 | 98% | 91% | 95% |
| CHOP | 6,253 | 85% | **98%** | 91% |

### Test Set Confusion Matrix

```
                 Predicted →
Actual ↓        TREND    TRANSITION    CHOP
TREND          10,143       397         0      (96% correct)
TRANSITION      1,161    23,450     1,054     (91% correct)
CHOP                0       105     6,148     (98% correct)
```

**Analysis**:
- Only 397 TREND bars misclassified (out of 10,540)
- Only 105 CHOP bars missed (out of 6,253)
- Strong diagonal = excellent performance

---

## ⚠️ The Forward-Looking Label Problem

### How Labels Are Created

```python
H = 20  # Horizon (look ahead 20 bars)

# Calculate efficiency ratio over NEXT H bars
close_change = bars['close'].diff(H).abs()  # |close[t+H] - close[t]|
sum_abs_changes = bars['close'].diff().abs().rolling(H).sum()
efficiency_ratio = close_change / sum_abs_changes

# Label bar t based on what happens from t to t+20
if efficiency_ratio[t] >= 0.30 and r_atr[t] >= 3.0:
    label[t] = 'TREND'  # Based on FUTURE 20 bars!
```

### The Issue

**Bar at index t**:
- **Features**: Use data up to time t (current and past) ✅
- **Label**: Uses data from t to t+20 (FUTURE data) ⚠️

**In real trading**:
- At bar t, you have features[t] ✅
- But you DON'T know label[t] yet (need to wait 20 bars) ⚠️
- Model predicts label[t], but this label is based on future you haven't seen

### What This Means for Your Trading

**Scenario 1: Nowcasting (What regime are we in NOW?)**
- Current bar: t = 100,000
- Model predicts: "TREND"
- Real meaning: "The last 20 bars (99,980-100,000) showed a trend"
- **Useful**: You can trade based on current regime ✅

**Scenario 2: Forecasting (What regime will we be in?)**
- Current bar: t = 100,000
- You want: "Will next 20 bars be TREND?"
- Model predicts based on features[100,000]
- Label was created using bars 100,000-100,020
- **Problem**: This is circular - label uses the future you're trying to predict ⚠️

---

## 🔧 How to Fix the Forward-Looking Problem

### Option 1: Lag the Labels (Recommended for Real Trading)

```python
# Shift labels back by H bars
df_merged['regime_lagged'] = df_merged['regime'].shift(-H)

# Now at bar t:
# - Features: Use data up to t
# - Label: What regime was 20 bars AGO (at t-20)
# - Model learns: "Given current features, what regime are we transitioning FROM?"
```

**Trade-off**:
- ❌ Accuracy will drop (maybe to 60-80%)
- ✅ Model is now truly predictive
- ✅ Usable in real-time trading

### Option 2: Accept the Labeling (Current Approach)

```python
# Keep current labels
# Interpret as: "What regime are we CURRENTLY in?"
# Use for regime-aware trading, not regime prediction
```

**Trade-off**:
- ✅ High accuracy (93%)
- ⚠️ Not truly predictive
- ✅ Still useful for adaptive strategies

---

## 📊 Comparison: Our Split vs Random Split

### ❌ BAD: Random Split (Common Mistake)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Problems**:
1. Future data leaks into training
2. Bar at t+1 might be in training, bar at t in test
3. Model sees future patterns during training
4. **Artificially inflates performance**

### ✅ GOOD: Time-Based Split (What We Do)

```python
# Sequential split
train = df[:train_end]
test = df[val_end:]
```

**Benefits**:
1. Respects temporal order
2. Test set is chronologically AFTER training
3. Mimics real-world deployment
4. **Honest performance estimate**

---

## 🎯 Summary: Answering Your Questions

### Q1: How is data split?

**Answer**: **60/20/20 time-based sequential split**
- Train: 127,374 samples (2020-2021)
- Val: 42,458 samples (2021)
- Test: 42,458 samples (2022)
- Sequential order maintained (no random shuffle)

### Q2: What is prediction accuracy?

**Answer**: **93.6% accuracy, 92.9% macro F1 on test set**
- TREND: 96% recall
- CHOP: 98% recall
- TRANSITION: 91% recall
- These metrics are on 2022 data (test set)

### Q3: Is prediction on never-seen data?

**Answer**: **YES for features, BUT labels have forward-looking bias**

**Features perspective** ✅:
- Test features (2022) NEVER seen during training
- Model has zero knowledge of 2022 feature values
- True generalization to new time period

**Labels perspective** ⚠️:
- Labels created using H=20 bars forward-looking window
- At bar t, label uses data from t to t+20 (future)
- Creates subtle data leakage in labeling process
- Real-time trading accuracy likely 60-80%, not 93%

---

## 💡 Recommendations

### For Backtesting
✅ Current approach is fine
- 93% accuracy represents how well model identifies regimes
- Use for historical analysis

### For Live Trading
⚠️ Be cautious
- Expect lower accuracy (60-80%) in real-time
- Consider lagging labels by H bars
- Monitor slippage between predicted and actual regimes

### For Research
📊 Run additional test
- Create truly forward-looking test
- Predict bar t+20 regime using features at bar t
- This gives honest forecasting performance

---

## 📈 Timeline Diagram

```
Data: 212,290 bars (2020-2022)
     |<------------- 60% ----------->|<--- 20% --->|<--- 20% --->|
     Training (127,374)               Validation    Test (42,458)
     2020-01 to 2021-03              (42,458)       2022-01 to 2022-12
                                     2021-03 to 2022-01

Model Training Process:
1. Fit model on Training ────────────►
2. Tune hyperparameters using Validation ────────►
3. Retrain on Training + Validation ──────────────►
4. Evaluate ONCE on Test ─────────────────────────► 93% F1!
                                                     (Never seen before)
```

**Key Insight**: Test set (2022) was NEVER used for training or tuning, ensuring unbiased evaluation.

---

**Bottom Line**: The 93% F1 score is achieved on genuinely unseen 2022 data (from the model's perspective), making it a valid out-of-sample performance metric. However, the forward-looking nature of the labeling process means real-time trading performance will likely be lower. The model is excellent at **nowcasting** (identifying current regime) but may be optimistic for **forecasting** (predicting future regime).
