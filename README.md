# StockCast

XGBoost and LSTM stock forecasting with walk-forward backtesting. Built to measure model performance honestly — the main failure mode in finance ML is data leakage that silently inflates backtest numbers.

## What it does

- Fetches 1–5 years of OHLCV data via yfinance
- Engineers 34 features: lag returns, rolling MAs, RSI, MACD, volatility, volume ratio
- Trains XGBoost and a PyTorch LSTM on walk-forward temporal splits (no random k-fold, no lookahead)
- Compares both models side-by-side on identical fold boundaries — MAPE, RMSE, directional accuracy
- Computes SHAP feature attribution for XGBoost
- Finds similar stocks via PCA on 34-feature profiles + cosine similarity across a 20-ticker universe
- Serves predictions through FastAPI with in-memory model caching (~85ms p50)
- Stores forecast history in MongoDB

## Models

**XGBoost** — one row per prediction (today's feature snapshot). Scale-invariant, fast to train, strong on non-linear feature interactions.

**LSTM** — 40-day rolling window (~2 months). Captures momentum, trend exhaustion, and regime patterns single-row models miss. 2-layer, hidden size 64, dropout 0.2, Adam/MSE. Features are StandardScaled per fold (fit on train only).

Both models use identical fold boundaries for a fair comparison.

## Leakage fix

Standard k-fold lets a row from day 300 appear in training while day 250 is in validation — invalid for time series. Walk-forward helps, but there's a subtler issue: the last `horizon` rows of each training fold have targets computed from prices that fall inside the test window. The fix is to end training at `train_end - horizon` so no training label touches a test-period price.

With the fix applied, directional accuracy dropped from ~64% to ~57% and RMSE increased. The gap was leakage.

## Metrics

All metrics are computed on **returns**, not prices. Price-based RMSE conflates model skill with price scale. Return-based RMSE is scale-invariant and cross-ticker comparable.

Directional accuracy is computed on conviction predictions only — days where `|predicted return| >= median(|predicted returns|)`. Near-zero predictions are noise; filtering them out gives a meaningful signal. The `+Xpp vs random` label shows margin above the 50% baseline.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Vue 3, Tailwind CSS, Chart.js |
| Backend | Node.js, Express, MongoDB |
| ML service | FastAPI, XGBoost, PyTorch, SHAP, scikit-learn, yfinance, pandas |

## Running locally

**1. ML service**
```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**2. Backend**
```bash
cd backend
npm install
cp .env.example .env   # fill in MongoDB URI
npm run dev
```

**3. Frontend**
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

**Smoke test:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","start_date":"2020-01-01","end_date":"2025-01-01","horizon":5}'
```

## Limitations

- No transaction cost simulation; directional accuracy alone does not imply profitability
- Model cache resets on service restart; no persistence for trained models
- Regime changes (e.g. 2020 crash) can invalidate a model trained on prior data
