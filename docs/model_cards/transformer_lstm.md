# Model Card — Transformer-LSTM Hybrid (Quantile Volatility Forecaster)

**Model**: Transformer encoder + LSTM decoder with quantile loss
**Scope**: Multi-asset volatility forecasting, horizon = 1–5 days
**Owner**: Quant Research / ML Lab
**Last Updated**: 2025-08-18

---

## 1. Purpose
This model combines **sequence attention** (transformer) with **temporal persistence** (LSTM) to forecast volatility distributions.
Unlike GARCH, it outputs **quantiles (q05, q50, q95)** → supporting **probabilistic risk management**.

---

## 2. Methodology
- Inputs: return lags, realized vol, macro features (rates, VIX, sentiment).
- Transformer layers capture **cross-asset attention** and exogenous drivers.
- LSTM decoder enforces **temporal smoothness**.
- Loss: **pinball loss** for quantiles, e.g.
  \[
  L_q = \max\{ q(y-\hat{y}), (q-1)(y-\hat{y}) \}
  \]
- Dropout used as **Bayesian approximation** (MC dropout).

---

## 3. Strengths
- **Distributional forecasts**: delivers volatility bands instead of point estimates.
- **Regime adaptivity**: attention picks up macro shocks earlier than GARCH.
- **Tail-awareness**: q95 quantile captures worst-case scenarios.

---

## 4. Limitations & Failure Modes
- **Data hungry**: needs ≥ 5 years clean time series.
- **Interpretability** lower than GARCH — requires SHAP/attention viz.
- Can **hallucinate volatility bands** if trained on sparse crises.
- Overfits without strict CV and dropout seeds.

---

## 5. Monitoring & Governance
- Monitor **quantile coverage**: actual % inside [q05, q95] should ~90%.
- Log calibration drift monthly.
- Retrain if out-of-sample coverage < 85%.
- Freeze random seeds in CI for reproducibility.

---

## 6. Example Output
- During 2022-07 tightening: q95 widened to 0.22 vs realized 0.18 → **captured fat-tail stress**.
- 2023 calm regime: q05–q95 band shrank to [0.06, 0.10].

---

## 7. Decision Guidance
- ✅ Use for **risk budgeting** and **portfolio CVaR optimization**.
- 🚫 Don’t use standalone for **capital allocation approval** without backtest.
