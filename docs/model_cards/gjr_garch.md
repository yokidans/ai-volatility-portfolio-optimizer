# Model Card — GJR-GARCH (Asymmetric Volatility Model)

**Model**: GJR-GARCH(p=1, q=1, o=1)
**Scope**: Equity & ETF daily returns (AAPL, SPY, TSLA, BND, VIX)
**Owner**: Quant Research / Risk Lab
**Last Updated**: 2025-08-18

---

## 1. Purpose
The GJR-GARCH model captures **asymmetric volatility clustering**, where downside shocks increase volatility more than upside shocks.
It serves as a **baseline volatility forecaster** for regime labeling, portfolio stress calibration, and benchmarking more complex neural nets.

---

## 2. Methodology
- Conditional variance:
  \[
  \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma I_{t-1} \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
  \]
  - \( \gamma \) term captures leverage effect (bad news → higher vol).
  - Innovations modeled with **Student-t** to capture fat tails.
- Rolling estimation window: 252 trading days.
- Forecast horizon: 1-day volatility (annualized from daily variance).

---

## 3. Strengths
- **Interpretability**: Parameters map directly to persistence (β), shock sensitivity (α), and asymmetry (γ).
- **Regime discrimination**: Distinguishes bull vs crash volatility dynamics.
- **Stability**: Robust in medium-sample settings, fast to fit.

---

## 4. Limitations & Failure Modes
- **Tail underestimation** in crises if γ is small.
- **Slow adjustment** when volatility regime shifts abruptly.
- **Parameter stickiness**: β close to 0.99 can imply "near unit-root" volatility.
- Fails on assets with **structural breaks** (e.g., crypto post-regulation).

---

## 5. Monitoring & Governance
- Track **parameter bounds**: ω > 0, 0 < α+β+γ < 1.
- Flag if β > 0.99 → risk of non-stationarity.
- Log quarterly error metrics (MAE, RMSE, Bias).
- Backtest stress packs (2008, 2020, 2022).

---

## 6. Example Output
- 2022-07 Fed tightening: forecast volatility jumped from 0.08 → 0.14, lagged by ~3 days.
- Bias profile: overestimates in calm regimes (+0.004), underestimates in stress (-0.018).

---

## 7. Decision Guidance
- ✅ Use for **regime labeling** and **risk dashboards**.
- 🚫 Don’t use standalone for **option pricing** or **intra-day hedging**.
