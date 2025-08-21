# Model Card — DCC-GARCH (Dynamic Conditional Correlation)

**Model**: Engle’s DCC-GARCH(1,1)
**Scope**: Multi-asset covariance estimation (TSLA, SPY, BND)
**Owner**: Quant Risk & Portfolio Lab
**Last Updated**: 2025-08-18

---

## 1. Purpose
DCC-GARCH models **time-varying correlations** between assets.
Essential for **portfolio optimization under stress**, where static correlations (e.g., sample cov) underestimate risk.

---

## 2. Methodology
- Step 1: Fit univariate GARCH to each asset → standardized residuals.
- Step 2: Estimate dynamic correlation matrix:
  \[
  Q_t = (1-a-b)\bar{Q} + a\epsilon_{t-1}\epsilon_{t-1}' + b Q_{t-1}
  \]
- Covariance = diag(σ) × Rt × diag(σ).

---

## 3. Strengths
- **Crisis adaptivity**: correlations spike (equities–bonds ~0.6 in 2022).
- **Positive-definiteness** guaranteed by design.
- Well-tested in institutional risk.

---

## 4. Limitations & Failure Modes
- **Over-smoothing**: correlation updates slowly vs realized correlation.
- **Estimation instability** with >10 assets.
- Requires **clean, synchronous data** (missing days → crash).

---

## 5. Monitoring & Governance
- Track **a+b < 1** → stationarity check.
- Monitor eigenvalues of covariance (must stay > 0).
- Validate against realized rolling correlations.
- Flag if equity–bond correlation > 0.5 (historical stress alert).

---

## 6. Example Output
- 2008 crisis: equity–bond correlation rose from -0.3 to +0.4.
- 2022 tightening: correlation surged +0.6, flagged as tail co-movement risk.

---

## 7. Decision Guidance
- ✅ Use for **CVaR optimization**, **stress test packs**, and **frontier estimation**.
- 🚫 Don’t use for **intraday hedging** or **crypto pairs** (structural breaks).
