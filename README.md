# Crypto Statistical Arbitrage

A reproducible research framework for **statistical arbitrage in cryptocurrencies**, built to demonstrate the full quant workflow: from **data acquisition** and **signal design** to **backtesting**, **robustness tests**, and **portfolio construction**.  

This project is designed to showcase **research rigor** and **trading relevance**, making it suitable for both a quant portfolio and professional applications.

---

## TL;DR
- Pull OHLCV from exchanges (via `ccxt`).
- Generate signals:
  - **Cross-sectional Reversal** (short-term mean reversion).
  - **Cross-sectional Momentum** (medium-term trend).
- Apply **residualization vs BTC**, **banding**, and **volatility scaling**.
- Backtest with **realistic costs, turnover, and rebalancing schemes**.
- Evaluate **Sharpe, drawdowns, turnover, cost drag, and alpha vs BTC**.
- Combine strategies into a **diversified multi-sleeve portfolio**.

---

## Repo Structure
crypto-stat-arb/  
│── notebooks/ # Analysis & results walkthroughs  
│── src/ # Core research code   
│ ├── signals # Reversal & momentum signal construction  
│ ├── backtest # Portfolio construction & simulation  
│ ├── etc.  
│── data/ # Example price/return datasets  
│── README.md # This file  

---

## Results Summary

### Cross-Sectional Reversal
- **Best config:**  
  `k=4`, `band=2.5`, `beta_win=168` (weekly residualization),  
  `every=24` (daily rebalancing), `vol_win=24`.
- **Full-sample net Sharpe:** **1.77** (gross 2.07).  
- **Ann. return:** 0.366 | **Ann. vol:** 0.207 | **Turnover:** ~89/yr.  
- **Robust OOS:** Train Sharpe 1.56 vs Test Sharpe 2.19 → strong generalization.  
- **Cost-resilient:** Net Sharpe remains ~0.95 even at 20 bps.

**Takeaway:**  
Reversal works best with **short lookbacks (k=2–4)**, strong banding, and daily rebalancing. Performance is robust across folds, costs, and parameter perturbations.

---

### Cross-Sectional Momentum
- **Best config:**  
  `k=336` (~14 days), `band=2.5`, no residualization,  
  `every=720` (~30 days), `vol_win=168`.
- **Full-sample net Sharpe:** **1.29**.  
- **Ann. return:** 0.277 | **Ann. vol:** 0.184 | **Turnover:** ~4.6/yr (cost drag ≈ 0.003).  
- **OOS Sharpe:** 1.58 (test) vs 1.21 (train).  
- **Alpha vs BTC:** ~0.26 ann., but not statistically significant (t ≈ 1.41).  
- **Market exposure negligible:** β ≈ 0.03, R² < 1%.

**Takeaway:**  
Momentum is competitive with **long lookbacks** and **monthly rebalancing**, offering **very low turnover** and near-zero market beta.

---

### Diversification & Mixed Portfolios
- **Sleeve correlation:** Corr(reversal, momentum) ≈ −0.055 → strong diversification.  
- **50/50 mix:** Sharpe **2.39**, ann. vol 0.135, ann. ret 0.321.  
- **Equal-vol mix:** Sharpe **2.38**, ann. vol 0.687, ann. ret 1.64 → higher return at higher risk.  
- **OOS stitched performance:** Sharpe ~**2.5** with max drawdowns <8%.  
- **Alpha vs BTC:** Ann. α ≈ **0.31**, t ≈ **2.5** (statistically significant).

**Takeaway:**  
- Reversal and Momentum are **complementary**; their low correlation enables a large Sharpe uplift.  
- **50/50 static mix** achieves the highest risk-adjusted returns.  
- **Equal-vol mix** delivers higher returns at higher risk.  
- Both generate **statistically significant alpha** vs BTC with **near-zero beta**.

---

## Robustness Checks
- **Walk-forward analysis:**  
  Expanding and rolling windows confirm stable out-of-sample Sharpe ~1.55 (single sleeve) and ~2.55 (mixed).  
- **Transaction cost sensitivity (equal-vol mix):**  
  - **7 bps:**  Exp mean Sharpe **2.32**, Roll mean Sharpe **2.34**  
  - **10 bps:** Exp **2.19**, Roll **2.22**  
  - **20 bps:** Exp **1.77**, Roll **1.80** 

---

## Overall Conclusions
- The **equal-vol mix of Reversal + Momentum** achieves **robust, high OOS Sharpe (~2.5)** with shallow drawdowns (<8%).  
- The sleeve produces **positive, significant alpha** vs BTC with **beta ≈ 0**, offering **diversified crypto exposure** rather than simple market beta.  
- Performance is **cost-resilient**, attractive even at 20 bps.  
- **Reversal** drives high-frequency edge; **Momentum** contributes low-cost persistence; **together** they deliver a balanced, robust profile.  

---

## Limitations
- Results are based on **~2.6 years of hourly data** across 12 assets → limited regime coverage.  
- Costs are simplified; real-world frictions may vary.  
- Outcomes depend on parameter search and benchmark choice.  
- Treat results as **indicative, not definitive**.

---
