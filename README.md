# FINA4380 Project

This is a Quant portfolio strategy.

## 1. Strategy Design

### Signal Generation

- Factor Selection
  - Fundamental: 
  - Statistical:
  - Macro:

- Construction of Factor Return
  - Non-negative Matrix Factorization (NMF)
    - Dimension deduction, construct statistical factor
    - Macro

  - Quantile Approach: (Fundamental)

  - Information Coefficient (IC)
    - filter low correlation factors

- Kalman Filter
  - State Variables: Factor Exposure ($\beta$)
  - Observed Variables: Asset Return
  - Output: Expected Future Return

### Portfolio Construction

- Long $x$ quantile, Short $1-x$ quantile. x is determined by optimizing Sharpe ratio. 


## 2. Backtest

### Data

- Bloomberg

### Tool

- Backtrader

### Performance Evaluation

- VaR, Max Drawdown, Sharpe Ratio, Cumulative Return

