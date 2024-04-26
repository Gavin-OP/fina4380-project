#  Quant Strategy Category

## Equity Quant Strategies

### **Equity Statistical Arbitrage**

- Utilize price data and price-related (i.e. correlation, volatility) and market data (i.e. volume, order book) to determine the existence of patterns.
- Signal Types[^1]:
  - Mean Reversion: revert to an equilibrium (i.e. paired trading, duel class share arbitrage, volatility strategy, contrarian).
  - Momentum: persistent price movements (i.e., trend).

### **Quantitative Equity Market Neutral (QEMN)**

- Utilize various data sources as predictors/metrics to generate signals and rank/score stocks in varying proportions by weighted signals. Buy certain proportions and short certain proportions.

- General Procedures:

  1. Data Collection & Processing

     - Fundamental data: financial (i.e. financial statement, cash flow, PE, PB, dividend, market cap), non-financial (i.e. industry trends, momentum, macroeconomic indicators, sector).

     - Technical data: (i.e. MA, RSI, volume, ROC, MACD, Bollinger Bands).

     - Sentiment data: event-driven (i.e. analyst earnings estimates, NLP, announced mergers, share buy-backs, index rebalancing, insider buying/selling).

     - Alternative data: non-traditional (i.e. satellite imagery, credit card data, weather patterns, geography, google search), mimicking (i.e. Barra risk factor).

  2. Signal Generation: generate signals for each stock with data as predictors.

  3. Signal Combination & Weighting[^3]: rank each stock against weighted signals. Machine learning can be used to process various scores.

     - Kelly criterion

  4. Portfolio Construction

     - Efficient Frontier

     - Smart $\beta$

| Summary[^2]                                                 | Statistical Arbitrage                     | QEMN                                                         |
| ----------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Typical Market Directionality                               | Primarily Market Neutral                  | Primarily Market Neutral                                     |
| Observed $\beta$ to Traditional Assets (Equities and Bonds) | Typically Very Low                        | Typically Very Low                                           |
| Long/short bias                                             | None                                      | None                                                         |
| Historical volatility                                       | Lower volatility than typical HF universe | Lower volatility than typical HF universe                    |
| Typical factor exposure                                     | Tightly hedged to generic factors         | May be hedged to genetic factors, but tends to take specific exposure to certain equity risk premia |
| Liquidity                                                   | Generally highly liquid                   | Generally highly liquid                                      |
| Leverage                                                    | Can vary significantly: typically 3-8x    | Can vary significantly: typically 3-8x                       |

## Multi-Asset Class Quant Strategy[^2]

- **Managed futures/Commodity trading advisors ("CTAs")/Global macro**
- **Quant macro and global asset allocation ("GAA")**
- **Alternative risk premia**

[^1]: Signals in the context of quant hedge funds refer to mathematical models and algorithms that analyze large volumes of financial data to identify patterns and trends. These signals are used to make investment decisions and execute trades.
[^2]: Retrieved from [Quant hedge fund primer: demystifying quantitative strategies | Aurum](https://www.aurum.com/insight/thought-piece/quant-hedge-fund-strategies-explained/)

## References

[Quant hedge fund primer: demystifying quantitative strategies | Aurum](https://www.aurum.com/insight/thought-piece/quant-hedge-fund-strategies-explained/)

[Kenneth R. French - Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Benchmarks)