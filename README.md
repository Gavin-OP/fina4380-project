# FINA4380 Project

This is a Quant portfolio strategy.

## Common Equity Quant Strategies

### **Equity Statistical Arbitrage**

- Utilize price data and price-related (i.e. correlation, volatility) and market data (i.e. volume, order book) to determine the existence of patterns. 
- Signal Types[^1]:
  - Mean Reversion: revert to an equilibrium (i.e. paired trading, duel class share arbitrage, volatility strategy, contrarian). 
  - Momentum: persistent price movements (i.e., trend).

### **Quantitative Equity Market Neutral (QEMN)**

- Utilize various data sources as predictors/metrics to generate signals and rank/score stocks in varying proportions by weighted signals. Buy certain proportions and short certain proportions. 
- Procedures:
  - Data Collection & Processing
    - Fundamental data: financial (i.e. financial statement, cash flow, PE, PB, dividend), non-financial (i.e. industry trends, macroeconomic indicators).
    - Technical data: (i.e. MA, RSI, volume).
    - Sentiment data: event-driven (i.e. analyst earnings estimates, NLP, announced mergers, share buy-backs, index rebalancing, insider buying/selling).
    - Alternative data: non-traditional (i.e. satellite imagery, credit card data, weather patterns).
  - Signal Generation: generate signals for each stock with data as predictors.  
  - Signal Combination & Weighting: rank each stock against weighted signals. Machine learning can be used to process various scores. 
    - Kelly criterion
  - Portfolio Construction & Risk Management
    - Efficient Frontier
    - Stop Loss Order
    - Position Size Limits
    - Portfolio Factor Exposure Limits
    - Liquidity Constraint

| Summary[^2]                                                 | Statistical Arbitrage                     | QEMN                                                         |
| ----------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Typical Market Directionality                               | Primarily Market Neutral                  | Primarily Market Neutral                                     |
| Observed $\beta$ to Traditional Assets (Equities and Bonds) | Typically Very Low                        | Typically Very Low                                           |
| Long/short bias                                             | None                                      | None                                                         |
| Historical volatility                                       | Lower volatility than typical HF universe | Lower volatility than typical HF universe                    |
| Typical factor exposure                                     | Tightly hedged to generic factors         | May be hedged to genetic factors, but tends to take specific exposure to certain equity risk premia |
| Liquidity                                                   | Generally highly liquid                   | Generally highly liquid                                      |
| Leverage                                                    | Can vary significantly: typically 3-8x    | Can vary significantly: typically 3-8x                       |

## Common Multi-Asset Class Quant Strategy[^2]

- **Managed futures/Commodity trading advisors ("CTAs")/Global macro**
- **Quant macro and global asset allocation ("GAA")** 
- **Alternative risk premia**

## Trading Strategy Design

### Predictors

- Bollinger Bands
- Relative Strength Index
- Moving Average Convergence Divergence
- Rate of Change
- Risk Factors
- Value
- Growth
- Market Cap
- Sector
- Momentum
- Geography

## References

[Beginner's Guide to Quantitative Trading | QuantStart](https://www.quantstart.com/articles/Beginners-Guide-to-Quantitative-Trading/)

[Quant hedge fund primer: demystifying quantitative strategies | Aurum](https://www.aurum.com/insight/thought-piece/quant-hedge-fund-strategies-explained/)

[WebClinic GitHub](https://github.com/webclinic017)

[BigQuant](https://bigquant.com/)

[邢不行 | 量化小讲堂](https://www.quantclass.cn/home)

[The Quant's Playbook](https://quantgalore.substack.com/)

[Trading Code | Programming For Traders](https://www.tradingcode.net/)

[Technical Analysis | Donchian Channel Indicator](https://medium.com/gitconnected/an-algo-trading-strategy-which-made-8-371-a-python-case-study-58ed12a492dc)

[SMA and EMA Crossover](https://forexop.com/strategy/sma-and-ema-crossover/)

[OpenQuant Blog](https://openquant.co/blog)

[Dual-Class Shares Arbitrage](https://alphaarchitect.com/2011/03/dual-class-shares-a-first-class-strategy/)

[Statistics & ML Concepts for Quant Finance](https://openquant.co/blog/statistics-and-ml-concepts-for-quant-finance-interview)

[Hands-on machine learning for algorithmic trading: design and implement investment strategies based on smart algorithms that learn from data using Python: Chapter 4](https://julac-cuhk.primo.exlibrisgroup.com/discovery/fulldisplay?docid=alma991039741106303407&context=L&vid=852JULAC_CUHK:CUHK&lang=en&search_scope=All&adaptor=Local)

[PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading | GitHub](https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading)

[Trading strategies | Quantified Strategies](https://www.quantifiedstrategies.com/category/trading-strategies/)

[Trading indicators | Quantified Strategies](https://www.quantifiedstrategies.com/category/trading-indicators/)

[Top 10 Quantitative Trading Strategies with Python](https://zodiactrading.medium.com/top-10-quantitative-trading-strategies-with-python-82b1eff67650)

[Quantitative Trading - Overview, Components, How It Works | Wall Street Oasis](https://www.wallstreetoasis.com/resources/skills/trading-investing/quantitative-trading)

[Quantitative Finance (arxiv.org)](https://arxiv.org/archive/q-fin)

[An introduction to NMF and how it differs from PCA | Medium](https://medium.com/@354047384/an-introduction-to-nmf-and-how-it-differs-from-pca-3d8e4080df83)

[The Inner Workings of a Quant Contrarian Strategy | Morningstar](https://www.morningstar.com/articles/650328/the-inner-workings-of-a-quant-contrarian-strategy)

[^1]: Signals in the context of quant hedge funds refer to mathematical models and algorithms that analyse large volumes of financial data to identify patterns and trends. These signals are used to make investment decisions and execute trades.
[^2]: Retrieved from [Quant hedge fund primer: demystifying quantitative strategies | Aurum](https://www.aurum.com/insight/thought-piece/quant-hedge-fund-strategies-explained/)
