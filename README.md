# FINA4380 Project

This is a Quant portfolio strategy.

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

### Common Quant Strategies

- **Equity statistical arbitrage**
  - utilize price data and its derivatives, such as correlation, volatility, and other forms of market data, such as volume and order-book information to determine the existence of patterns. 
  - Signal types:
    - Mean-reversion: revert to an equilibrium level (i.e. Paired trading). 
    - Momentum: price movements will be more persistent (i.e., trend).
    - Event-driven: analyst earnings estimates, NLP, announced mergers, share buy-backs, index rebalancing, and corporate insider buying/selling.
- **Quantitative equity market neutral**
  - take fundamental and/or event-oriented data, such as balance sheet information and cash flow statement statistics, and systematically rank/score stocks against these metrics in varying proportions. The weights of the scores of the different fundamental data sources may be fixed or dynamic. Use machine learning algorithms to analyze and process various signals. 
  - Signal types:
    - Fundamental data: financial data such as earnings, revenue, profit margins, and cash flow, as well as non-financial data such as industry trends and macroeconomic indicators.
    - Technical data: past market trends and patterns, such as moving averages, relative strength, and trading volume.
    - Sentiment data: investor sentiment and market sentiment, such as news articles, social media posts, and analyst reports.
    - Alternative data: non-traditional data sources, such as satellite imagery, credit card data, and weather patterns, which can provide insight into market trends and consumer behavior.

- **Managed futures/CTAs**
- **Quant macro**
- **Alternative risk premia**
- **Quant volatility**

| Risk Return Summary                                      | Statistical arbitrage                     | QEMN                                                                                                | CTAs                                                                                       | Quant macro/GAA                                                                            | Alternative risk premia                                                                                          |
| -------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Typical assets traded                                    | Equities                                  | Equities                                                                                            | Liquid futures – equity, fixed income, commodities.                                        | Similar to CTAs + cash instruments, bonds, FX, ETFs, Derivatives                           | Primarily equities, but may also trade some derivatives and instruments similar to quant macro                   |
| Typical market directionality /neutrality                | Primarily market neutral                  | Primarily market neutral                                                                            | Generally directional                                                                      | Generally relative value. Some have directional positions                                  | Generally market neutral long-term (some exceptions)                                                             |
| Observed beta to traditional assets (equities and bonds) | Typically very low                        | Typically very low                                                                                  | Typically low                                                                              | Typically low                                                                              | Typically low to moderate                                                                                        |
| Long/short bias                                          | None                                      | None                                                                                                | May be directional but should have no systemic bias to be long or short over the long-term | May be directional but should have no systemic bias to be long or short over the long-term | Typically no bias                                                                                                |
| Historical volatility                                    | Lower volatility than typical HF universe | Lower volatility than typical HF universe                                                           | Higher volatility than wider HF universe                                                   | Higher volatility than wider HF universe                                                   | Potential exposure to large factor moves – can be large/long drawdowns                                           |
| Typical factor exposure                                  | Tightly hedged to generic factors         | May be hedged to generic factors, but tends to take specific exposure to certain equity risk premia | Typically highly exposed to momentum                                                       | Varied, may be tightly hedged; could have a momentum or value bias                         | High factor exposure by design. Typical ARP fund looks to offer diversified exposure to many risk-premia factors |
| Liquidity                                                | Generally highly liquid                   | Generally highly liquid                                                                             | Generally highly liquid                                                                    | Generally highly liquid                                                                    | Generally highly liquid                                                                                          |
| Leverage                                                 | Can vary significantly: typically 3-8x    | Can vary significantly: typically 3-8x                                                              | Typical 2-4x (with MTE typically 10-30%)                                                   | Typical 2-4x (with MTE typically 15-40%)                                                   | Varied (typically 1.5 to 2.0x)                                                                                   |

- Long Short Strategy
- Trend Following Strategy
- Volatility Trading

### Capital Allocation

- Kelly criterion
- Efficient frontier

### Risk Management

- Stop loss order
- Position size limits
- Portfolio factor exposure limits
- Liquidity constraint

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
