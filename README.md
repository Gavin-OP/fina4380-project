# FINA4380 Project

This is a Quant portfolio strategy.

## Quant Strategy Identification

- **Quantitative Equity Market Neutral (QEMN)**

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

## Strategy Backtesting

- Sharpe ratio
- BackTrader

## Risk Management

- Stop Loss Order
- Position Size Limits
- Portfolio Factor Exposure Limits
- Liquidity Constraint
- Drawdown
- Percentage of Profitable Trades

## To Do

- latency arbitrage
- index arbitrage
- risk parity arbitrage

## References

### Overview

[Beginner's Guide to Quantitative Trading | QuantStart](https://www.quantstart.com/articles/Beginners-Guide-to-Quantitative-Trading/)

[Quant hedge fund primer: demystifying quantitative strategies | Aurum](https://www.aurum.com/insight/thought-piece/quant-hedge-fund-strategies-explained/)

[Quantitative Trading - Overview, Components, How It Works | Wall Street Oasis](https://www.wallstreetoasis.com/resources/skills/trading-investing/quantitative-trading)

[Quantitative Investing - What Is It, Strategies, Examples, Benefits | Wall Street Mojo](https://www.wallstreetmojo.com/quantitative-investing/)

[Trading Quantitative Strategies Explained | Admirals](https://admiralmarkets.com/education/articles/automated-trading/trading-quantitative-strategies)

### Predictors, Signals, Strategies

[SMA and EMA Crossover Strategy](https://forexop.com/strategy/sma-and-ema-crossover/)

[Donchian Channel Indicator | Medium](https://medium.com/gitconnected/an-algo-trading-strategy-which-made-8-371-a-python-case-study-58ed12a492dc)

[Dual-Class Shares Arbitrage Strategy](https://alphaarchitect.com/2011/03/dual-class-shares-a-first-class-strategy/)

[Top 10 Quantitative Trading Strategies with Python | Medium](https://zodiactrading.medium.com/top-10-quantitative-trading-strategies-with-python-82b1eff67650)

[Trading strategies | Quantified Strategies](https://www.quantifiedstrategies.com/category/trading-strategies/)

[Trading indicators | Quantified Strategies](https://www.quantifiedstrategies.com/category/trading-indicators/)

[Constructing a Machine Learning Model](https://medium.datadriveninvestor.com/introduction-to-quantitative-trading-constructing-a-machine-learning-model-9165f2d986de)

[Pairs-Trading](https://medium.datadriveninvestor.com/citadels-strategy-anyone-can-use-pairs-trading-7b81428a6c67)

### Platforms

[WebClinic GitHub](https://github.com/webclinic017)

[Quantitative Finance (arxiv.org)](https://arxiv.org/archive/q-fin)

[BigQuant Wiki](https://bigquant.com/wiki/home)

[邢不行 | 量化小讲堂](https://www.quantclass.cn/home)

[The Quant's Playbook](https://quantgalore.substack.com/)

[OpenQuant Blog](https://openquant.co/blog)

[Trading Code | Programming For Traders](https://www.tradingcode.net/)

[SigTech User Guide](https://guide.sigtech.com/)

[Industry insights and news for Quant | SigTech](https://sigtech.com/insights/)

[Articles | QuantStart](https://www.quantstart.com/articles/)

[Quantivity | Some Articles Before 2012](https://quantivity.wordpress.com/)

[Elite Trader](https://www.elitetrader.com/et/)

[Seeking Alpha](https://seekingalpha.com/)

[QuantPedia](https://quantpedia.com/blog/)

[Quant Investing Research Library | Savvy Investor](https://www.savvyinvestor.net/quant-and-tools/articles-and-white-papers)

### Techniques

[An introduction to NMF and how it differs from PCA | Medium](https://medium.com/@354047384/an-introduction-to-nmf-and-how-it-differs-from-pca-3d8e4080df83)

[Statistics & ML Concepts for Quant Finance](https://openquant.co/blog/statistics-and-ml-concepts-for-quant-finance-interview)

[Hands-on machine learning for algorithmic trading: design and implement investment strategies based on smart algorithms that learn from data using Python: Chapter 4](https://julac-cuhk.primo.exlibrisgroup.com/discovery/fulldisplay?docid=alma991039741106303407&context=L&vid=852JULAC_CUHK:CUHK&lang=en&search_scope=All&adaptor=Local)

[PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading | GitHub](https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading)

[Stock price prediction based on PCA-LSTM model](https://dl.acm.org/doi/abs/10.1145/3545839.3545852)

### Books

[Multi-Factor Models and Signal Processing Techniques : Application to Quantitative Finance](https://www.proquest.com/docview/2131273191/$N?accountid=10371&sourcetype=Books)



[^1]: Signals in the context of quant hedge funds refer to mathematical models and algorithms that analyse large volumes of financial data to identify patterns and trends. These signals are used to make investment decisions and execute trades.
[^2]: Retrieved from [Quant hedge fund primer: demystifying quantitative strategies | Aurum](https://www.aurum.com/insight/thought-piece/quant-hedge-fund-strategies-explained/)
[^3]: please refer to [Hands-on machine learning for algorithmic trading: design and implement investment strategies based on smart algorithms that learn from data using Python: Chapter 4](https://julac-cuhk.primo.exlibrisgroup.com/discovery/fulldisplay?docid=alma991039741106303407&context=L&vid=852JULAC_CUHK:CUHK&lang=en&search_scope=All&adaptor=Local), and the GitHub repository [PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading | GitHub](https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading)
