import datetime
import os
import pandas as pd
import numpy as np
import quantstats as qs
import backtrader as bt
import matplotlib.pyplot as plt


class PandasData(bt.feeds.PandasData):
    lines = ("open", "close")
    params = (
        ("datetime", None),  # use index as datetime
        ("open", 0),  # the [0] column is open price
        ("close", 1),  # the [1] column is close price
        ("high", 0),
        ("low", 0),
        ("volume", 0),
        ("openinterest", 0),
    )


class BLStrategy(bt.Strategy):
    # list for tickers
    params = (("stocks", []), ("printnotify", False), ("printlog", False))

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self, weights):
        self.datafeeds = {}
        self.weights = weights  # weights for all stocks
        self.committed_cash = 0
        self.bar_executed = 0

        # price data and order tracking for each stock
        for i, ticker in enumerate(self.params.stocks):
            self.datafeeds[ticker] = self.datas[i]

    def notify_order(self, order):
        if self.params.printnotify:
            if order.status in [order.Submitted, order.Accepted]:
                print(
                    f"Order for {order.size} shares of {order.data._name} at {order.created.price} is {order.getstatusname()}")

            if order.status in [order.Completed]:
                if order.isbuy():
                    print(
                        f"Bought {order.executed.size} shares of {order.data._name} at {order.executed.price}, cost: {order.executed.value}, comm: {order.executed.comm}"
                    )
                elif order.issell():
                    print(
                        f"Sold {order.executed.size} shares of {order.data._name} at {order.executed.price}, cost: {order.executed.value}, comm: {order.executed.comm}"
                    )

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(
                    f"Order for {order.size} shares of {order.data._name} at {order.created.price} is {order.getstatusname()}")

    # for each date, place orders according to the weights
    def next(self):
        date = self.data.datetime.date(0)
        weights = self.weights.loc[date.strftime("%Y-%m-%d")]

        if not self.position:
            self.log("We do not hold any positions at the moment")
        self.log(f"Total portfolio value: {self.broker.getvalue()}")

        for ticker in self.params.stocks:
            # Calculate the target value for this stock based on the target percentage
            data = self.datafeeds[ticker]
            target_percent = weights[ticker]

            self.log(
                f"{ticker} Open: {data.open[0]}, Close: {data.close[0]}, Target Percent: {target_percent}")
            self.orders = self.order_target_percent(
                data, target=target_percent)


class PortfolioValueObserver(bt.Observer):
    lines = ("value",)
    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()


# class SortinoRatio(bt.Analyzer):
#     def __init__(self):
#         self.returns = []

#     def start(self):
#         self.returns = []

#     def next(self):
#         self.returns.append(self.strategy.pnl)

#     def get_analysis(self):
#         returns = np.array(self.returns)
#         downside_returns = returns[returns < 0]
#         downside_deviation = np.std(downside_returns, ddof=1)
#         # Assuming daily returns and 252 trading days
#         annualized_return = np.mean(returns) * 252
#         if downside_deviation != 0:
#             sortino_ratio = annualized_return / downside_deviation
#         else:
#             sortino_ratio = None
#         return {'sortino_ratio': sortino_ratio}


def SyntheticData(duration=499, num_stocks=5):
    # date range
    start_date = datetime.date(2002, 1, 1)
    end_date = start_date + datetime.timedelta(days=duration)
    date_range = pd.bdate_range(start=start_date, end=end_date)

    # fake prices for x stocks
    num_days = len(date_range)
    prices = pd.DataFrame(
        np.random.normal(loc=100, scale=10, size=(num_days, num_stocks)), index=date_range, columns=[f"Stock{i}" for i in range(1, num_stocks + 1)]
    )
    prices = prices.reset_index().rename(columns={"index": "Date"})
    prices_open = prices.copy()
    prices_open.iloc[:, 1:] = prices_open.iloc[:, 1:] + \
        np.random.normal(loc=0, scale=1, size=prices_open.iloc[:, 1:].shape)

    # fake weights for x stocks
    weights = np.random.uniform(-1, 1, (num_days, num_stocks))
    weights = weights / weights.sum(axis=1, keepdims=True)
    weights = pd.DataFrame(weights, index=date_range, columns=[
                           f"Stock{i}" for i in range(1, num_stocks + 1)])
    weights = weights.reset_index().rename(columns={"index": "Date"})

    return prices, prices_open, weights


def data_cleaning(data: pd.DataFrame, start: int = None, end: int = None):
    if not start:
        start = 0
    if not end:
        end = len(data)
    data = data.rename(columns={data.columns[0]: "Date"})
    data = data.replace(
        r"^\s*$", np.nan, regex=True).iloc[start:end, :].set_index("Date").dropna(axis=1)
    data.index = pd.to_datetime(data.index)
    return data


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    target_return = "002"
    sheet_name = "Bayesian"
    comm = "001"
    comm_fee = int(comm) / 1000

    prices, prices_open, weights = SyntheticData()

    # save price and weights
    # prices.to_csv(f'{dirname}/../data/synthetic_close_prices.csv')
    # prices_open.to_csv(f'{dirname}/../data/synthetic_open_prices.csv')
    # weights.to_csv(f'{dirname}/../data/synthetic_weights.csv')

    # load price and weights data
    # close_prices_df = pd.read_csv(
    #     f'{dirname}/../data/synthetic_close_prices.csv', index_col='Date', parse_dates=True)
    # open_prices_df = pd.read_csv(
    #     f'{dirname}/../data/synthetic_open_prices.csv', index_col='Date', parse_dates=True)
    # weights_df = pd.read_csv(
    #     f'{dirname}/../data/synthetic_weights.csv', index_col='Date', parse_dates=True)

    # close_prices_df = pd.read_csv(f"{dirname}/../data/spx_close_2014_24.csv", index_col="Date", parse_dates=True)
    # open_prices_df = pd.read_csv(f"{dirname}/../data/spx_open_2014_24.csv", index_col="Date", parse_dates=True)
    # weights_df = pd.read_csv(f"{dirname}/../data/long_SpecReturn_002.csv", index_col="Date", parse_dates=True)

    close_prices_df = pd.read_excel(
        f"{dirname}/../data/S&P500 Daily Closing Price 2014-2024.xlsx", sheet_name="S&P500 2014-2024")
    close_prices_df = data_cleaning(close_prices_df)
    open_prices_df = pd.read_excel(
        f"{dirname}/../data/S&P 500 Trading Volume,  Open Price 14-24.xlsx", sheet_name="S&P 500 Opening Price 14-24")
    open_prices_df = data_cleaning(open_prices_df)
    weights_df = pd.read_excel(
        f"{dirname}/../output/long_SpecReturn_{target_return}.xlsx", sheet_name=sheet_name)
    # weights_df = pd.read_excel(
    #     f"{dirname}/../output/long_SpecReturn_0025.xlsx", sheet_name="Bayesian")
    weights_df = data_cleaning(weights_df)

    weights_df = weights_df / \
        weights_df.sum(axis=1).values.reshape(-1, 1) * 0.9

    # Combine open and close prices into one DataFrame
    combined_df = open_prices_df.join(
        close_prices_df, lsuffix="_open", rsuffix="_close")
    combined_df = combined_df.dropna()

    # align the date of price and weights
    combined_df = combined_df.loc[weights_df.index]

    # initialize cerebro engine
    cerebro = bt.Cerebro()

    # read data feeds
    for col in close_prices_df.columns:
        data = PandasData(
            dataname=combined_df[[col + "_open", col + "_close"]])
        cerebro.adddata(data, name=col)

    # strategy setting
    cerebro.broker.setcash(100000000)
    cerebro.broker.setcommission(commission=comm_fee)
    cerebro.broker.set_shortcash(True)
    cerebro.addstrategy(BLStrategy, weights=weights_df,
                        stocks=close_prices_df.columns, printnotify=False, printlog=False)

    # analyze strategy
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    cerebro.addanalyzer(bt.analyzers.TimeReturn,
                        timeframe=bt.TimeFrame.NoTimeFrame, _name="CummulativeReturn")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="AnnualReturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
    # cerebro.addanalyzer(bt.analyzers.Calmar, _name='CalmaraRatio')
    # cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03, annualize=True, _name='SharpeRatio')

    # inital value
    print("Starting Portfolio Value:", cerebro.broker.getvalue())

    # add observer
    for data in cerebro.datas:
        data.plotinfo.plot = False  # Disable plotting of individual stocks
    cerebro.addobserver(PortfolioValueObserver)

    # run the strategy
    results = cerebro.run()
    cerebro.plot()

    # store portfolio returns
    strat = results[0]
    portfolio_stras = strat.analyzers.getbyname("pyfolio")
    returns, positions, transactions, gross_lev = portfolio_stras.get_pf_items()
    print(returns.head())
    returns.to_csv(
        f"{dirname}/../output/returns_{target_return}_{comm}_{sheet_name}.csv")
    # f"{dirname}/../output/returns.csv")

    # performance matrices
    print("Final Portfolio Value:", cerebro.broker.getvalue())
    print("Cummulative Return:", strat.analyzers.CummulativeReturn.get_analysis())
    print("Annual Return:", strat.analyzers.AnnualReturn.get_analysis())
    print("Draw Down:", strat.analyzers.DrawDown.get_analysis())
    # print('PyFolio:', strat.analyzers.pyfolio.get_pf_items())
    # print('Sharpe Ratio:', strat.analyzers.SharpeRatio.get_analysis())
    # print('Calmar Ratio:', strat.analyzers.CalmaraRatio.get_analysis())
