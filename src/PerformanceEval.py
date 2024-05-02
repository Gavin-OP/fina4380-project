import os
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if __name__ == "__main__":
    # initialization
    dirname = os.path.dirname(__file__)
    aum0 = 100000

    # this part may need read returns for several periods
    # load returns and benchmark
    returns = pd.read_csv(f"{dirname}/../data/returns.csv", index_col=0)
    returns = returns.squeeze()
    returns.index = pd.to_datetime(returns.index, format='%Y-%m-%d')
    spx_prices = pd.read_excel(
        f'{dirname}/../data/SPX Daily Closing Price 14-24.xlsx', index_col=0)
    spx_prices.index = pd.to_datetime(spx_prices.index, format='%Y-%m-%d')
    port_prices = (1 + returns).cumprod() * aum0

    # quantstats report
    qs.reports.html(
        returns, output=f'{dirname}/../doc/fina4380_backtest_report.html', title='FINA4380 Portfolio')

    # align the date of price and returns
    returns.index = returns.index.to_period('D')
    spx_prices.index = spx_prices.index.to_period('D')
    spx_prices = spx_prices.reindex(returns.index, method='ffill')
    spx_prices.index = spx_prices.index.to_timestamp()
    returns.index = returns.index.to_timestamp()
    spx_prices = spx_prices / spx_prices.iloc[0] * aum0

    # plot the price and benchmark with drawdown
    plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(7, 1)

    # add subplots
    ax1 = plt.subplot(gs[0:5, :])
    ax2 = plt.subplot(gs[5:6, :])

    # portfolio vs benchmark
    ax1.plot(port_prices, label='Portfolio', color='#5aa2d4')
    ax1.plot(spx_prices, label='SPX', color='#c0c0c0')

    # beautify the plot
    ax1.get_xaxis().set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=False, fontsize=12,
               facecolor='none', edgecolor='none', labelcolor='#595959', ncol=2)

    # plot the drawdown of portfolio
    drawdown = qs.stats.to_drawdown_series(port_prices)
    ax2.plot(drawdown, color='#5aa2d4')
    ax2.fill_between(drawdown.index, drawdown, 0, color='#5aa2d4', alpha=0.1)
    ax2.set_title('Drawdown')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.suptitle('Portfolio vs SPX', fontsize=20, y=0.95)
    plt.savefig(f'{dirname}/../img/port_vs_spx.png')
    plt.show()
