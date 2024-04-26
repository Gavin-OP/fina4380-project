import numpy as np
import pandas as pd
from Factor_Prep import Factor_Data
from Bayesian_Posterior import Bayesian_Posteriors
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

if __name__ == "__main__":
    # Get stock returns
    stock_data = pd.read_excel("data/S&P500 Daily Closing Price 2014-2024.xlsx")
    stock_data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    stock_data = stock_data.replace(r"^\s*$", np.nan, regex=True).iloc[:2516, :]
    stock_return = stock_data.set_index("Date").pct_change().iloc[1:, :].dropna(axis=1)

    # Get factor data
    factor_data = Factor_Data("data/10_Industry_Portfolios_Daily.csv", skiprows=9, nrows=25690).factor_data
    common_index = stock_return.index.intersection(factor_data.index)
    stock_return, factor_data = stock_return.loc[common_index, :], factor_data.loc[common_index, :]
    print("Data loading and cleaning done.")

    y = []
    z = []
    g = []
    length = 2000
    stock_slice = 10
    for i in range(length):
        time_period = (i, 252 + i)
        miu, cov_mat, g_star = Bayesian_Posteriors(
            factor_data.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice]
        ).posterior_predictive()
        miu_simple = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].mean()
        cov_mat_simple = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].cov()
        y.append(sum(abs(miu - miu_simple)))
        z.append(sum(sum(abs(cov_mat - cov_mat_simple.values))))
        g.append(g_star)
        print("Loop", i + 1, "done.")

    x = pd.to_datetime(factor_data.index[252 : 252 + length])
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(x, y)
    plt.title("Distance In Mean Estimate", fontdict={"fontweight": "bold"})
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.subplot(3, 1, 2)
    plt.plot(x, z)
    plt.title("Distance In Covariance Estimate", fontdict={"fontweight": "bold"})
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.subplot(3, 1, 3)
    plt.plot(x, g)
    plt.title("Strength Of Shrinkage", fontdict={"fontweight": "bold"})
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig("img/result_compare.png")
    plt.show()
