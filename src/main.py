import os
import numpy as np
import pandas as pd
from Factor_Prep import Factor_Data
from Bayesian_Posterior import Bayesian_Posteriors
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def plot_results(
    stock_return: pd.DataFrame, factor_data: pd.DataFrame, stock_slice: int, length=None, sample_size=251, rebalance_freq: int = 1, jump: int = 0
):
    if not length:
        length = (len(stock_return) - sample_size) // rebalance_freq
    y, z, g = [], [], []
    for i in range(jump, length):
        time_period = (i * rebalance_freq, sample_size + i * rebalance_freq)
        miu, cov_mat, g_star = Bayesian_Posteriors(
            factor_data.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice]
        ).posterior_predictive()
        miu_sample = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].mean()
        cov_mat_sample = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].cov()
        y.append(sum(abs(miu - miu_sample)))
        z.append(sum(sum(abs(cov_mat - cov_mat_sample.values))))
        g.append(g_star)
        print("Loop", i + 1, "done.")

    x = pd.to_datetime(factor_data.index[sample_size + jump : sample_size + length * rebalance_freq])
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


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Get stock returns
    stock_data = pd.read_excel(os.path.join(base_dir, "data/S&P500 Daily Closing Price 2014-2024.xlsx"))
    stock_data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    stock_data = stock_data.replace(r"^\s*$", np.nan, regex=True).iloc[:2516, :]
    stock_return = stock_data.set_index("Date").pct_change().iloc[1:, :].dropna(axis=1)

    # Get factor data and clean data
    factor_data = Factor_Data(os.path.join(base_dir, "data/10_Industry_Portfolios_Daily.csv"), skiprows=9, nrows=25690).factor_data
    common_index = stock_return.index.intersection(factor_data.index)
    stock_return, factor_data = stock_return.loc[common_index, :], factor_data.loc[common_index, :]
    print("Data loading and cleaning finished.")

    plot_results(stock_return, factor_data, stock_slice=10)
