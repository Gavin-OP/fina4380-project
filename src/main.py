import os
import numpy as np
import pandas as pd
from Factor_Prep import Factor_Data
from Bayesian_Posterior import Bayesian_Posteriors
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from Weight_Calc import Weight_Calc
from typing import Literal


def diff_tracking(
    stock_return: pd.DataFrame,
    factor_data: pd.DataFrame,
    plot_name: str = "diff_tracking.png",
    stock_slice: int = 1,
    length=None,
    sample_size=251,
    rebalance_freq: int = 1,
    jump: int = 0,
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
    plt.savefig(os.path.join("img", plot_name))
    plt.show()


def return_compare(
    stock_return: pd.DataFrame,
    factor_data: pd.DataFrame,
    smartScheme: Literal["EW", "RP", "MDR", "GMV", "MSR"],
    plot_name: str = "return_compare.png",
    stock_slice: int = 1,
    start: int = 0,
    end: int = None,
    sample_size=251,
    rebalance_freq: int = 1,
):
    if not end:
        end = (len(stock_return) - sample_size) // rebalance_freq
    return_series, return_series_pca, return_series_sample = [], [], []
    for i in range(start, end):
        time_period = (i * rebalance_freq, sample_size + i * rebalance_freq)
        # Parameters estimated via Bayesian approach (Non-PCA)
        miu, cov_mat, _ = Bayesian_Posteriors(
            factor_data.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice]
        ).posterior_predictive()
        beta = Weight_Calc(smartScheme, miu, cov_mat).retrieve_beta()
        return_series.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta)
        # Parameters estimated via Bayesian approach (PCA)
        miu_pca, cov_mat_pca, _ = Bayesian_Posteriors(
            factor_data.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice], pca=True
        ).posterior_predictive()
        beta_pca = Weight_Calc(smartScheme, miu_pca, cov_mat_pca).retrieve_beta()
        return_series_pca.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_pca)
        # Parameters estimated via Bayesian approach and views (assume future factor returns are known)
        # if i < end - 1:
        #     miu_view, cov_mat_view, _ = Bayesian_Posteriors(
        #         factor_data.iloc[time_period[0] : time_period[1], :],
        #         stock_return.iloc[time_period[0] : time_period[1], ::stock_slice],
        #         P="absolute",
        #         Q=np.array(factor_data.values[time_period[1] : time_period[1] + 1, :][0]),
        #     ).posterior_predictive()
        # else:
        #     miu_view, cov_mat_view, _ = Bayesian_Posteriors(
        #         factor_data.iloc[time_period[0] : time_period[1], :],
        #         stock_return.iloc[time_period[0] : time_period[1], ::stock_slice],
        #     ).posterior_predictive()
        # beta_view = WeightCalc(smartScheme, miu_view, cov_mat_view).retrieve_beta()
        # return_series_view.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_view)
        # Parameters estimated via sample data
        miu_sample = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].mean()
        cov_mat_sample = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].cov()
        beta_sample = Weight_Calc(smartScheme, miu_sample, cov_mat_sample).retrieve_beta()
        return_series_sample.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_sample)
        print("Loop", i + 1, "done.")

    # Calculate cumulative return
    for i in range(1, len(return_series)):
        return_series[i] = (1 + return_series[i - 1]) * (1 + return_series[i]) - 1
        return_series_pca[i] = (1 + return_series_pca[i - 1]) * (1 + return_series_pca[i]) - 1
        return_series_sample[i] = (1 + return_series_sample[i - 1]) * (1 + return_series_sample[i]) - 1

    x = pd.to_datetime(factor_data.index[sample_size + start : sample_size + end * rebalance_freq])
    plt.figure(figsize=(10, 4))
    plt.plot(x, return_series, label="Bayesian")
    plt.plot(x, return_series_pca, label="Bayesian (PCA)")
    plt.plot(x, return_series_sample, label="Sample")
    plt.title("Cumulative Return", fontdict={"fontweight": "bold"})
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.legend()
    plt.savefig(os.path.join("img", plot_name))
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Get stock returns
    # stock_data = pd.read_excel(os.path.join(base_dir, "data/S&P500 Daily Closing Price 2014-2024.xlsx"))
    # Get selected stock returns
    stock_data = pd.read_excel(
        os.path.join(base_dir, "data/Selected Stock Daily Closing Price 2014-2024.xlsx"), sheet_name="Selected Stock 2014-2024"
    )
    stock_data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    stock_data = stock_data.replace(r"^\s*$", np.nan, regex=True).iloc[:2516, :]
    stock_return = stock_data.set_index("Date").pct_change().iloc[1:, :].dropna(axis=1)

    # Get factor data and clean data
    factor_data = Factor_Data(os.path.join(base_dir, "data/10_Industry_Portfolios_Daily.csv"), skiprows=9, nrows=25690).factor_data
    common_index = stock_return.index.intersection(factor_data.index)
    stock_return, factor_data = stock_return.loc[common_index, :], factor_data.loc[common_index, :]
    print("Data loading and cleaning finished.")

    # plot_results(stock_return, factor_data, stock_slice=10)
    return_compare(stock_return, factor_data, smartScheme="MDR", plot_name="return_compare_selected_MDR.png")
