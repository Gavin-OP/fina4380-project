import os
import numpy as np
import pandas as pd
from Factor_Prep import Factor_Data
from Bayesian_Posterior import Bayesian_Posteriors
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from Weight_Calc import Weight_Calc
from typing import Literal
from datetime import datetime, timedelta
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


def process_date(start_date: datetime | str, end_date: datetime | str, stock_return: pd.DataFrame, sample_size: int):
    if start_date:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        while True:
            try:
                start = stock_return.index.get_loc(str(start_date.date())) - sample_size
                start = max(start, 0)
                break
            except KeyError:
                print(f"Data of start date {str(start_date.date())} is not available, adding one day.")
                start_date = start_date + timedelta(days=1)
    else:
        start = 0
    if end_date:
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        while True:
            try:
                end = stock_return.index.get_loc(str(end_date.date())) - sample_size
                end = min(end, len(stock_return) - sample_size)
                break
            except KeyError:
                print(f"Data of end date {str(end_date.date())} is not available, subtracting one day.")
                end_date = end_date - timedelta(days=1)
    else:
        end = len(stock_return) - sample_size
    return start, end


def tracking_diff(
    stock_return: pd.DataFrame,
    factor_return: pd.DataFrame,
    plot_name: str = "diff_tracking.png",
    stock_slice: int = 1,
    start_date: datetime | str = None,
    end_date: datetime | str = None,
    sample_size=251,
):
    start, end = process_date(start_date, end_date, stock_return, sample_size)
    y, z, g = [], [], []
    for i in range(start, end):
        time_period = (i, sample_size + i)
        miu, cov_mat, g_star = Bayesian_Posteriors(
            factor_return.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice]
        ).posterior_predictive()
        miu_sample = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].mean()
        cov_mat_sample = stock_return.iloc[time_period[0] : time_period[1], ::stock_slice].cov()
        y.append(sum(abs(miu - miu_sample)))
        z.append(sum(sum(abs(cov_mat - cov_mat_sample.values))))
        g.append(g_star)
        print(str(stock_return.index[time_period[1]]), "finished.")

    x = pd.to_datetime(factor_return.index[sample_size + start : sample_size + end])
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
    stock_data: pd.DataFrame,
    factor_return: pd.DataFrame,
    smart_scheme: Literal["EW", "RP", "MDR", "GMV", "MSR"],
    equal_weight: bool = False,
    mv_weight: bool = False,
    plot_name: str = "return_compare.png",
    stock_slice: int = 1,
    start_date: str = None,
    end_date: str = None,
    sample_size=251,
):
    start, end = process_date(start_date, end_date, stock_return, sample_size)
    return_series, return_series_pca, return_series_sample = [], [], []
    return_series_ew, return_series_mv = [], []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="red bold", finished_style="green bold"),
        TaskProgressColumn("[blue bold]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(compact=True),
        transient=True,
    ) as progress:
        task = progress.add_task("Running", total=end - start)
        for i in range(start, end):
            time_period = (i, sample_size + i)
            # Parameters estimated via Bayesian approach (Non-PCA)
            miu, cov_mat, _ = Bayesian_Posteriors(
                factor_return.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice]
            ).posterior_predictive()
            beta = Weight_Calc(smart_scheme, miu, cov_mat).retrieve_beta()
            return_series.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta)

            # Parameters estimated via Bayesian approach (PCA)
            miu_pca, cov_mat_pca, _ = Bayesian_Posteriors(
                factor_return.iloc[time_period[0] : time_period[1], :], stock_return.iloc[time_period[0] : time_period[1], ::stock_slice], pca=True
            ).posterior_predictive()
            beta_pca = Weight_Calc(smart_scheme, miu_pca, cov_mat_pca).retrieve_beta()
            return_series_pca.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_pca)

            # Parameters estimated via Bayesian approach and views (assume future factor returns are known)
            # if i < end - 1:
            #     miu_view, cov_mat_view, _ = Bayesian_Posteriors(
            #         factor_data.iloc[time_period[0] : time_period[1], :],
            #         stock_return.iloc[time_period[0] : time_period[1], ::stock_slice],
            #         P="absolute",
            #         Q=np.array(factor_data.values[time_period[1], :]),
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
            beta_sample = Weight_Calc(smart_scheme, miu_sample, cov_mat_sample).retrieve_beta()
            return_series_sample.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_sample)

            # Equal weight allocation
            if equal_weight:
                N = stock_return.iloc[::stock_slice].shape[1]
                beta_ew = np.ones(N) / N
                return_series_ew.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_ew)

            # Market value weight allocation
            if mv_weight:
                beta_mv = np.array(1 / stock_data.iloc[time_period[1] - 1, :])
                return_series_mv.append(stock_return.iloc[time_period[1], ::stock_slice] @ beta_mv)

            print(str(stock_return.index[time_period[1]]), "finished.")
            progress.update(task, advance=1)

    # Calculate cumulative return
    for i in range(1, len(return_series)):
        return_series[i] = (1 + return_series[i - 1]) * (1 + return_series[i]) - 1
        return_series_pca[i] = (1 + return_series_pca[i - 1]) * (1 + return_series_pca[i]) - 1
        return_series_sample[i] = (1 + return_series_sample[i - 1]) * (1 + return_series_sample[i]) - 1
        if equal_weight:
            return_series_ew[i] = (1 + return_series_ew[i - 1]) * (1 + return_series_ew[i]) - 1
        if mv_weight is not None:
            return_series_mv[i] = (1 + return_series_mv[i - 1]) * (1 + return_series_mv[i]) - 1

    x = pd.to_datetime(factor_return.index[sample_size + start : sample_size + end])
    plt.figure(figsize=(10, 4))
    plt.plot(x, return_series, label="Bayesian")
    plt.plot(x, return_series_pca, label="Bayesian (PCA)")
    plt.plot(x, return_series_sample, label="Sample")
    if equal_weight:
        plt.plot(x, return_series_ew, label="Equal Weight")
    if mv_weight is not None:
        plt.plot(x, return_series_mv, label="Market Value Weight")
    plt.title(f"Cumulative Return ({smart_scheme})", fontdict={"fontweight": "bold"})
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.legend()
    plt.savefig(os.path.join("img", plot_name))
    plt.show()


def compare_efficient_fronter(
    stock_return: pd.DataFrame, factor_return: pd.DataFrame, date: str | datetime, stock_slice: int = 1, sample_size: int = 251
) -> None:
    date_index, _ = process_date(date, None, stock_return, sample_size)
    miu, cov_mat, _ = Bayesian_Posteriors(
        factor_return.iloc[date_index : date_index + sample_size, :], stock_return.iloc[date_index : date_index + sample_size, ::stock_slice]
    ).posterior_predictive()

    # Bayesian approach
    R = miu
    Sigma = cov_mat
    One = np.ones(len(R))

    A = R.T @ np.linalg.inv(Sigma) @ R
    B = R.T @ np.linalg.inv(Sigma) @ One
    C = One.T @ np.linalg.inv(Sigma) @ One

    mu_p_list = np.arange(1e-3, 1e-3, 1e-5)
    alpha_list = []
    sd_list = []
    for mu_p in mu_p_list:
        lambda_ = (C * mu_p - B) / (A * C - B**2)
        gamma = (A - B * mu_p) / (A * C - B**2)
        alpha = lambda_ * np.linalg.inv(Sigma) @ R + gamma * np.linalg.inv(Sigma) @ One
        alpha_list.append(alpha)
        sd = np.sqrt(alpha.T @ Sigma @ alpha)
        sd_list.append(sd)
    plt.plot(sd_list, mu_p_list, label="Bayesian")
    plt.xlabel("$\sigma_p$")
    plt.ylabel("$\mu_p$")
    plt.title("Efficient Frontier", fontsize=10)

    # Sample approach
    R = stock_return.iloc[date_index : date_index + sample_size, ::stock_slice].mean()
    Sigma = stock_return.iloc[date_index : date_index + sample_size, ::stock_slice].cov()
    One = np.ones(len(R))

    A = R.T @ np.linalg.inv(Sigma) @ R
    B = R.T @ np.linalg.inv(Sigma) @ One
    C = One.T @ np.linalg.inv(Sigma) @ One

    mu_p_list = np.arange(1e-3, 1e-3, 1e-5)
    alpha_list = []
    sd_list = []
    for mu_p in mu_p_list:
        lambda_ = (C * mu_p - B) / (A * C - B**2)
        gamma = (A - B * mu_p) / (A * C - B**2)
        alpha = lambda_ * np.linalg.inv(Sigma) @ R + gamma * np.linalg.inv(Sigma) @ One
        alpha_list.append(alpha)
        sd = np.sqrt(alpha.T @ Sigma @ alpha)
        sd_list.append(sd)
    plt.plot(sd_list, mu_p_list, label="Sample")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # Get stock universe returns
    # stock_data = pd.read_excel(os.path.join(base_dir, "data/S&P500 Daily Closing Price 2014-2024.xlsx"))
    # Get selected stock returns
    stock_data = pd.read_excel(
        os.path.join(base_dir, "data/Selected Stock Daily Closing Price 2014-2024.xlsx"), sheet_name="Selected Stock 2014-2024"
    )
    stock_data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    stock_data = stock_data.replace(r"^\s*$", np.nan, regex=True).iloc[:2516, :].set_index("Date")
    stock_return = stock_data.pct_change().iloc[1:, :].dropna(axis=1)

    # Get factor data and clean data
    factor_return = Factor_Data(os.path.join(base_dir, "data/10_Industry_Portfolios_Daily.csv"), skiprows=9, nrows=25690).factor_data
    common_index = stock_return.index.intersection(factor_return.index)
    stock_return, factor_return, stock_data = stock_return.loc[common_index, :], factor_return.loc[common_index, :], stock_data.loc[common_index, :]
    print("Data loading and cleaning finished.")

    # tracking_diff(stock_return, factor_data, stock_slice=10)
    # return_compare(
    #     stock_return,
    #     stock_data,
    #     factor_return,
    #     smart_scheme="MDR",
    #     plot_name="return_compare_selected_MDR.png",
    #     equal_weight=True,
    #     mv_weight=True,
    # )
    compare_efficient_fronter(stock_return, factor_return, "2023-10-10")
