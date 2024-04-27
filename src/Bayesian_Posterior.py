import pandas as pd
import numpy as np
from Factor_Prep import Factor_Data, PCA
from scipy.optimize import minimize


class Bayesian_Posteriors:
    def __init__(self, factor_data: pd.DataFrame, stock_data: pd.DataFrame, pca: bool = False, explained_variance_ratio: float = 0.9):
        self.factor_data = factor_data.values
        self.stock_data = stock_data.values
        self.T = stock_data.shape[0]  # Number of time periods
        self.M = stock_data.shape[1]  # Number of stocks
        self.explained_variance_ratio = explained_variance_ratio
        if pca:
            # Integrate PCA approach
            self.F = self.factor_data @ PCA(factor_data, explained_variance_ratio=self.explained_variance_ratio).eigenvectors
        else:
            self.F = self.factor_data
        self.K = self.F.shape[1]  # Number of factors
        self.g_star = minimize(self.g_likelihood, 0).x[0]

    # List of mean of sigma^2_m (length: m, m is number of stocks)
    def post_sig2_mean(self, beta_0=None, g=None) -> list[float]:
        if not beta_0:
            beta_0 = np.zeros(self.K)
        if not g:
            g = self.g_star
        sig2_list = []
        for m in range(self.M):
            r_m = self.stock_data[:, m]
            beta_hat_m = np.linalg.inv(self.F.T @ self.F) @ self.F.T @ r_m
            SSR = (r_m - self.F @ beta_hat_m).T @ (r_m - self.F @ beta_hat_m) + 1 / (g + 1) * (beta_hat_m - beta_0).T @ self.F.T @ self.F @ (
                beta_hat_m - beta_0
            )
            sig2_list.append(SSR / 2 / (self.T / 2 - 1))
        return sig2_list

    # List of mean and var of beta_m
    def post_beta(self, beta_0=None, g=None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not beta_0:
            beta_0 = np.zeros(self.K)
        if not g:
            g = self.g_star
        beta_mean_list = []
        beta_var_list = []
        for m in range(self.M):
            r_m = self.stock_data[:, m]
            beta_hat_m = np.linalg.inv(self.F.T @ self.F) @ self.F.T @ r_m
            beta_m_bar = (beta_0 + g * beta_hat_m) / (1 + g)
            beta_mean_list.append(np.array(beta_m_bar))

            SSR = (r_m - self.F @ beta_hat_m).T @ (r_m - self.F @ beta_hat_m) + 1 / (g + 1) * (beta_hat_m - beta_0).T @ self.F.T @ self.F @ (
                beta_hat_m - beta_0
            )
            sig_m = g / (g + 1) * np.linalg.inv(self.F.T @ self.F) * SSR / self.T
            beta_var_list.append(self.T / (self.T - 2) * sig_m)
        return beta_mean_list, beta_var_list

    # Objective function for finding g*
    def g_likelihood(self, g) -> float:
        R_squared_list = []
        for m in range(self.M):
            r_m = self.stock_data[:, m]
            r_m_bar = r_m.mean(axis=0)
            beta_hat_m = np.linalg.inv(self.F.T @ self.F) @ self.F.T @ r_m
            R_squared_m = 1 - ((r_m - self.F @ beta_hat_m).T @ (r_m - self.F @ beta_hat_m)) / ((r_m - r_m_bar).T @ (r_m - r_m_bar))
            R_squared_list.append(R_squared_m)
        R_squared_list = np.array(R_squared_list)
        return sum(-(self.T - self.K - 1) / 2 * np.log(1 + g) + (self.T - 1) / 2 * np.log(1 + g * (1 - R_squared_list)))

    # Mean and var of miu_f
    def post_miu_f(self) -> tuple[np.ndarray, np.ndarray]:
        f_bar = self.F.mean(axis=0)
        Lambda_n = np.zeros((self.K, self.K))
        for t in range(self.T):
            f_t = self.F[t, :]
            Lambda_n += np.outer(f_t - f_bar, f_t - f_bar)
        miu_f_mean = f_bar
        miu_f_var = 1 / (self.T - self.K - 2) * Lambda_n / self.T
        return miu_f_mean, miu_f_var

    # Mean of Lambda_n
    def post_Lambda_n(self) -> np.ndarray:
        f_bar = self.F.mean(axis=0)
        Lambda_n = np.zeros((self.K, self.K))
        for t in range(self.T):
            f_t = self.F[t, :]
            Lambda_n += np.outer(f_t - f_bar, f_t - f_bar)
        return Lambda_n / (self.T - self.K - 2)

    # Posterior predictive return distribution (mean vector and covariance matrix) and shrinkage parameter g*
    def posterior_predictive(self) -> tuple[np.ndarray, np.ndarray, float]:
        sig2_mean = self.post_sig2_mean()
        miu_f_mean, miu_f_var = self.post_miu_f()
        Lambda_n_mean = self.post_Lambda_n()
        beta_mean_list, beta_var_list = self.post_beta()

        f_ft_mean = Lambda_n_mean + miu_f_var + np.outer(miu_f_mean, miu_f_mean)
        f_var = Lambda_n_mean + miu_f_var

        r_mean_list = []
        r_cov_mat = np.zeros((self.M, self.M))
        for m in range(self.M):
            r_mean = beta_mean_list[m] @ miu_f_mean
            r_mean_list.append(r_mean)
            for j in range(m, self.M):
                if m == j:
                    r_cov_mat[m, m] = sig2_mean[m] + np.trace(f_ft_mean @ beta_var_list[m]) + beta_mean_list[m].T @ f_var @ beta_mean_list[m]
                else:
                    r_cov_mat[m, j] = beta_mean_list[m].T @ f_var @ beta_mean_list[j]
                    r_cov_mat[j, m] = r_cov_mat[m, j]
        return np.array(r_mean_list), np.array(r_cov_mat), self.g_star
