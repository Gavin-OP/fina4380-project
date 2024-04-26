import pandas as pd
import numpy as np


class Factor_Data:
    def __init__(self, factor_data_path: str, non_return_list: list = None, **kwargs):
        self.factor_data_path = factor_data_path
        self.non_return_list = non_return_list
        self.kwargs = kwargs
        self.factor_data = self.load_data()

    def load_data(self):
        data: pd.DataFrame = pd.read_csv(self.factor_data_path, **self.kwargs)
        data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
        data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
        data.set_index("Date", inplace=True)
        if self.non_return_list:
            for non_return in self.non_return_list:
                data[non_return] = data[non_return].pct_change()
        return data.dropna()


class PCA:
    def __init__(self, factor_data: pd.DataFrame, n_components: int = None, explained_variance_ratio: float = None):
        self.factor_data = factor_data
        self.cov_matrix = self.factor_data.cov()
        self.eigenvalues, self.eigenvectors = self.fit_pca()
        if n_components:
            self.eigenvalues = self.eigenvalues[:n_components]
            self.eigenvectors = self.eigenvectors[:, :n_components]
        elif explained_variance_ratio:
            total_variance = sum(self.eigenvalues)
            single_explained_variance_ratio = self.eigenvalues / total_variance
            cum_explained_variance_ratio = np.cumsum(single_explained_variance_ratio)
            n_components = np.argmax(cum_explained_variance_ratio >= explained_variance_ratio) + 1
            self.eigenvalues = self.eigenvalues[:n_components]
            self.eigenvectors = self.eigenvectors[:, :n_components]

    def fit_pca(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors
