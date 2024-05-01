# A Comprehensive Approach to Construct a Portfolio: Factor Model, Bayesian Shrinkage and Smart Beta

## Integration with Bayesian Shrinkage on Factors

### Model Specification

With the factors specified above, each stock's return $r_m\text{, } m \in (1, M)$, is modeled by the below equations:

$$
r_m = F\beta_m + \epsilon_m \\
\epsilon_m \sim \mathcal{N}(0, \sigma^2_m\mathbb{I}_T) \\
f_t \sim \mathcal{N}(\mu_f, \Omega_f)
$$

where $r_m$ is a row vector that represents the return time series of stock $m$ spanning in time $T$, $F = [f_1, \cdots, f_t]^T$ is a $T \times K$ matrix that represents the $K$ factors return time series spanning in time $T$, $\beta_m$ is a $K \times 1$ row vector that represents the factor loadings.

We are aiming to model the Bayesian posterior predictive moments $\text{E}(r_m)$ and $\text{Cov}(r_i, r_j)$, where $m, i, j \in (0, M)$. This will be the input for our Smart Beta (stock weights) calculation.

### Prior Distributions

To maintain closed-form solutions in MV analysis, we adopted fully conjugate and well established priors: **Zellner’s g-prior** for $\beta_m$ and **Normal-Inverse-Wishart prior (Jeffrey’s priors)** for $\sigma_m^2$ and $(\mu_f, \Omega_f)$.

$$
\beta\mid\sigma_m^2 \sim \mathcal{N}(\beta_{m, 0}, g\sigma_m^2(F^TF)^{-1}) \\
p(\sigma_m^2) \propto \frac{1}{\sigma_m^2} \\
p(\mu_f, \Omega_f) \propto |\Omega_f|^{-\frac{K+1}{2}}
$$

Here we propose $\beta_{m, 0} = \overrightarrow{0}$ to ridge regression, because it benefits estimation by striking a balance between bias and variance. $g$ emerges as a measure of shrinkage intensity. The smaller value of $g$, the stronger shrinkage towards the prior mean $\beta_{m, 0}$. This hyperparameter ($g^*$) will be optimized in the below section. The priors for $\sigma_m^2$ and $(\mu_f, \Omega_f)$ are essentially uninformative, so we "let the data speak for itself".

### Posterior Distributions

The marginal posterior of $\sigma_m^2$ and $\beta_m$ under the set of prior assumptions is:

$$
\sigma_m^2\mid\mathcal{F}  \sim \text{Inverse-Gamma }(\frac{T}{2}, \frac{SSR_{g, m}}{2}) \\
\beta_m\mid\mathcal{F} \sim \text{Multivariate t }(T, \overline{\beta_m}, \Sigma_m)
$$

where

$$
SSR_{g, m}=(r_m - F\hat{\beta}_m)^T (r_m - F\hat{\beta}_m) + \frac{1}{g+1}(\hat{\beta}_m - \beta_{m, 0})^T F^T F(\hat{\beta}_m - \beta_{m, 0}) \\
\overline{\beta_m}=\frac{1}{g+1} \beta_{m, 0} + \frac{g}{g+1} \hat{\beta}_m \\
\hat{\beta}_m=(F^T F)^{-1} F^T r_m \\
\Sigma_m=\frac{g}{g+1}(F^T F)^{-1} \frac{SSR_{g, m}}{T}
$$

The marginal posterior of $\mu_f$ and $\Omega_f$ under the set of prior assumptions is:

$$
\mu_f\mid\mathcal{F} \sim \text{Multivariate t }(T-K, \bar{f}, \frac{\Omega_n}{T(T-K)}) \\
\Omega_f\mid\mathcal{F} \sim \text{Inverse-Wishart }(T − 1, \Omega_n)
$$

where

$$
\Omega_n = \sum^T_{t=1}(f_t - \bar{f})(f_t - \bar{f})^T \\
\bar{f} = \frac{1}{T}\sum^T_{t=1}f_t
$$

### Determining Shrinkage Intensity $g^*$

For Zellner’s g-prior with $\beta_{m,0} = \overrightarrow{0}$, the marginal likelihood $p(r_m\mid g)$ has a known explicit form:

$$
p(r_m \mid g)=\Gamma(\frac{T-1}{2}) \pi^{-\frac{T-1}{2}} T^{-\frac{1}{2}}\|r_m-\overline{r_m}\|^{-(T-1)} \frac{(1+g)^{(T-K-1) / 2}}{(1+g(1-R^2))^{(T-1) / 2}}
$$

where $R = 1 - \frac{(r_m - F\hat{\beta_m})^T (r_m - F\hat{\beta_m})}{(r_m - \overline{r_m})^T (r_m - \overline{r_m})}$ is the coefficient of determination.

Then we employ the empirical Bayes estimate $g^∗$, which maximizes the marginal (log) likelihood:

$$
\begin{align*}
g^∗ &= \arg\max_g \prod_{m=1}^m p(r_m \mid g) \\
    &= \arg\max_g \prod_{m=1}^m \ln p(r_m \mid g) \\
    &= \arg\min_g \sum_{m=1}^M[-\frac{T-K-1}{2} \ln(1+g)+\frac{T-1}{2} \ln(1+g(1-R^2))]
\end{align*}
$$

### Determining Posterior Predictive Moments of $r_m$

Denote $\operatorname{E}[\cdot\mid\mathcal{F}] = \operatorname{E}[\cdot], \operatorname{Var}[\cdot\mid\mathcal{F}] = \operatorname{Var}[\cdot], \operatorname{Cov}[\cdot\mid\mathcal{F}] = \operatorname{Cov}[\cdot]$, then the posterior predictive moments of stock returns under the Bayesian factor model are:

$$
\operatorname{E}[r_m]=\operatorname{E}[\beta_m]^T \operatorname{E}[\mu_f] \\
\operatorname{Var}[r_m]=\operatorname{E}[\sigma_m^2]+\operatorname{Tr}(\operatorname{E}[f f^T] \operatorname{Var}[\beta_m])+\operatorname{E}[\beta_m]^T \operatorname{Var}[f] \operatorname{E}[\beta_m] \\
\operatorname{Cov}(r_i, r_j)=\operatorname{E}[\beta_i]^T \operatorname{Var}[f] \operatorname{E}[\beta_j]
$$

where

$$
\operatorname{E}[f f^T]=\operatorname{E}[\Omega_f]+\operatorname{Var}[\mu_f]+\operatorname{E}[\mu_f] \mathbb{E}[\mu_f]^T \\
\operatorname{Var}[f]=\operatorname{E}[\Omega_f]+\operatorname{Var}[\mu_f]
$$

with $\operatorname{E}[\mu_f], \operatorname{Var}[\mu_f], \operatorname{E}[\sigma_m^2], \operatorname{E}[\beta_m], \operatorname{Var}[\beta_m], \operatorname{E}[\Omega_f]$ obtained from the posterior distributions after Bayesian updates mentioned above.
