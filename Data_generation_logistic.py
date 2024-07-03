import pandas as pd
import numpy as np
import os, random
from numpy import linalg as LA


def generate_logistic_data(n, p, rho, k, sigma, seed=426):
    """
    Input:
    n       - Number of data points
    p       - Number of features
    rho     - Pairwise correlation
    k       - Best k-feature
    sigma   - Covariance between the data points

    Output:
    X       - Feature
    y       - Target

    Source:
    Dimitris Bertsimas and Angela King. (2017). Logistic Regression: From Art to Science,
    Statistical Science, Vol. 32, No.3, 367-384.

    Simulation datasets in Section 4.2 (page 373)
    """

    # Set random seeds for reproducibility
    rand_for_seeds = np.random.default_rng(seed)
    all_seeds = rand_for_seeds.integers(low=0, high=50000, size=50000)

    record = []

    seed_index = 0
    for setting_index_rho in rho:
        for setting_index_n, setting_index_p in zip(n, p):
            rng = np.random.default_rng(seed=all_seeds[seed_index])

            # Generate Sigma_ij: covariance matrix
            pair_sigma = setting_index_rho ** np.abs(
                np.subtract.outer(
                    np.arange(setting_index_p), np.arange(setting_index_p)
                )
            )

            # Generate X: n * p matrix from multivariate normal distribution
            X = rng.multivariate_normal(
                mean=np.zeros(setting_index_p), cov=pair_sigma, size=setting_index_n
            )

            # Standardize X
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            correlation_matrix = np.corrcoef(X, rowvar=False)

            # Generate beta: with k non-zero entries
            beta = np.zeros(setting_index_p)
            non_zero_indices = np.arange(0, setting_index_p, setting_index_p // k)
            beta[non_zero_indices] = 1

            # Generate y
            epsilon = rng.normal(loc=0, scale=sigma**2, size=setting_index_n)
            linear_combination = X @ beta + epsilon
            y = np.round(1 / (1 + np.exp(-linear_combination)))

            # Save the data
            output_dir = f"dataset/Bertsimas_logistic/Setting_{seed_index + 1}/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            data = pd.DataFrame(X)
            data.to_csv(f"{output_dir}/Data_{seed_index + 1}.csv", index=False)

            record.append(
                {
                    "n": setting_index_n,
                    "p": setting_index_p,
                    "rho": setting_index_rho,
                    "correlation matrix": correlation_matrix,
                }
            )

            """-------------- Show the summary information you want to know --------------"""
            print(
                f"Setting {seed_index + 1}: n = {setting_index_n}, p = {setting_index_p}, rho = {setting_index_rho}\n"
            )
            print(f"Correlation matrix:")
            print(np.round(correlation_matrix, decimals=2), "\n")
            print("-" * 80)
            """---------------------------------------------------------------------------"""

            seed_index += 1

    record = pd.DataFrame(record)
    record.to_excel(f"dataset/Bertsimas_logistic/record.xlsx", index=False)


# Simulation settings in Section 4.2 (page 373)
n = [100, 1000, 2000]
p = [10, 100, 200]
rho = [0.4, 0.8]
k = 5
sigma = 2

# Start simulation
generate_logistic_data(n, p, rho, k, sigma)
