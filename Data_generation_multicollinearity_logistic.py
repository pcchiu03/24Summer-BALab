import pandas as pd
import numpy as np
import os, random

from numpy import linalg as LA


def generate_multicollinearity_data(n, p, rho2, seed=426):
    """
    Input:
    n        - Number of data points
    p        - Number of features
    rho2     - Degree of correlation

    Output:
    X         - Feature
    Xt_Wh_X   - Eigenvalues

    Source:
    Yasin Asar. (2017). Some new methods to solve multicollinearity in logistic regression.
    Communications in Statistics - Simulation and Computation, 46:4, 2576-2586.

    Simulation datasets in section 3.1 (page 2581)

    Functions:
    # (1.1):
        Beta_MLE = inv(Xt @ Wh @ X) @ Xt @ Wh @ zh
        where:
            Wh = diag(P_i (1 - P_i))
            zh = log(P_i) + (y_i - P_i) / (P_i (1 - P_i))

    # (3.3):
        X_ij = sqrt(1 - rho^2) * z_ij + rho * z_ip
        where z_ij and z_ip obtain from the standard normal distribution (loc = 0, scale = 1)

    # (3.4):
        P_i = exp(x_i @ beta) / (1 + exp(x_i @ beta))
    """

    # Set random seeds for reproducibility
    rand_for_seeds = np.random.default_rng(seed=426)
    all_seeds = rand_for_seeds.integers(low=0, high=50000, size=50000)

    record = []

    seed_index = 0
    for setting_index_p in range(len(p)):
        for setting_index_rho2 in range(len(rho2)):
            for setting_index_n in range(len(n)):
                rng = np.random.default_rng(seed=all_seeds[seed_index])

                # Generate X: n * (p + 1) matrix
                X = np.zeros((n[setting_index_n], p[setting_index_p] + 1))
                for i in range(n[setting_index_n]):
                    z_ip = rng.normal(loc=0, scale=1)
                    for j in range(p[setting_index_p] + 1):
                        z_ij = rng.normal(loc=0, scale=1)
                        X[i, j] = (
                            np.sqrt(1 - rho2[setting_index_rho2]) * z_ij
                            + np.sqrt(rho2[setting_index_rho2]) * z_ip
                        )

                # Calculate the correlation matrix
                correlation_matrix = np.corrcoef(X, rowvar=False)

                # Generate beta: (p + 1) * 1 coefficient vector
                beta = rng.standard_normal(p[setting_index_p] + 1)
                # Normalize beta to ensure β'β = 1
                beta = beta / LA.norm(beta)
                # Check if β'β = 1
                beta_norm = LA.norm(beta)
                assert np.isclose(
                    beta_norm, 1
                ), f"β'β is not equal to 1, got {beta_norm}"
                # print(f"Setting {setting_index + 1}: β'β = {beta_norm}")

                # Generate P:
                P = np.zeros(n[setting_index_n])
                for i in range(n[setting_index_n]):
                    P[i] = np.exp(X[i] @ beta) / (1 + np.exp(X[i] @ beta))

                # Generate W_hat:
                W_hat = np.diag(P * (1 - P))

                # Calculate XtWhX
                Xt_Wh_X = X.T @ W_hat @ X
                # Here should use 'LA.eigh()' since it's a symmetric matrix, or it may produce imaginary numbers
                eigenvalues, eigenvectors = LA.eigh(Xt_Wh_X)

                """-------------- Show the summary information you want to know --------------"""
                print(
                    f"Setting {seed_index + 1}: n = {n[setting_index_n]}, p = {p[setting_index_p]}, rho^2 = {rho2[setting_index_rho2]}\n"
                )
                print(f"Correlation matrix:")
                print(np.round(correlation_matrix, decimals=2), "\n")

                print(f"Eigenvalues:")
                print(np.round(eigenvalues, decimals=3))
                # print('max = %.3f, min = %.3f\n' % (max(eigenvalues), min(eigenvalues)))
                print("-" * 80)
                """---------------------------------------------------------------------------"""

                # Save the data
                output_dir = (
                    f"dataset/Asar_multicollinearity_Table2/Setting_{seed_index + 1}/"
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                data = pd.DataFrame(X)
                data.to_csv(f"{output_dir}/Data_{seed_index + 1}.csv", index=False)

                record.append(
                    {
                        "n": n[setting_index_n],
                        "p": p[setting_index_p],
                        "rho^2": rho2[setting_index_rho2],
                        "eigenvalue": eigenvalues,
                        "correlation matrix": correlation_matrix,
                    }
                )

                seed_index += 1

    record = pd.DataFrame(record)
    record.to_excel("dataset/Asar_multicollinearity_Table2/record.xlsx", index=False)


# Settings for each dataset in Table.2 and Table.3 (page 2581)
n = [50, 100, 200]
p = [4, 8]
rho2 = [0.90, 0.95, 0.99]

# Start simulation
generate_multicollinearity_data(n, p, rho2, seed=426)
