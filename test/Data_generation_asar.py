import pandas as pd
import numpy as np
import os
from numpy import linalg as LA


def generate_asar_data(n, p, rho2, seed, setting_index, save_file=False):
    """
    Input
    - n              : Number of data
    - p              : Number of features
    - rho2           : Degree of correlation
    - seed           : Random seed
    - setting_index  : Number of the experiments
    - save_file      : Whether to save the generated data

    Output
    - X              : Feature
    - y              : Target
    - beta           : True beta
    - record         : The detail information in each dataset (n, p, rho^2, eigenvalue, and correlation matrix)

    Source
    - Yasin Asar. (2017). Some new methods to solve multicollinearity in logistic regression.
    Communications in Statistics - Simulation and Computation, 46:4, 2576-2586. Section 3.1 (page 2581)
    """

    # Set random seeds for reproducibility
    rng = np.random.default_rng(seed)

    # Generate X: n * (p + 1) matrix
    X = np.zeros((n, p + 1))
    for i in range(n):
        z_ip = rng.normal(loc=0, scale=1)
        for j in range(p + 1):
            z_ij = rng.normal(loc=0, scale=1)
            X[i, j] = np.sqrt(1 - rho2) * z_ij + np.sqrt(rho2) * z_ip

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(X, rowvar=False)

    # Generate beta: (p + 1) * 1 coefficient vector
    beta = rng.standard_normal(p + 1)
    # Normalize beta to ensure β'β = 1
    beta = beta / LA.norm(beta)
    # Check if β'β = 1
    beta_norm = LA.norm(beta)
    assert np.isclose(beta_norm, 1), f"β'β is not equal to 1, got {beta_norm}"

    # Generate P:
    P = np.zeros(n)
    P = np.exp(X @ beta) / (1 + np.exp(X @ beta))

    # Generate y:
    y = rng.binomial(1, P)

    # Generate W_hat:
    W_hat = np.diag(P * (1 - P))

    # Calculate XtWhX
    Xt_Wh_X = X.T @ W_hat @ X
    eigenvalues, _ = LA.eigh(Xt_Wh_X)

    # Save the data
    if save_file == True:
        output_dir = f"dataset/Asar/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data = pd.DataFrame(X)
        data["y"] = y
        data.to_csv(f"{output_dir}/Data_{setting_index + 1}.csv", index=False)

    return (
        X,
        y,
        beta,
        {
            "n": n,
            "p": p,
            "rho^2": rho2,
            "eigenvalue": eigenvalues,
            "correlation matrix": correlation_matrix,
        },
    )


"""
# Settings for each dataset in Table 1 to Table 4 (page 2581)
n_values = [50, 100, 200]
p_values = [4, 8]
rho2_values = [0.90, 0.95, 0.99]
base_seed = 426
rng_seed_generator = np.random.default_rng(base_seed)
seeds = rng_seed_generator.integers(low=0, high=50000, size=50000)

# Start simulation
record = []
setting_index = 0
for p in p_values:
    for rho2 in rho2_values:
        for n in n_values:
            current_seed = seeds[setting_index]
            X, y, beta, result = generate_asar_data(
                n,
                p,
                rho2,
                seed=current_seed,
                setting_index=setting_index,
                save_file=True,
            )
            record.append(result)
            setting_index += 1

record_df = pd.DataFrame(record)
record_df.to_excel("dataset/Asar/record.xlsx", index=False)
"""
