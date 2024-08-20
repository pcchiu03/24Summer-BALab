import os
import numpy as np
import pandas as pd
from numpy import linalg as LA
from Data_generation_asar import generate_asar_data


def MLE(X, y, tol, setting_index, save_file=False):
    """
    Input
    - X               : Feature
    - y               : Target
    - tol             : Tolerance for convergence
    - setting_index   : Number of the experiments
    - save_file       : Whether to save the computation result of MLE

    Output
    - beta_MLE        : The optimal beta from maximum likelihood esitmator (MLE)
    - X_TWX           : The marix X^T @ W @ X

    Source
    - Murphy-Probabilistic Machine Learning-book1
    Page 346, Algorithm 10.1, Iteratively reweighted least squares (IRLS)
    """
    n, p = X.shape
    beta = np.zeros(p)
    converge = np.inf

    eigh_val_rec, determinant_rec = [], []

    iteration = 0
    while converge > tol:
        old_beta = np.copy(beta)
        a = X @ beta
        P = 1 / (1 + np.exp(-a))

        if np.any(P <= 0) or np.any(P >= 1):
            # print(f"Iteration {iteration}: P contains invalid values, break iteration.")
            break

        z = np.log(P) + (y - P) / (P * (1 - P))
        W = np.diag(P * (1 - P))

        X_TWX = X.T @ W @ X
        beta = LA.inv(X_TWX) @ X.T @ W @ z
        converge = LA.norm(old_beta - beta)

        eigenvalues, _ = LA.eigh(X_TWX)
        determinant = LA.det(X_TWX)
        eigh_val_rec.append(eigenvalues)
        determinant_rec.append(determinant)

        iteration += 1

        # Print debugging information
        # print(
        #    f"Iteration {iteration}: \nConvergence = {converge}, \nBeta_MLE = {beta}\n"
        # )

    beta_MLE = beta

    if save_file == True:
        data = {
            "Iteration": [i for i in range(iteration)],
            "Eigenvalues": eigh_val_rec,
            "Determinant": determinant_rec,
        }
        df = pd.DataFrame(data)

        # Save the information data
        output_dir = f"output/Asar_iteration_info/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        filename = f"info_{setting_index + 1}.xlsx"
        filepath = os.path.join(output_dir, filename)

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)

            # Adjust column widths
            worksheet = writer.sheets["Sheet1"]
            for i, col in enumerate(df.columns):
                column_len = max(df[col].astype(str).map(len).max(), len(col))
                worksheet.set_column(i, i, column_len + 2)

    return beta_MLE, X_TWX


"""
# Settings for each dataset in Table 1 to Table 4 (page 2581)
n_values = [50, 100, 200]
p_values = [4, 8]
rho2_values = [0.90, 0.95, 0.99]
tolerance = 1e-7

base_seed = 426
rng_seed_generator = np.random.default_rng(base_seed)
seeds = rng_seed_generator.integers(low=0, high=50000, size=50000)

# Start simulation
setting_index = 0
for p in p_values:
    for rho2 in rho2_values:
        for n in n_values:
            current_seed = seeds[setting_index]
            X, y, beta_true, _ = generate_asar_data(
                n, p, rho2, seed=current_seed, setting_index=setting_index
            )
            print(f"\ndata {setting_index + 1}: {n}/{p}/{rho2}:")

            # correlation_matrix = np.corrcoef(X, rowvar=False)
            # print(f"\nCorrelation matrix:")
            # print(np.round(correlation_matrix, decimals=2), "\n")

            beta_MLE, X_TWX = MLE(X, y, tolerance, setting_index, save_file=True)

            setting_index += 1

            print(f"beta true: {beta_true}")
            print(f"beta MLE: {beta_MLE}")
            print("-" * 80)

            # Save the data
            data = {
                "beta true": beta_true,
                "beta MLE": beta_MLE,
            }
            df = pd.DataFrame(data)

            output_dir = f"output/Asar_MLE/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            filename = f"MLE_{setting_index}.xlsx"
            filepath = os.path.join(output_dir, filename)

            with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False)

                # Adjust column widths
                worksheet = writer.sheets["Sheet1"]
                for i, col in enumerate(df.columns):
                    column_len = max(df[col].astype(str).map(len).max(), len(col))
                    worksheet.set_column(i, i, column_len + 2)
"""
