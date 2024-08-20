import os, time, numpy as np, pandas as pd
from numpy import linalg as LA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from MLE_asar import MLE
from Data_generation_asar import generate_asar_data


def LLT(X, beta_MLE, X_TWX, k_type):
    """
    Input
    - X           : Feature
    - beta_MLE    : Optimal beta received by using Maximum Likelihood Estimation
    - X_TWX       : Marix X^T @ W @ X
    - k_type      : Type of estimator k for Logistic Liu-type Estimation

    Output
    - beta_LLT    : Optimal beta obtained using Logistic Liu-type Estimation

    Source
    - Yasin Asar. (2017). Some new methods to solve multicollinearity in logistic regression.
    Communications in Statistics - Simulation and Computation, 46:4, 2576-2586. Section 2 and 2.2 (page 2578-2579)
    """
    _, p = X.shape
    beta_LLT = np.zeros(p)

    # Eigenvalues and eigenvectors
    lambda_vals, Q = LA.eigh(X_TWX)

    # Compute alpha = Q.T @ beta_MLE (page 2581)
    alpha = Q.T @ beta_MLE
    d_initial = np.min(lambda_vals / (1 + lambda_vals * alpha**2))
    d = np.random.uniform(d_initial / 2, d_initial)
    # d_initial = np.min(lambda_vals / (1 + lambda_vals * alpha**2)) / 6
    # d = d_initial

    iteration = 0
    while not np.all(d < lambda_vals / (1 + lambda_vals * alpha**2)):
        d -= 1e-10
        iteration += 1

    k_LT1 = (lambda_vals - d * (1 + lambda_vals * alpha**2)) / (lambda_vals * alpha**2)
    # k_LT1 = np.clip(k_LT1, 1e-10, np.inf)

    k = k_estimator(k_type, k_LT1, p)
    # k = np.clip(k, 1e-10, np.inf)

    beta_LLT = LA.inv(X_TWX + k * np.eye(p)) @ (X_TWX - d * np.eye(p)) @ beta_MLE

    # print(f"Iteration {iteration} times for {k_type}: d = {d}, k = {k}")

    return beta_LLT


def k_estimator(k_type, k_LT1, p):
    """
    Input
    - k_type     : Type of estimator k for Logistic Liu-type Estimation
    - k_LT1      : Individual parameter k_LT1
    - p          : Number of features

    Output
    - The calculated k value based on the specified k_type
        - 'AM'        : Arithmetic mean of k_LT1
        - 'GM'        : Geometric mean of k_LT1
        - 'MED'       : Median of k_LT1
        - Otherwise   : Raises ValueError for invalid k_type

    - Source
    Yasin Asar. (2017). Some new methods to solve multicollinearity in logistic regression.
    Communications in Statistics - Simulation and Computation, 46:4, 2576-2586. Section 2.2 (page 2579-2580)
    """
    if k_type == "AM":
        return np.mean(k_LT1)
    if k_type == "GM":
        return np.prod(k_LT1) ** (1 / p)
    if k_type == "MED":
        return np.median(k_LT1)
    else:
        raise ValueError("Invalid k_type. Please choose from 'AM', 'GM' or 'MED'")


def calculate_errors(beta_pred, beta_true):
    """
    Input
    - beta_pred   : Predicted beta
    - beta_true   : True beta

    Output
    - MSE   : Mean squared error
    - MAE   : Mean absolute error

    Source
    - Yasin Asar. (2017). Some new methods to solve multicollinearity in logistic regression.
    Communications in Statistics - Simulation and Computation, 46:4, 2576-2586. Section 3.1 (page 2580)
    """
    n = beta_true.shape[0]
    MSE = mean_squared_error(beta_pred, beta_true) * n
    MAE = mean_absolute_error(beta_pred, beta_true) * n

    return MSE, MAE


def predict(X, y, beta):
    """
    Input
    - X       : Feature
    - y       : Target
    - beta    : Caculated beta by MLE or LLT

    Output
    - RMSE    : Root mean squared error between predicted and actual y
    """
    P_pred = np.zeros(X.shape[0])
    P_pred = np.exp(X @ beta) / (1 + np.exp(X @ beta))
    y_pred = np.random.binomial(1, P_pred)

    RMSE = root_mean_squared_error(y, y_pred)

    return RMSE


def save_to_excel(data, filename, columns_per_group=3, transpose=True):
    """
    Input
    - data                : Data to be saved
    - filename            : Name of the Excel file
    - columns_per_group   : Number of columns per group to change the background color
    - transpose           : Whether to transpose the DataFrame before saving

    Output
    - Save the data into an Excel file in .xlsx format
    """
    output_dir = f"output/Asar_LLT/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    df = pd.DataFrame(data)

    if transpose:
        df = df.T
        df = df.reset_index()
        df.columns = ["Data"] + list(df.columns[1:] + 1)

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets["Sheet1"]

        # Format for light gray background
        light_gray_format = writer.book.add_format({"bg_color": "#F0F0F0"})
        num_cols = len(df.columns)
        for start_col in range(1, num_cols, columns_per_group * 2):
            end_col = min(start_col + columns_per_group - 1, num_cols - 1)
            # Apply light gray background to every other block of columns_per_group
            worksheet.conditional_format(
                1,
                start_col,
                len(df),
                end_col,
                {
                    "type": "formula",
                    "criteria": f"MOD(COLUMN()-{start_col+1}, {columns_per_group*2}) < {columns_per_group}",
                    "format": light_gray_format,
                },
            )

        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(str(col)))
            worksheet.set_column(i, i, column_len + 2)

"""
# Settings for each dataset in Table 1 to Table 4 (page 2581)
n_values = [50, 100, 200]
p_values = [4, 8]
rho2_values = [0.90, 0.95, 0.99]
tolerance = 1e-7
num_simulations = 5000


base_seed = 426
rng_seed_generator = np.random.default_rng(base_seed)
seeds = rng_seed_generator.integers(low=0, high=500000, size=500000)


MSE_table, MAE_table, RMSE_table = [], [], []

# Start simulation
star_time = time.time()
setting_index, test_num = 0, 0
for p in p_values:
    for rho2 in rho2_values:
        for n in n_values:
            MSE_results, MAE_results, RMSE_results = (
                np.zeros(4),
                np.zeros(4),
                np.zeros(4),
            )

            for sim in range(num_simulations):
                # print(f"simulation: {sim+1}")
                X, y, beta_true, _ = generate_asar_data(
                    n, p, rho2, seed=seeds[setting_index + sim], setting_index=sim
                )

                beta_MLE, X_TWX = MLE(X, y, tolerance, setting_index)
                beta_LLT_k_AM = LLT(X, beta_MLE, X_TWX, "AM")
                beta_LLT_k_GM = LLT(X, beta_MLE, X_TWX, "GM")
                beta_LLT_k_MED = LLT(X, beta_MLE, X_TWX, "MED")

                beta_tilde = [beta_LLT_k_AM, beta_LLT_k_GM, beta_LLT_k_MED, beta_MLE]

                for i, beta_pred in enumerate(beta_tilde):
                    mse, mae = calculate_errors(beta_pred, beta_true)
                    rmse = predict(X, y, beta_pred)
                    MSE_results[i] += mse
                    MAE_results[i] += mae
                    RMSE_results[i] += rmse

            MSE_results /= num_simulations
            MAE_results /= num_simulations
            RMSE_results /= num_simulations

            print(f"Test {test_num + 1}: Results for n = {n}, p = {p}, rho2 = {rho2}:")

            for metric_name, metric_results in zip(
                ["MSE", "MAE", "RMSE"], [MSE_results, MAE_results, RMSE_results]
            ):
                print(f"\n{metric_name} values of estimators")
                for estimator_name, result in zip(
                    ["k_AM", "k_GM", "k_MED", "MLE"], metric_results
                ):
                    print(f"{estimator_name}: {result}")
            print("-" * 80)

            MSE_table.append(
                {
                    "p": p,
                    "rho^2": rho2,
                    "n": n,
                    "k_AM": MSE_results[0],
                    "k_GM": MSE_results[1],
                    "k_MED": MSE_results[2],
                    "MLE": MSE_results[3],
                }
            )

            MAE_table.append(
                {
                    "p": p,
                    "rho^2": rho2,
                    "n": n,
                    "k_AM": MAE_results[0],
                    "k_GM": MAE_results[1],
                    "k_MED": MAE_results[2],
                    "MLE": MAE_results[3],
                }
            )

            RMSE_table.append(
                {
                    "p": p,
                    "rho^2": rho2,
                    "n": n,
                    "k_AM": RMSE_results[0],
                    "k_GM": RMSE_results[1],
                    "k_MED": RMSE_results[2],
                    "MLE": RMSE_results[3],
                }
            )

            setting_index += num_simulations
            test_num += 1


end_time = time.time()
print(f"execution time: {end_time - star_time} (sec)")


# Save the data
save_to_excel(MSE_table, "MSE_Table.xlsx")
save_to_excel(MAE_table, "MAE_Table.xlsx")
save_to_excel(RMSE_table, "RMSE_Table.xlsx")
"""