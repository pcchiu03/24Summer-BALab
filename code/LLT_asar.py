import os, numpy as np, pandas as pd
from numpy import linalg as LA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)


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

    k = k_estimator(k_type, k_LT1, p)

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
