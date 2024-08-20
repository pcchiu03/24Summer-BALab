import os, sys, time, numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Data_generation_asar import generate_asar_data
from MLE_asar import MLE
from LLT_asar import LLT, calculate_errors, predict, save_to_excel


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
