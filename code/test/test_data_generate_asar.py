import os, sys, numpy as np, pandas as pd
from Data_generation_asar import generate_asar_data
from MLE_asar import MLE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
