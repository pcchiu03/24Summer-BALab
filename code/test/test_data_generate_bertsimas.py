import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Data_generation_bertsimas import generate_data_bertsimas


# Simulation settings in Section 4.2 (page 373)
n = [100, 1000, 2000]
p = [10, 100, 200]
rho = [0.4, 0.8]
k = 5
sigma = 2

# Start simulation
generate_data_bertsimas(n, p, rho, k, sigma)