# 2024-Summer-BAlab_intern

## Contents
This project aims to deal with the multicollinearity in logistic regression. All the simulation setting is based on Asar (2016) or Bertsimas and King (2017).

## Files
```
|- code
|  |- test/                    # Unit test for the code and monte-carlo simulation
|  |- Data_generation_asar     # Function to create the datasets based on Asar (2016)
|  |- MLE_asar.py              # Function to compute MLE using the IRLS algorithm
|  |- LLT_asar.py              # Function to compute LLT with various parameter
|  |- Plot_result_asar.ipynb   # Visualization of analysis results
|  |- Symbolic_asar.ipynb      # Symbolic calculations related to Asar's methods
|- dataset                     # Contains two simulation datasets retrieved from the paper and five real datasets
|- doc/                        # Reports on the results
|- fig/                        # Visualizations of results from the data in the output
|- output/                     # Stores all results computed from the 'code'
|- README.md                   # Overview of the project and instructions
```

## Reference
Asar, Y. (2016). Some new methods to solve multicollinearity in logistic regression. Communications in Statistics-Simulation and Computation, 46(4), 2576–2586. https://doi.org/10.1080/03610918.2015.1053925

Bertsimas, D., & King, A. (2017). Logistic Regression: From Art to Science. Statistical Science, 32(3), 367–384. https://doi.org/10.1214/16-STS602

## Notice
