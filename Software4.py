# This code was developed by Saâdaoui et al. (2024) for performing multiresolution causality analysis 
# between two time series using Variational Mode Decomposition (VMD). 
# 
# If you use this code in your research, please cite it as follows:
# ---------------------------------------------------------------------------------------------
# Saâdaoui, F., Herch, H., & Rabbouch, H. (2024). Multiresolution Granger causality testing 
# with variational mode decomposition: A Python software. Submitted to the Journal of 
# Applied Statistics.
# ---------------------------------------------------------------------------------------------
#
# Install and import required libraries
!pip install vmdpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from vmdpy import VMD
from google.colab import files
import io
import contextlib

# Function to perform VMD and return IMF components
def perform_vmd(data, alpha, num_modes):
    vmd = VMD(data, alpha=alpha, tau=0.0, K=num_modes, DC=0, init=1, tol=1e-6)
    u, u_hat, omega = vmd
    return u


def granger_causality_test(data1, data2, maxlag, significance_level=0.05):
    results = {}
    with contextlib.redirect_stdout(io.StringIO()):  # Redirect the verbose output to an in-memory buffer
        results = grangercausalitytests(np.column_stack((data1, data2)), maxlag)
        
    for lag in range(1, maxlag + 1):
        f_statistic = results[lag][0]['ssr_ftest'][0]
        p_value = results[lag][0]['ssr_ftest'][1]
        recommendation = "Reject H0" if p_value < significance_level else "Fail to Reject H0"
        print(f"Lag {lag}: F-statistic = {f_statistic:.4f}, p-value = {p_value:.4f}, Recommendation: {recommendation}")

# Function to plot data
def plot_data(data, title, xlabel, ylabel, legend):
    plt.plot(data, label=legend)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

# User inputs for VMD and Granger causality
num_modes = int(input("Enter the number of modes (IMFs) to extract: "))
maxlag = int(input("Enter the maximum lag for Granger causality test: "))
alpha = 2000  # Trade-off parameter for VMD

# Upload the data file
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_excel(filename)

# Display the first few rows of the DataFrame to verify the content
print("Data Preview:")
print(df.head())

# Extract the columns
try:
    column1, column2 = df.columns[:2]
    X = df[column1].values
    Y = df[column2].values
except KeyError:
    print("Error: Column names do not match. Please verify the column names in the Excel file.")
    exit()

# Differencing the data for standard Granger causality test
X_diff = np.diff(X)
Y_diff = np.diff(Y)

# Reminder of H0 before the Granger causality test
print("\nH0: Column 2 does not Granger-cause Column 1")

# Perform Granger causality test on differenced data
print("\nGranger Causality Test on Differenced Data:")
granger_causality_test(X_diff, Y_diff, maxlag)

# Perform VMD on raw X and Y data (not differenced)
vmd_components = [perform_vmd(series, alpha, num_modes) for series in [X, Y]]

# Perform Granger causality test on VMD components (IMFs 2, 3, and 4) from raw data
for i in range(1, num_modes):  # Start from IMF 2 (omit IMF 1)
    print(f"\nGranger Causality Test for IMF {i+1} on Raw Data:")
    granger_causality_test(vmd_components[0][i, :], vmd_components[1][i, :], maxlag)

# Plot original data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_data(X, 'Original Data - X', 'Time', 'Value', 'X')

plt.subplot(2, 1, 2)
plot_data(Y, 'Original Data - Y', 'Time', 'Value', 'Y')
plt.tight_layout()
plt.show()

# Plot VMD components
plt.figure(figsize=(12, 8))
for i in range(num_modes):
    # Plot the X VMD component
    plt.subplot(num_modes, 2, 2*i + 1)  # Corrected subplot: (nrows, ncols, index)
    plot_data(vmd_components[0][i, :], f'VMD Component IMF {i + 1} - X', 'Time', 'Amplitude', f'IMF {i + 1} - X')

    # Plot the Y VMD component
    plt.subplot(num_modes, 2, 2*i + 2)  # Corrected subplot: (nrows, ncols, index)
    plot_data(vmd_components[1][i, :], f'VMD Component IMF {i + 1} - Y', 'Time', 'Amplitude', f'IMF {i + 1} - Y')

plt.tight_layout()
plt.show()
