"""
Curve fitting currents and voltages from a lightbulb circuit.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define chi squared function
def reduced_chi_squared(y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


def uncertainty(per, cout,x):
    return np.round(per*x + cout, 4)


def log_uncertainty(unc, value):
    return unc / value

def log_model(x, a, b):
    return b*x + np.log(a)

def model(x, a, b):
    return a*x**b


def ideal_model(x, a):
    return a*x**0.6


# Uncertainty constants
delta_v_per = 0.0005
delta_v_cout = 0.002
delta_I_per = 0.003
delta_I_cout = 0.001

# Load data
data = np.genfromtxt('mes.csv', delimiter=',', skip_header=1)
voltage = data[:,0]
current = data[:,1]

# Calculate uncertainty
uncertainty_v = uncertainty(delta_v_per, delta_v_cout, voltage)
uncertainty_I = uncertainty(delta_I_per, delta_I_cout, current)

# Load logarithmic data
voltage_log = np.log(voltage)
current_log = np.log(current)
uncertainty_v_log = log_uncertainty(uncertainty_v, voltage)
uncertainty_I_log = log_uncertainty(uncertainty_I, current)

# Fit model
popt, pcov = curve_fit(model, voltage, current, sigma=uncertainty_I, absolute_sigma=True, p0 = [1, 0.6])
pstd = np.sqrt(np.diag(pcov))

# Fit log model
popt_log, pcov_log = curve_fit(log_model, voltage_log, current_log, sigma=uncertainty_I_log, absolute_sigma=True)
pstd_log = np.sqrt(np.diag(pcov_log))

# Fit ideal model
popt_i, pcov_i = curve_fit(ideal_model, voltage, current, sigma=uncertainty_I, absolute_sigma=True)
pstd_i = np.sqrt(np.diag(pcov_i))

# Plot models on linear scale
plt.errorbar(voltage, current, yerr=uncertainty_I, xerr=uncertainty_v, fmt='.', label='Measured Current', markersize=3)
plt.plot(voltage, model(voltage, *popt), lw=3, label='Power Law Fit', color='red')
plt.plot(voltage, np.exp(log_model(voltage_log, *popt_log)), label='Logarithm Fit', color='orange')
plt.plot(voltage, ideal_model(voltage, *popt_i), label='Ideal Model', color='green')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.legend()
plt.title('Current vs Voltage')
plt.tight_layout()
plt.show()

# Plot models on log scale
plt.errorbar(voltage, current, yerr=uncertainty_I, xerr=uncertainty_v, fmt='.', label='Measured Current', markersize=3)
plt.plot(voltage, model(voltage, *popt), lw=3, label='Power Law Fit', color='red')
plt.plot(voltage, np.exp(log_model(voltage_log, *popt_log)), label='Logarithm Fit', color='orange')
plt.plot(voltage, ideal_model(voltage, *popt_i), label='Ideal Model', color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.legend()
plt.title('Current vs Voltage (Log Scale)')
plt.tight_layout()
plt.show()

# Plot residuals
plt.title('Residuals from Current over Voltage Power Law Fit')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(voltage, current - model(voltage, *popt), yerr=uncertainty_I, marker='o', ls='', lw=2, label='Residuals for Power Law Fit', color='red')
plt.ylabel("Current (A)")
plt.xlabel("Voltage (V)")
plt.legend()
plt.show()
plt.title('Residuals from Current over Voltage Logarithm Fit')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(voltage, current - np.exp(log_model(voltage_log, *popt_log)), yerr=uncertainty_I, marker='o', ls='', lw=2, label='Residuals for Logarithm Fit', color='orange')
plt.ylabel("Current (A)")
plt.xlabel("Voltage (V)")
plt.legend()
plt.show()
plt.title('Residuals from Current over Voltage Ideal Model')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(voltage, current - ideal_model(voltage, *popt_i), yerr=uncertainty_I, marker='o', ls='', lw=2, label='Residuals for Ideal Model', color='green')
plt.ylabel("Current (A)")
plt.xlabel("Voltage (V)")
plt.legend()
plt.show()

# Calculate chi squared
chi_2 = reduced_chi_squared(current, model(voltage, *popt), uncertainty_I, 2)
chi_2_log = reduced_chi_squared(current_log, log_model(voltage_log, *popt), uncertainty_I_log, 2)
chi_2_i = reduced_chi_squared(current, ideal_model(voltage, *popt_i), uncertainty_I, 1)

# Print statistical values
print("Power Law Fit")
print(f'chi^2 = {chi_2}')
print(f'a =  {popt[0]} u(a) = {pstd[0]}, b = {popt[1]} u(b) = {pstd[1]}')
print("Logarithm Fit")
print(f'chi^2 = {chi_2_log}')
print(f'a =  {popt_log[0]} u(a) = {pstd_log[0]}, b = {popt_log[1]} u(b) = {pstd_log[1]}')
print("Ideal Model")
print(f'chi^2 = {chi_2_i}')
print(f'a =  {popt_i[0]} u(a) = {pstd_i[0]}, b = 0.6')
