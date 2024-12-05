"""
Curve fitting wavelength over temperature for blackbody radiation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
A = 13900
B = 1.689
T_0 = 293
R_0 = 1.1
a_0 = 4.5e-3


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

def get_wavelength(seperation):
    return np.sqrt(A/(-B + np.sqrt(((2/np.sqrt(3))*np.sin(seperation*(np.pi/180)) + (1/2))**2 + (3/4))))

def get_temperature(voltage, current):
    return T_0 + ((voltage/(current * R_0)) - 1) / a_0

def f(x, a, b):
    return a/x

def lambda_unc(x):
    return 0.045*x

def ideal(x):
    return 2.898e-3/x


# Load data
data = np.loadtxt('langle2.csv', delimiter=',', skiprows=1)
voltage = data[:,0]
current = data[:,1]
seperation = (data[:,3] - data[:,2])
# area_curve = data[:,4]

wavelength = get_wavelength(seperation) *1e-9
y_unc = lambda_unc(wavelength)
temperature = get_temperature(voltage, current)
print(seperation)
print(wavelength)
print(temperature)

# Define model

# Curve fit
popt, pcov = curve_fit(f, temperature, wavelength)
# pstd = np.sqrt(np.diag(pcov))

# Plot data
plt.errorbar(temperature, wavelength, yerr=y_unc, fmt='o', label='Data')
plt.plot(temperature, f(temperature, *popt), label='Fit')
plt.plot(temperature, ideal(temperature), label='Ideal')
plt.ylabel('')
plt.xlabel('')
plt.legend()
plt.show()

# # Plot residuals
# residuals = radius - f(current, *popt)
# plt.errorbar(current, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
# plt.axhline(0, color='black', lw=1, linestyle='--')
# plt.ylabel('Residuals')
# plt.xlabel('Current (A)')
#
# # Print curve fit values
# chi2 = reduced_chi_squared(current, radius, f(current, *popt), x_unc, 1)
# print('Reduced Chi Squared:', chi2)
# print('e/m:', popt[0])
# print('Uncertainty:', pstd[0])
print(popt[0])
