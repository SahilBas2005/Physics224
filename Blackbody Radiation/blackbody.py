"""
Curve fitting wavelength over temperature for blackbody radiation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
A = 13900
B = 1.689


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

def get_wavelength(seperation):
    return A/(-B + np.sqrt(((2/np.sqrt(3))*np.sin(seperation) + (1/2))**2 + (3/4)))


# Load data
data = np.loadtxt('langle.csv', delimiter=',', skiprows=1)
voltage = data[:,0]
current = data[:,1]
seperation = data[:,3] - data[:,2]
area_curve = data[:,4]

wavelength = get_wavelength(seperation)
print(wavelength)

# # Define model
# def f(x, em_ratio):
#     return (np.sqrt(voltage))/(np.sqrt(em_ratio)*k*(x + (B_e/(np.sqrt(20)*k))))
#
# # Curve fit
# popt, pcov = curve_fit(f, current, radius, sigma=x_unc)
# pstd = np.sqrt(np.diag(pcov))

# # Plot data
# plt.errorbar(current, radius, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
# plt.plot(current, f(current, *popt), label='Fit')
# plt.ylabel('Radius (cm)')
# plt.xlabel('Current (A)')
# plt.legend()
#
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
