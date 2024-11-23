"""
Curve fitting the extra b field over changes in radius of the electron and plotting residuals
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
mu_0 = 4*np.pi*10e-7
R = 13.2*10e-2
n = 15


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

def get_b_field(current):
    return ((mu_0 * n * current) / R) * (4 / 5) ** (3 / 2)

def correct_b_field(b_field, r):
    return b_field*(1 - ((r**4) / (R**4*(0.6583 + 0.29*(r/R)**2)**2)))

# Load data
data = np.loadtxt('var_current.csv', delimiter=',', skiprows=1)
current = data[:,0]
x_unc = (data[:,3]/2)*10e-2
radius = (data[:,2]/2)*10e-2
y_unc = get_b_field(data[:,1])
b_coil = correct_b_field(get_b_field(current), radius)

# Define model
def f(x, a, B_e):
    return a*(1/x) - B_e  # Apply corrected current

# Curve fit
popt, pcov = curve_fit(f, radius, b_coil, sigma=x_unc, p0= [5.9*10e-5, 2.3*10e-5])
pstd = np.sqrt(np.diag(pcov))

# Plot data
plt.errorbar(radius, b_coil, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.plot(radius, f(radius, *popt), label='Fit')
plt.xlabel('Radius (m)')
plt.ylabel('Coil B-field (T)')
plt.legend()
plt.show()

# Plot residuals
residuals = b_coil - f(radius, *popt)
plt.errorbar(radius, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Radius (m)')
plt.show()

# Print curve fit values
chi2 = reduced_chi_squared(radius, b_coil, f(radius, *popt), x_unc, 1)
print('Reduced Chi Squared:', chi2)
print('a:', popt[0])
print('Uncertainty:', pstd[0])
print('B_e:', popt[1])
print('Uncertainty:', pstd[1])
