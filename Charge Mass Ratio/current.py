"""
Curve fitting changes in radius of the electron beam over varying current and plotting residuals
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
mu_0 = 4*np.pi*10e-7
R = 14.2*10e-2
n = 15
k = ((mu_0*n)/(np.sqrt(2)*R))*(4/5)**(3/2)
voltage = 304.501


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

# Load data
data = np.loadtxt('var_current.csv', delimiter=',', skiprows=1)
current = data[:,0]
x_unc = data[:,1]
radius = (data[:,2]/2)*10e-2
y_unc = (data[:,3]/2)*10e-2

# Define model
def f(x, em_ratio, I_0):
    return (np.sqrt(voltage))/(np.sqrt(em_ratio)*k*(x + (I_0/np.sqrt(2))))

# Curve fit
popt, pcov = curve_fit(f, current, radius, sigma=x_unc)
pstd = np.sqrt(np.diag(pcov))

# Plot data
plt.errorbar(current, radius, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.plot(current, f(current, *popt), label='Fit')
plt.ylabel('Radius (cm)')
plt.xlabel('Current (A)')
plt.legend()
plt.show()

# Plot residuals
residuals = radius - f(current, *popt)
plt.errorbar(current, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Current (A)')
plt.show()

# Print curve fit values
chi2 = reduced_chi_squared(current, radius, f(current, *popt), x_unc, 1)
print('Reduced Chi Squared:', chi2)
print('e/m:', popt[0])
print('Uncertainty:', pstd[0])
print('I:', popt[1])
print('I:', pstd[1])
