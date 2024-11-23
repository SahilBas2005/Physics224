"""
Curve fitting changes in radius of the electron beam over varying voltage and plotting residuals
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
mu_0 = 4*np.pi*10e-7
R = 14.2*10e-2
n = 15
k = ((mu_0*n)/(np.sqrt(2)*R))*(4/5)**(3/2)
current = 1.497


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

# Load data
data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Charge Mass Ratio/var_voltage.csv', delimiter=',', skiprows=1)
voltage = data[:,0]
x_unc = data[:,1]
radius = (data[:,2]/2)*10e-2
y_unc = (data[:,3]/2)*10e-2

# Define model
def f(x, em_ratio):
    return (np.sqrt(x))/(np.sqrt(em_ratio)*k*(current + (-0.33/np.sqrt(2))))

# Curve fit
popt, pcov = curve_fit(f, voltage, radius, sigma=x_unc, maxfev=1000)
pstd = np.sqrt(np.diag(pcov))

# Plot data
#plt.errorbar(voltage, radius, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
#plt.plot(voltage, f(voltage, *popt), label='Fit')
#plt.ylabel('Radius (cm)')
#plt.xlabel('Voltage (V)')
#plt.legend()
#plt.savefig('voltage.png')

# Plot residuals
residuals = radius - f(voltage, *popt)
plt.errorbar(voltage, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Voltage (V)')
plt.savefig('voltage_residuals.png')

# Print curve fit values
chi2 = reduced_chi_squared(voltage, radius, f(voltage, *popt), x_unc, 1)
print('Reduced Chi Squared:', chi2)
print('e/m:', popt[0])
print('Uncertainty:', pstd[0])
