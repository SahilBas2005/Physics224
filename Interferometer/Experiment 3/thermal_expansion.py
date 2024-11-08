import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Define constants
λ = 552*10e-9
L_0 = 94.9*10e-3

# Skip unrepresentative data points
num_skipped = 2


def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Interferometer/Experiment 3/thermal_expansion.csv', delimiter=',', skiprows=num_skipped+1)
fringes = data[:,0]
temperature = data[:,1]
x_unc = data[:,2]
y_unc = data[:,3]

def f(x, a, b):
    return (2*L_0/λ)*a*x + b

popt, pcov = curve_fit(f, temperature, fringes, sigma=x_unc, p0=[2.3*1e-5, 0])
pstd = np.sqrt(np.diag(pcov))

#plt.errorbar(temperature, fringes, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
#plt.plot(temperature, f(temperature, *popt), label='Fit')
#plt.ylabel('Number of Fringes')
#plt.xlabel('Change in Temperature (°C)')
#plt.legend()
#plt.savefig('thermal_expansion.png')

residuals = fringes - f(temperature, *popt)
plt.errorbar(temperature, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Change in Temperature (°C)')
plt.legend()
plt.savefig('thermal_expansion_residuals.png')

chi2 = reduced_chi_squared(temperature, fringes, f(temperature, *popt), x_unc, 1)
print('Reduced Chi Squared:', chi2)
print('Thermal Expansion Coefficient:', popt[0])
print('b', popt[1])
print('u(b)', pstd[1])
print('Uncertainty:', pstd[0])
