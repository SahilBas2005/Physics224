import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 # Define constants
λ = 552*10e-9
t = 7.71*10e-3


def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

data = np.loadtxt('refraction.csv', delimiter=',', skiprows=1)
fringes = data[:,0]
angle = data[:,1]
x_unc = data[:,2]
y_unc = data[:,3]

def f(x, n):
    return (t/λ) * (x**2) * (1-(1/n))

popt, pcov = curve_fit(f, angle, fringes, sigma=x_unc)
pstd = np.sqrt(np.diag(pcov))

plt.errorbar(angle, fringes, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.plot(angle, f(angle, *popt), label='Fit')
plt.ylabel('Number of Fringes')
plt.xlabel('Angle (radians)')
plt.legend()
plt.show()

residuals = fringes - f(angle, *popt)
plt.errorbar(angle, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.ylabel('Residuals')
plt.xlabel('Angle (radians)')
plt.show()

chi2 = reduced_chi_squared(angle, fringes, f(angle, *popt), x_unc, 1)
print('Reduced Chi Squared:', chi2)
print('Index of Refraction:', popt[0])
print('Uncertainty:', pstd[0])
