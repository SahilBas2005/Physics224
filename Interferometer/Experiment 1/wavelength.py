import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

data = np.loadtxt('knob.csv', delimiter=',', skiprows=1)
fringes = data[:,0]
knob = data[:,1]
x_unc = data[:,2]
y_unc = data[:,3]

def f(x, λ):
    return 2*x/λ

popt, pcov = curve_fit(f, knob, fringes, sigma=x_unc, p0=[0.5])
pstd = np.sqrt(np.diag(pcov))

plt.errorbar(knob, fringes, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.plot(knob, f(knob, *popt), label='Fit')
plt.ylabel('Number of Fringes')
plt.xlabel('Knob Position (μm)')
plt.legend()
plt.show()

residuals = fringes - f(knob, *popt)
plt.errorbar(knob, residuals, xerr=x_unc, yerr=y_unc, fmt='o', label='Data')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xlabel('Knob Position (μm)')
plt.ylabel('Residuals')
plt.show()

chi2 = reduced_chi_squared(knob, fringes, f(knob, *popt), x_unc, 1)
print('Reduced Chi Squared:', chi2)
print('Wavelength:', popt[0], 'μm')
print('Uncertainty:', pstd[0], 'μm')
