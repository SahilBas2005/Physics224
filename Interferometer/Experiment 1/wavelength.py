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
unc = data[:,2]

def f(x, λ):
    return 2*x/λ
def ideal(x):
    return 3.33*x

popt, pcov = curve_fit(f, knob, fringes, sigma=unc, p0=[0.5])

plt.errorbar(knob, fringes, yerr=unc, fmt='o', label='Data')
plt.plot(knob, f(knob, *popt), label='Fit')
plt.plot(knob, ideal(knob), label='Ideal')
plt.ylabel('Number of Fringes')
plt.xlabel('Knob Position (μm)')
plt.legend()
plt.show()

residuals = fringes - f(knob, *popt)
plt.scatter(knob, residuals)
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xlabel('Number of Fringes')
plt.ylabel('Residuals (μm)')
plt.show()

chi2 = reduced_chi_squared(knob, fringes, f(knob, *popt), unc, 1)
print('Reduced Chi Squared:', chi2)
pstd = np.sqrt(np.diag(pcov))
print(pstd)
print('Wavelength:', popt[0], 'μm')
print('Uncertainty:', pstd[0], 'μm')
