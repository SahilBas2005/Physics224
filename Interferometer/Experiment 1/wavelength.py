import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Interferometer/Experiment 1/knob.csv', delimiter=',', skiprows=1)
fringes = data[:,0]
knob = data[:,1]
unc = data[:,2]

def f(x, λ):
    return 2*x/λ

popt, pcov = curve_fit(f, fringes, knob, sigma=unc, p0=[0.5])

plt.errorbar(fringes, knob, yerr=unc, fmt='o', label='Data')
plt.plot(fringes, f(fringes, *popt), label='Fit')
plt.xlabel('Number of Fringes')
plt.ylabel('Knob Position (μm)')
plt.legend()
plt.show()

residuals = knob - f(fringes, *popt)
plt.scatter(fringes, residuals)
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xlabel('Number of Fringes')
plt.ylabel('Residuals (μm)')
plt.show()

chi2 = reduced_chi_squared(fringes, knob, f(fringes, *popt), unc, 1)
print('Reduced Chi Squared:', chi2)
pstd = np.sqrt(np.diag(pcov))
print(pstd)
print('Wavelength:', popt[0], 'μm')
print('Uncertainty:', pstd[0], 'μm')
