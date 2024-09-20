import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/FIttingEx1/co2_annmean_mlo.csv', delimiter=',', skiprows=89)
year = data[:,0]
mean = data[:,1]
uncertainty = data[:,2]

#linear fit

def f(x, a, b):
    return a*x + b

popt, pcov = curve_fit(f, year, mean, sigma=uncertainty, absolute_sigma=True)
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.errorbar(year, mean, yerr=uncertainty, fmt='o', label='data')
plt.plot(year, f(year, *popt), label='fit')
plt.xlabel('Year')
plt.ylabel('CO2 concentration (ppm)')
plt.legend()
plt.title('Linear Fit')

plt.xticks(np.arange(int(year.min()), int(year.max()) + 1, 2)) 

residuals = mean - f(year, *popt)
plt.subplot(2, 1, 2)
plt.errorbar(year, residuals, yerr=uncertainty, fmt='o', label='residuals')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residuals')

plt.xticks(np.arange(int(year.min()), int(year.max()) + 1, 2)) 

plt.tight_layout()
plt.savefig('Exercise 1')
print('a =', popt[0])
print('b =', popt[1])
print('r^2 =', np.corrcoef(mean, f(year, *popt))[0,1]**2)
