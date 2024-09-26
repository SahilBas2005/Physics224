import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/FIttingEx1/co2_annmean_mlo.csv', delimiter=',', skiprows=89)
year = data[:,0]
mean = data[:,1]
uncertainty = data[:,2]

#linear fit

#def f(x, a, b):
#    return a*x + b

#quadratic fit

def f(x, a, b, c):
    return a*x**2 + b*x + c

popt, pcov = curve_fit(f, year, mean, sigma=uncertainty, absolute_sigma=True)
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.errorbar(year, mean, yerr=uncertainty, fmt='o', label='data')
plt.plot(year, f(year, *popt), label='fit')
plt.xlabel('Year')
plt.ylabel('CO2 concentration (ppm)')
plt.legend()
plt.title('Quadratic Fit')

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
print('a =', popt[0])
print('b =', popt[1])
print('c =', popt[2])
print('chi2 =', np.sum((residuals/uncertainty)**2)/(len(year)-3))
plt.savefig('Exercise 1-2.png')

a = popt[0]
b = popt[1]
c = popt[2]
chi2 = np.sum((residuals/uncertainty)**2)/(len(year)-3)

print(a, b, c, chi2)

data_2060 = f(2060, a, b, c)
data_1960 = f(1960, a, b, c)
print(data_2060, data_1960)
