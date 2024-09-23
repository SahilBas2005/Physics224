"""
Curve fitting (polynomial) for c02 emissions data from 1959 to 2023
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt


#Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


#Define model
def polynomial(x, a, b, c):
    return (a/2)*(x**2) + b*x + c


#Load data
data = np.loadtxt('co2_annmean_mlo.csv', delimiter=',', skiprows=44)

year = data[:, 0]
mean = data[:, 1]
unc = data[:, 2]

#Calculate parameters for fit
p_opt, p_cov = curve_fit(polynomial, year, mean, sigma=unc, absolute_sigma=True)
p_std = np.sqrt(np.diag(p_cov))

#Create figure
plt.figure(figsize=(8, 2))

#Plot data with model
plt.subplot(2, 2, 1)
plt.title("Mean C02 Emissions from 2004 to 2023")
plt.plot(year, polynomial(year, p_opt[0], p_opt[1], p_opt[2]), label='Quadratic Regression')
plt.errorbar(year, mean, yerr=unc, marker='o', ls='', lw=2, label='Mauna Loa Observatory')
plt.xlabel("Time (year)")
plt.ylabel("Mean C02 Emissions (ppm)")
plt.legend()

#Plot residuals
plt.subplot(2, 2, 2)
plt.title('Residuals from Mean C02 Emissions from 2004 to 2023')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(year, mean - polynomial(year, p_opt[0], p_opt[1], p_opt[2]), yerr=unc, marker='o', ls='', lw=2, label='Residuals')
plt.xlabel("Time (year)")
plt.ylabel("Residuals for Mean C02 Emissions (ppm)")
plt.legend()

#Plot extrapolated model
plt.subplot(2, 2, 4)
plt.title("Mean C02 Emissions from 2004 to 2023 Extrapolated Linear Model")
plt.plot(np.arange(1960, 2061), polynomial(np.arange(1960, 2061), p_opt[0], p_opt[1], p_opt[2]), label='Polynomial Regression')
plt.xlabel("Time (year)")
plt.ylabel("Mean C02 Emissions (ppm)")
plt.legend()

plt.show()

#Print statistical values
print(f'chi^2 = {reduced_chi_squared(year, mean, polynomial(year, p_opt[0], p_opt[1], p_opt[2]), unc, 3)}')
print(f'a =  {p_opt[0]} u(a) = {p_std[0]}, b = {p_opt[1]} u(b) = {p_std[1]}, c = {p_opt[2]} u(c) = {p_std[2]}')
print(f'1960 (Predicted): {polynomial(1960, p_opt[0], p_opt[1], p_opt[2])}ppm')
print(f'2060 (Predicted): {polynomial(2060, p_opt[0], p_opt[1], p_opt[2])}ppm')
