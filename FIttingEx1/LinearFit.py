"""
Curve fitting (linear) for c02 emissions data from 1959 to 2023
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
def linear(x, a, b):
    return a * (x - 1990) + b


#Load data
data = np.loadtxt('co2_annmean_mlo.csv', delimiter=',', skiprows=89)

year = data[:, 0]
mean = data[:, 1]
unc = data[:, 2]

#Calculate parameters for fit
p_opt, p_cov = curve_fit(linear, year, mean, sigma=unc, absolute_sigma=True)
p_std = np.sqrt(np.diag(p_cov))

#Create figure
plt.figure(figsize=(8, 2))

#Plot data with model
plt.subplot(2, 2, 1)
plt.title("Mean C02 Emissions from 2004 to 2023")
plt.plot(year, linear(year, p_opt[0], p_opt[1]), label='Linear Regression')
plt.errorbar(year, mean, yerr=unc, marker='o', ls='', lw=2, label='Mauna Loa Observatory')
plt.xlabel("Time (year)")
plt.ylabel("Mean C02 Emissions (ppm)")
plt.legend()
plt.xticks(np.arange(int(year.min()), int(year.max()) + 1, 2))

#Plot residuals
plt.subplot(2, 2, 2)
plt.title('Residuals from Mean C02 Emissions from 2004 to 2023')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(year, mean - linear(year, p_opt[0], p_opt[1]), yerr=unc, marker='o', ls='', lw=2, label='Residuals')
plt.xlabel("Time (year)")
plt.ylabel("Residuals for Mean C02 Emissions (ppm)")
plt.legend()
plt.xticks(np.arange(int(year.min()), int(year.max()) + 1, 2))

#Plot extrapolated model
plt.subplot(2, 2, 4)
plt.title("Mean C02 Emissions from 2004 to 2023 Extrapolated Linear Model")
plt.plot(np.arange(1960, 2061), linear(np.arange(1960, 2061), p_opt[0], p_opt[1]), label='Linear Regression')
plt.xlabel("Time (year)")
plt.ylabel("Mean C02 Emissions (ppm)")
plt.legend()

plt.show()

#Print statistical values
print(f'chi^2 = {reduced_chi_squared(year, mean, linear(year, p_opt[0], p_opt[1]), unc, 2)}')
print(f'a =  {p_opt[0]} u(a) = {p_std[0]}, b = {p_opt[1]} u(b) = {p_std[1]}')
print(f'1960 (Predicted): {linear(1960, p_opt[0], p_opt[1])}ppm')
print(f'2060 (Predicted): {linear(2060, p_opt[0], p_opt[1])}ppm')
