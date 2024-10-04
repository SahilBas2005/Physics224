"""
Curve fitting (periodic) for c02 emissions data from March 1958 through July 2024
"""
import math

import numpy as np
from numpy.ma.core import floor
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


# Define model
def exponential(t, v_0, time_const):
    return v_0**(-t/time_const)


# Load data
data = np.loadtxt('co2_mm_mlo.csv', delimiter=',', skiprows=41)

data_clean(data, 7)

year = data[:, 0]
month = data[:, 1]
decimal_date = data[:, 2]
mean = data[:, 3]
unc = data[:, 7]

# Calculate parameters for fit
p_opt, p_cov = curve_fit(periodic, decimal_date, mean, sigma=unc, absolute_sigma=True)
p_std = np.sqrt(np.diag(p_cov))

# Create figure
plt.figure(figsize=(8, 2))

# Plot data with model
plt.subplot(2, 2, 1)
plt.title("Mean C02 Emissions from 1958 to 2024")
plt.plot(decimal_date, periodic(decimal_date, p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4], p_opt[5]), label='Periodic Regression')
plt.errorbar(decimal_date, mean, yerr=unc, marker='.', ls='', lw=2, label='NOAA Data')
plt.xlabel("Time (year)")
plt.ylabel("Mean C02 Emissions (ppm)")
plt.legend()

# Plot residuals
plt.subplot(2, 2, 2)
plt.title('Residuals from Mean C02 Emissions from 1958 to 2024')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(decimal_date, mean - periodic(decimal_date, p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4], p_opt[5]), yerr=unc, marker='o', ls='', lw=2, label='Residuals')
plt.xlabel("Time (year)")
plt.ylabel("Residuals for Mean C02 Emissions (ppm)")
plt.legend()

# Plot model on its own
plt.subplot(2, 2, 4)
plt.title("Mean C02 Emissions from 1958 to 2100 Extrapolated Periodic Model")
plt.plot(np.arange(1958, 2101, step=0.083), periodic(np.arange(1958, 2101, step=0.083), p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4], p_opt[5]), label='Periodic Regression')
plt.xlabel("Time (year)")
plt.ylabel("Mean C02 Emissions (ppm)")
plt.legend()

plt.show()

# Create a new extrapolated time range and model the data on this range to answer the three questions in the lab
data_range = np.arange(1958, 2101, step=0.083)
model_data = periodic(data_range, p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4], p_opt[5])

# Determine highest emission months each year
print(f'Highest month in each year {highest_months(year, month, mean)}')

# Determine when the model surpasses 570ppm
print(f'The model surpasses 2570ppm at {get_prediction(570, data_range, model_data)}')  # Evaluates to April 2071

# Determine when the CO2 minimum in a year surpasses the CO2 maximum in 2000
max_in_2000 = max_in_year(2000, decimal_date, mean)
iterator = 0
while min_in_year(np.floor(data_range[iterator]), data_range, model_data) <= max_in_2000:
    iterator += 1
print(f'The CO2 minimum in {np.floor(data_range[iterator])} surpasses the CO2 maximum in 2000')


# Print statistical values
print(f'chi^2 = {reduced_chi_squared(decimal_date, mean, periodic(decimal_date, p_opt[0], p_opt[1], p_opt[2],
                                                                  p_opt[3], p_opt[4], p_opt[5]), unc, 6)}')
print(f'a =  {p_opt[0]} u(a) = {p_std[0]}, b = {p_opt[1]} u(b) = {p_std[1]}, c = {p_opt[2]} u(c) = {p_std[2]}')
print(f'd =  {p_opt[3]} u(d) = {p_std[3]}, e = {p_opt[4]} u(e) = {p_std[4]}, phi = {p_opt[5]} u(phi) = {p_std[5]}')
