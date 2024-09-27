import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load and preprocess the data
temp_data = np.genfromtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/FittingEx3/co2_temp_1000_cleanend.csv', delimiter=',')

year = temp_data[:, 0]
temp_mean = temp_data[:, 1]
temp_std = temp_data[:, 2]
conc_mean = temp_data[:, 3]
conc_std = temp_data[:, 4]

# Filter out NaN and inf values
valid_indices = ~np.isnan(year) & ~np.isnan(temp_mean) & ~np.isnan(conc_mean) & ~np.isinf(year) & ~np.isinf(temp_mean) & ~np.isinf(conc_mean)
year = year[valid_indices]
temp_mean = temp_mean[valid_indices]
conc_mean = conc_mean[valid_indices]

# Define the logarithmic function
def log(x, a, b):
    return a * np.log(x) + b

# Fit the logarithmic function to the data
popt, pcov = curve_fit(log, year, temp_mean)
plt.scatter(conc_mean, temp_mean, label='Data', color='blue')
plt.plot(conc_mean, log(conc_mean, *popt), label='Fit', color='red')
plt.xlabel('CO2 Concentration (ppm)')
plt.ylabel('Temperature Mean (C)')
plt.title('Logarithmic Relationship between Temperature Mean and CO2 Concentration')
plt.legend()
plt.show()