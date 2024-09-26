import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

#decimal date to date (for later)
def decimal_to_date(decimal_date):
    year = int(decimal_date)
    rem = decimal_date - year
    month = int(rem * 12) + 1
    day = int((rem * 12 - month + 1) * 30) + 1
    return year, month, day

#Loading the data
data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/FittingEx2/co2_mm_mlo.csv', delimiter=',', skiprows=236)

#define a Function to filter data with zero (or undefined) uncertainty vals
def zero_uncertainty(data):
    mask = data[:, 6] >= 0
    return data[mask]

data = zero_uncertainty(data)

#Extracting columns from the data
year = data[:,0]
month = data[:,1]
decimal_date = data[:,2]
average = data[:,3]
deseasonalized = data[:,4]
ndays = data[:,5]
sdev = data[:,6]
uncertainty = data[:,7]

#Defining the model function
def sinusodal_quadratic(t ,a, b, c, d, e, phi, psi):
    return a*t**2 + b*t + c + d*np.sin(2*np.pi*t - phi) + e*np.sin(4*np.pi*t + psi)

#curve fit to the data
popt, pcov = curve_fit(sinusodal_quadratic, decimal_date, average, sigma=uncertainty, absolute_sigma=True)

#data and fit
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.errorbar(decimal_date, average, yerr=uncertainty, fmt='.', label='data', markersize=3)
ax1.plot(decimal_date, sinusodal_quadratic(decimal_date, *popt), label='fit')
ax1.set_xlabel('Date')
ax1.set_ylabel('CO2 concentration (ppm)')
ax1.legend()
ax1.set_title('Sinusoidal Quadratic Fit')
plt.tight_layout()
#plt.savefig('fitting_data_and_fit.png')
plt.show()

#residuals
fig2, ax2 = plt.subplots(figsize=(10, 6))
residuals = average - sinusodal_quadratic(decimal_date, *popt)
ax2.errorbar(decimal_date, residuals, yerr=uncertainty, fmt='o', label='residuals', markersize=3)
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('Residuals')
ax2.legend()
ax2.set_title('Residuals')
plt.tight_layout()
#plt.savefig('fitting_residuals.png')
plt.show()

#params and chi squared.
a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
e = popt[4]
phi = popt[5]
psi = popt[6]
chi22 = np.sum((residuals/uncertainty)**2) / (len(decimal_date) - 6)
print(a, b, c, d, e, phi, psi, chi22)

#second question
#function to find the date
def find_date(t, a, b, c, d, e, phi, psi, y):
    return a*t**2 + b*t + c + d*np.sin(2*np.pi*t - phi) + e*np.sin(4*np.pi*t + psi) - y
date = fsolve(find_date, 2020, args=(a, b, c, d, e, phi, psi, 2*285))
year, month, day = decimal_to_date(date)
print(year, month, day)

#third question
#finding the co2 maximum in 2020
def compute_y(t, a, b, c, d, e, phi, psi):
    return a*t**2 + b*t + c + d*np.sin(2*np.pi*t - phi) + e*np.sin(4*np.pi*t + psi)
t = 2020.43
#it usually occurs in may (xx.43)
y = compute_y(t, a, b, c, d, e, phi, psi)
print(y)

#function to find when this value will be reached
def find_date2(t, a, b, c, d, e, phi, psi, y):
    return compute_y(t, a, b, c, d, e, phi, psi) - y
date2 = fsolve(find_date2, 2020.75, args=(a, b, c, d, e, phi, psi, y))
year2, month2, day2 = decimal_to_date(date2)
print(year2, month2, day2)