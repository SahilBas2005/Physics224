import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/FIttingEx2/co2_mm_mlo.csv', delimiter=',', skiprows=41)

def average_uncertainty(uncertainty):
    return np.mean(uncertainty)

def zero_uncertainty(data):
    mask = data[:,7] > 0
    return data[mask]

data_2 = zero_uncertainty(data)
uncertainty = data_2[:,7]
avg_unc = average_uncertainty(uncertainty)

def dezeroify(data, avg_unc):
    mask = data[:,7] <= 0
    data[mask,7] = avg_unc
    return data

data = dezeroify(data, avg_unc)

year = data[:,0]
month = data[:,1]
decimal_date = data[:,2]
average = data[:,3]
deseasonalized = data[:,4]
ndays = data[:,5]
sdev = data[:,6]
uncertainty = data[:,7]

def sinusodal_quadratic(t ,a, b, c, d, e, phi):
    return a*t**2 + b*t + c + d*np.sin(2*np.pi*t - phi) + e*np.sin(4*np.pi*t + phi)

popt, pcov = curve_fit(sinusodal_quadratic, decimal_date, average, sigma=uncertainty, absolute_sigma=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First subplot: data and fit
ax1.errorbar(decimal_date, average, yerr=uncertainty, fmt='o', label='data')
ax1.plot(decimal_date, sinusodal_quadratic(decimal_date, *popt), label='fit')
ax1.set_xlabel('Decimal Date')
ax1.set_ylabel('CO2 concentration (ppm)')
ax1.legend()
ax1.set_title('Sinusoidal Quadratic Fit')

# Second subplot: residuals
residuals = average - sinusodal_quadratic(decimal_date, *popt)
ax2.errorbar(decimal_date, residuals, yerr=uncertainty, fmt='o', label='residuals')
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel('Decimal Date')
ax2.set_ylabel('Residuals')
ax2.legend()
ax2.set_title('Residuals')

#plt.tight_layout()
#plt.show()

a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
e = popt[4]
phi = popt[5]
chi2 = np.sum((residuals/uncertainty)**2/(len(decimal_date)-6))
print(a, b, c, d, e, phi, chi2)
