"""
Curve fitting wavelength over temperature for blackbody radiation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define constants
A = 13900
B = 1.689
T_0 = 293
R_0 = 1.1
a_0 = 4.5e-3
V_unc = 0.01
I_unc = 0.005
h = 6.63e-34
c = 3e8
k = 1.38e-23


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)

def get_wavelength(seperation):
    return np.sqrt(A/(-B + np.sqrt(((2/np.sqrt(3))*np.sin(seperation*(np.pi/180)) + (1/2))**2 + (3/4))))

def get_temperature(voltage, current):
    return T_0 + ((voltage/(current * R_0)) - 1) / a_0

def f(x, a, b):
    return a/x + b

def f2(x, a, b):
    return a*x**b

def ideal(x):
    return 5.67e-8 * x**4

def lambda_unc(x):
    return 0.045*x

def temperature_unc(voltage, current):
    return (np.sqrt((current*V_unc)**2) + ((voltage*I_unc)**2))/((current**2)*R_0*a_0)


# Load data
data = np.loadtxt('langle2.csv', delimiter=',', skiprows=1)
voltage = data[:,0]
current = data[:,1]
seperation = (data[:,3] - data[:,2])
area_curve = data[:,4] * 2*h*(c**2)/k # Convert intensities using Planck's radiation law
area_unc = data[:,5] * 2*h*(c**2)/k #  Convert intensities using Planck's radiation law

wavelength = get_wavelength(seperation) *1e-9
y_unc = lambda_unc(wavelength)
temperature = get_temperature(voltage, current)
temp_unc = temperature_unc(voltage, current)

# Define model

# Curve fit
popt, pcov = curve_fit(f, temperature, wavelength, sigma=temp_unc)
pstd = np.sqrt(np.diag(pcov))

# Plot data
plt.errorbar(temperature, wavelength, xerr=temp_unc, yerr=y_unc, fmt='o', label='Data')
plt.plot(temperature, f(temperature, *popt), label='Fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Wavelength (m)')
plt.legend()
plt.show()

# Plot residuals
residuals = wavelength - f(temperature, *popt)
plt.errorbar(temperature, residuals, xerr=temp_unc, yerr=y_unc, fmt='o', label='Residuals')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals (m)')
plt.show()
print(popt[0])
print(pstd[0])
print(popt[1])
print(pstd[1])

# Stefan Boltzmann
popt, pcov = curve_fit(f2, temperature, area_curve, sigma=temp_unc, p0=[5.67e-8, 4])
pstd = np.sqrt(np.diag(pcov))

# Plot data
plt.errorbar(temperature, area_curve, xerr=temp_unc, yerr=area_unc, fmt='o', label='Data')
plt.plot(temperature, f2(temperature, *popt), label='Fit')
plt.plot(temperature, ideal(temperature), label='ideal')
plt.xlabel('Temperature (K)')
plt.ylabel('Total Intentsity (V)')
plt.legend()
plt.show()

# Plot residuals
residuals = area_curve - f2(temperature, *popt)
plt.errorbar(temperature, residuals, xerr=temp_unc, yerr=area_unc, fmt='o', label='Residuals')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.xlabel('Temperature (K)')
plt.ylabel('Residuals (m)')
plt.show()
print("-----------")
print(popt[0])
print(pstd[0])
print(popt[1])
print(pstd[1])
