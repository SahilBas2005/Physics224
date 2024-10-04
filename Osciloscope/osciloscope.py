import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Osciloscope/osc.csv', delimiter=',', skiprows=1)

time = data[:,0]
data_1 = data[:,1]
data_2 = data[:,2]

unc_raw = np.sqrt(20)
unc = np.full(len(data_1), unc_raw)


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


# Define model
def exponential(t, v_0, time_const, b):
    return v_0*np.exp(-t/time_const) + b

def linear_fit(x, a, b):
    return a*x + b

plt.scatter(time, data_1, label='Data 1')
popt, pcov = curve_fit(exponential, time, data_1, sigma = unc)
plt.plot(time, exponential(time, *popt), label='Fit 1')
plt.errorbar(time, data_1, yerr=unc, fmt='o')
plt.legend()
plt.show()

residuals = data_1 - exponential(time, *popt)
plt.scatter(time, residuals)
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.show()

chi_2 = reduced_chi_squared(time, data_1, exponential(time, *popt), unc, 3)