import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Osciloscope/osc.csv', delimiter=',', skiprows=1)

time = data[:,0]
data_1 = data[:,1]
data_2 = data[:,2]
unc = data[:,3]

ms_time = time * 1e-3

# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


# Define model
def exponential(t, v_0, time_const):
    return v_0*np.exp(-t/time_const)

def linear_fit(x, a, b):
    return a*x + b

plt.scatter(ms_time, data_2, label='Data 1')
popt, pcov = curve_fit(exponential, ms_time, data_2, sigma = unc)
plt.plot(ms_time, exponential(ms_time, *popt), label='Fit 1')
plt.legend()
plt.show()

print(popt)