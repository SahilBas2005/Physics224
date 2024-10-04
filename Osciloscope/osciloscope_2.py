import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Osciloscope/osc_auto.csv', delimiter=',', skiprows=413)
time = data[:,0][data[:,0] < -100.0000E-06]
data_1 = data[:,1][data[:,0] < -100.0000E-06]
data_2 = data[:,2][data[:,0] < -100.0000E-06]

time_offset = -500.0000E-06
time = time - time_offset

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
popt, pcov = curve_fit(exponential, time, data_1, sigma = unc, p0=[1000000000000,10e-4, data_1[0]])
plt.plot(time, exponential(time, *popt), label='Fit 1')
plt.legend()
plt.show()

residuals = data_1 - exponential(time, *popt)
plt.scatter(time, residuals)
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.show()

chi_2 = reduced_chi_squared(time, data_1, exponential(time, *popt), unc, 3)

max_time = time[np.argmax(data_1)]
max_10 = max_time*0.1
max_90 = max_time*0.9

fall_time = np.abs(max_10 - max_90)

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/Osciloscope/osc_auto.csv', delimiter=',', skiprows=1)
time = data[:,0][data[:,0] < -294.5000E-06]
data_1 = data[:,1][data[:,0] < -294.5000E-06]

max_time = time[np.argmax(data_1)]
max_10 = max_time*0.1
max_90 = max_time*0.9

rise_time = np.abs(max_90 - max_10)

print(fall_time)
print(rise_time)
