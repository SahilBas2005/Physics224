import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.genfromtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/CurrentEx1/measurements.csv', delimiter=',', skip_header=1)
voltage = data[:,0]
current = data[:,1]

delta_v_per = 0.0005
delta_v_cout = 0.002

delta_I_per = 0.002
delta_I_cout = 0.005

def uncertainty(per, cout,x):
    return np.round(per*x + cout, 4)

uncertainty_v = uncertainty(delta_v_per, delta_v_cout, voltage)
uncertainty_I = uncertainty(delta_I_per, delta_I_cout, current)

def model(i, r, b):
    return i*r + b

popt, pcov = curve_fit(model, voltage, current, sigma=uncertainty_I, absolute_sigma=True)
p_std = np.sqrt(np.diag(pcov))

plt.errorbar(voltage, current, yerr=uncertainty_I, xerr=uncertainty_v, fmt='.', label='data', markersize=3)
plt.plot(voltage, model(voltage, *popt), label='fit')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.legend()
plt.title('Current vs Voltage')
plt.tight_layout()
plt.show()

residuals = current - model(voltage, *popt)
chi22 = np.sum((residuals/uncertainty_I)**2) / (len(voltage) - 2)

print(f'Chi^2 = {chi22}')
print(f'R = {1/popt[0]}')
print(f'uncertainty in R = {p_std[0]}')
print(f'B = {popt[1]}')
print(f'uncertainty in B = {p_std[1]}')
