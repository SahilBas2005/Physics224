import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.genfromtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/CurrentEx2/mes.csv', delimiter=',', skip_header=1)
voltage = data[:,0]
current = data[:,1]

delta_v_per = 0.0005
delta_v_cout = 0.002

delta_I_per = 0.003
delta_I_cout = 0.001

def chi2(x, y, y_exp, unc, params):
    chi2 = 0
    for i in range(len(y)):
        chi2 += ((y[i] - y_exp[i]) / unc[i])**2
    return chi2 / (len(y) - params)

def uncertainty(per, cout,x):
    return np.round(per*x + cout, 4)

uncertainty_v = uncertainty(delta_v_per, delta_v_cout, voltage)
uncertainty_I = uncertainty(delta_I_per, delta_I_cout, current)

print(uncertainty_I)
print(uncertainty_v)

def model(x, a, b, c):
    return a*x**b + c

popt, pcov = curve_fit(model, voltage, current, sigma=uncertainty_I, absolute_sigma=True)
p_std = np.sqrt(np.diag(pcov))

plt.errorbar(voltage, current, yerr=uncertainty_I, xerr=uncertainty_v, fmt='.', label='data', markersize=3)
plt.plot(voltage, model(voltage, *popt), label='fit')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.legend()
plt.title('Current vs Voltage')
plt.tight_layout()
plt.show()

chi_2 = chi2(voltage, current, model(voltage, *popt), uncertainty_I, 2)
residuals = current - model(voltage, *popt)

#graph residuals

print(f'Chi^2 = {chi_2}')
