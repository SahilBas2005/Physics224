"""
Curve fitting currents and voltages from a diode circuit.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


def model(I, R, I_s, c, d):
    return I*R + np.log(I/I_s + 1)*c + d


def uncertainty(per, cout, x):
    return np.round(per*x + cout, 4)


data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# Uncertainty constants
delta_v_per = 0.0005
delta_v_cout = 0.002
delta_I_per = 0.003
delta_I_cout = 0.00001

voltage = data[:,0]
current = data[:,1]
current = current * 1e-3

uncertainty_v = uncertainty(delta_v_per, delta_v_cout, voltage)
uncertainty_I = uncertainty(delta_I_per, delta_I_cout, current)

popt, pcov = curve_fit(model, current, voltage, p0=[220, 1e-3, 39*1e-3, 0], maxfev=10000)
pstd = np.sqrt(np.diag(pcov))

# Plot models on linear scale
plt.errorbar(current, voltage, yerr=uncertainty_v, xerr=uncertainty_I, fmt='.', label='Measured Voltage', markersize=3)
plt.plot(current, model(current, *popt), label='Shockley Equation Fit', color='red')
plt.ylabel('Voltage (V)')
plt.xlabel('Current (A)')
plt.legend()
plt.title('Voltage vs Current')
plt.tight_layout()
plt.show()

# Plot residuals
plt.title('Residuals from Voltage over Current')
plt.axhline(color='grey', linestyle='--')
plt.errorbar(current, voltage - model(current, *popt), yerr=uncertainty_v, xerr=uncertainty_I, marker='o', ls='', lw=2, label='Residuals for Shockley Fit', color='red')
plt.ylabel('Voltage (V)')
plt.xlabel('Current (A)')
plt.legend()
plt.show()

# Print statistical values
chi_2 = reduced_chi_squared(voltage, current, model(current, *popt), uncertainty_v, 4)

print(f'chi^2 = {chi_2}')
print(f'R =  {popt[0]} u(R) = {pstd[0]}, I_s = {popt[1]} u(I_s) = {pstd[1]}')
print(f'c =  {popt[2]} u(c) = {pstd[2]}, V_0 = {popt[3]} u(V_0) = {pstd[3]}')
