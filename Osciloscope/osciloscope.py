import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('Osciloscope/osciloscope.csv', delimiter=',', skiprows=1)

time = data[:,0]
data_1 = data[:,1]
data_2 = data[:,2]
unc = data[:,3]

# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y) - params)


# Define model
def exponential(t, v_0, time_const):
    return v_0**(-t/time_const)
