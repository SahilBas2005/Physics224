"""
Graphs the voltage data from the oscilloscope and calculates rise and fall times
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Define chi squared function
def reduced_chi_squared(x, y, y_exp, unc, params):
    chi_squared = 0
    for i in range(len(y_exp)):
        chi_squared += ((y[i] - y_exp[i]) / unc[i]) ** 2

    return chi_squared / (len(y_exp) - params)

# Define model
def exponential(t, v_0, time_const, b):
    return v_0*np.exp(-t/time_const) + b

def get_prediction_fall(threshold, independent_range, model_data):
    for i in range(len(model_data)):
        if model_data[i] < threshold:
            return independent_range[i]

def get_prediction_rise(threshold, independent_range, model_data):
    for i in range(len(model_data)):
        if model_data[i] > threshold:
            return independent_range[i]

def stats(t, popt, pstd, unc, model_data, params):
    # Print statistical values
    # print(f'chi^2 = {reduced_chi_squared(t, data_1_rise, model_data, unc, params)}')
    print(f'v0 =  {popt[0]} u(v0) = {pstd[0]}, time_const = {popt[1]} u(time_const) = {pstd[1]}, b = {popt[2]} u(b) = {pstd[2]}')

# Load data
data = np.loadtxt('osc_auto.csv', delimiter=',', skiprows=1)

unc_raw = np.sqrt(20) * 10E-06
time_offset = -500.0000E-06

# Split data into partitions
time = data[:,0] - time_offset
data_1 = data[:,1]
data_2 = data[:,2]
unc = np.full(len(data_1), unc_raw)

# First Rise
time_rise = data[:,0][data[:,0] < -294.5000E-06] - time_offset
data_1_rise = data[:,1][data[:,0] < -294.5000E-06]
data_2_fall = data[:,2][data[:,0] < -294.5000E-06]
unc_rise = np.full(len(data_1_rise), unc_raw)

# First Fall
data = np.loadtxt('osc_auto.csv', delimiter=',', skiprows=413)
time_fall = data[:,0][data[:,0] < -100.0000E-06] - time_offset
data_1_fall = data[:,1][data[:,0] < -100.0000E-06]
data_2_rise = data[:,2][data[:,0] < -100.0000E-06]
unc_fall = np.full(len(data_1_fall), unc_raw)

# Generate models
popt_1_rise, pcov_1_rise = curve_fit(exponential, time_rise, data_1_rise, sigma = unc_rise, p0=[1000000000000,10e-4, data_1[0]])
pstd_1_rise = np.sqrt(np.diag(pcov_1_rise))
popt_1_fall, pcov_1_fall = curve_fit(exponential, time_fall, data_1_fall, sigma = unc_fall, p0=[1000000000000,10e-4, data_1[0]])
pstd_1_fall = np.sqrt(np.diag(pcov_1_fall))
popt_2_rise, pcov_2_rise = curve_fit(exponential, time_fall, data_2_rise, sigma = unc_fall, p0=[1000000000000,10e-4, data_1[0]])
pstd_2_rise = np.sqrt(np.diag(pcov_2_rise))
popt_2_fall, pcov_2_fall = curve_fit(exponential, time_rise, data_2_fall, sigma = unc_rise, p0=[1000000000000,10e-4, data_1[0]])
pstd_2_fall = np.sqrt(np.diag(pcov_2_fall))

# Plot data with model
plt.title("Voltages over Time")
plt.errorbar(time, data_1, yerr=unc, marker='o', ls='', lw=2, label='Channel 1')
plt.errorbar(time, data_2, yerr=unc, marker='o', ls='', lw=2, label='Channel 2')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.show()

# Create figure
plt.figure(figsize=(4, 3))

# Plot first rise with model (channel 1)
plt.subplot(2, 2, 1)
model_data = exponential(time_rise, *popt_1_rise)
plt.title("Voltages over Time (First Rise)")
plt.errorbar(time_rise, data_1_rise, yerr=unc_rise, marker='o', ls='', lw=2, label='Channel 1')
plt.plot(time_rise, model_data, label='Exponential Regression')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Calculate rise time
data_range = max(model_data) - min(model_data)
voltage_10 = min(model_data) + 0.1 * data_range
voltage_90 = min(model_data) + 0.9 * data_range
rise_1_time = get_prediction_rise(voltage_90, time_rise, model_data) - get_prediction_rise(voltage_10, time_rise, model_data)

# Print stats
print('Channel 1 Rise')
print(rise_1_time)
stats(time_rise, popt_1_rise, pstd_1_rise, unc_rise, model_data, 3)

# Plot first rise with model (channel 2)
plt.subplot(2, 2, 2)
model_data = exponential(time_fall, *popt_2_rise)
plt.title("Voltages over Time (First Rise)")
plt.errorbar(time_fall, data_2_rise, yerr=unc_fall, marker='o', ls='', lw=2, label='Channel 2')
plt.plot(time_fall, model_data, label='Exponential Regression')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Calculate rise time
data_range = max(model_data) - min(model_data)
voltage_10 = min(model_data) + 0.1 * data_range
voltage_90 = min(model_data) + 0.9 * data_range
rise_2_time = get_prediction_rise(voltage_90, time_fall, model_data) - get_prediction_rise(voltage_10, time_fall, model_data)

# Print stats
print('Channel 2 Rise')
print(rise_2_time)
stats(time_fall, popt_2_rise, pstd_2_rise, unc_fall, model_data, 3)

# Plot first fall with model (channel 1)
plt.subplot(2, 2, 3)
model_data = exponential(time_fall, *popt_1_fall)
plt.title("Voltages over Time (First Fall)")
plt.errorbar(time_fall, data_1_fall, yerr=unc_fall, marker='o', ls='', lw=2, label='Channel 1')
plt.plot(time_fall, model_data, label='Exponential Regression')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Calculate fall time
data_range = max(model_data) - min(model_data)
voltage_10 = min(model_data) + 0.1 * data_range
voltage_90 = min(model_data) + 0.9 * data_range
fall_1_time = get_prediction_fall(voltage_10, time_fall, model_data) - get_prediction_fall(voltage_90, time_fall, model_data)

# Print stats
print('Channel 1 Fall')
print(fall_1_time)
stats(time_fall, popt_1_fall, pstd_1_fall, unc_fall, model_data, 3)

# Plot first fall with model (channel 2)
plt.subplot(2, 2, 4)
model_data = exponential(time_rise, *popt_2_fall)
plt.title("Voltages over Time (First Fall)")
plt.errorbar(time_rise, data_2_fall, yerr=unc_rise, marker='o', ls='', lw=2, label='Channel 2')
plt.plot(time_rise, model_data, label='Exponential Regression')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Calculate fall time
data_range = max(model_data) - min(model_data)
voltage_10 = min(model_data) + 0.1 * data_range
voltage_90 = min(model_data) + 0.9 * data_range
fall_2_time = get_prediction_fall(voltage_10, time_rise, model_data) - get_prediction_fall(voltage_90, time_rise, model_data)


# Print stats
print('Channel 2 Fall')
print(fall_2_time)
stats(time_rise, popt_2_fall, pstd_2_fall, unc_rise, model_data, 3)

plt.show()

# Plot residuals
plt.title("Residuals for First Rise on Channel 1")
residuals = data_1_rise - exponential(time_rise, *popt_1_rise)
plt.scatter(time_rise, residuals)
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.show()
