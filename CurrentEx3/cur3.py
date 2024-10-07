import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/CurrentEx3/data.csv', delimiter=',', skiprows=1)

voltage = data[:,0]
current = data[:,1]

current = current * 1e-3

def model(x, a, b, c):
    return a*x + np.log(x/b + 1)*c

popt, pcov = curve_fit(model, current, voltage, p0=[220, 1, 39*1e-3])

plt.scatter(current, voltage, label='Data')
plt.plot(current, model(current, *popt), label='Fit')
plt.legend()
plt.show()

a = popt[0]
b = popt[1]
c = popt[2]
print(a, b, c)