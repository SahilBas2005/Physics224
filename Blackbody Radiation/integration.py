"""
Integrating intensity over position from the Blackbody Radition lab
"""
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from numpy.core.fromnumeric import argmax

# Load data
data = np.loadtxt('data new/data 8.csv', delimiter='\t', skiprows=2)
position = data[:,0]
intensity = data[:,1]

# Remove noise
noise = intensity[0:100]
noise_value = numpy.max(noise)
intensity -= noise_value

# Define position bounds of first peak
lower_bound = argmax(intensity > 0.05)
upper_bound = argmax(intensity[lower_bound + 10:] < 0.05) + lower_bound + 10


# Plot data
plt.scatter(position, intensity, label='Data')
plt.ylabel('Position')
plt.xlabel('Intensity')
plt.legend()
plt.show()

#Integrate
print(scipy.integrate.cumulative_trapezoid(intensity[lower_bound:upper_bound], position[lower_bound:upper_bound]))

# Get uncertainty by moving bounds of integral
lower_bound -= 5
upper_bound += 5
print(scipy.integrate.cumulative_trapezoid(intensity[lower_bound:upper_bound], position[lower_bound:upper_bound]))

lower_bound += 10
upper_bound -= 10
print(scipy.integrate.cumulative_trapezoid(intensity[lower_bound:upper_bound], position[lower_bound:upper_bound]))
