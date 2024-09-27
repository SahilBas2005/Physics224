import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

temp_data = np.genfromtxt('/Users/hachemfattouh/Desktop/random files 5 the will to survive/Physics224-1/FittingEx3/co2_temp_1000_cleanend.csv', delimiter=',')

year = temp_data[:,0]
temp_mean = temp_data[:,1]
temp_std = temp_data[:,2]
conc_mean = temp_data[:,3]
conc_std = temp_data[:,4]

industrial_rev = 1760
#histogram for temp pre and post ind rev

pre_ind_rev = temp_mean[year < industrial_rev]
post_ind_rev = temp_mean[year >= industrial_rev]
pi_mean = np.mean(pre_ind_rev)
pi_std = np.std(pre_ind_rev)
post_mean = np.mean(post_ind_rev)
post_std = np.std(post_ind_rev)

plt.hist(pre_ind_rev, bins=100, alpha=0.5, label='Pre Industrial Revolution', density=True)
plt.hist(post_ind_rev, bins=100, alpha=0.5, label='Post Industrial Revolution', density=True)
plt.axvline(pi_mean, color='r', linestyle='solid', linewidth=1, label='Pre Industrial Mean')
plt.axvline(post_mean, color='b', linestyle='solid', linewidth=1, label='Post Industrial Mean')
plt.axvline(pi_mean + pi_std, color='r', linestyle='--', linewidth=1, label='Pre Industrial Std')
plt.axvline(pi_mean - pi_std, color='r', linestyle='--', linewidth=1)
plt.axvline(post_mean + post_std, color='b', linestyle='dotted', linewidth=1, label='Post Industrial Std')
plt.axvline(post_mean - post_std, color='b', linestyle='dotted', linewidth=1)
plt.legend()
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Temperature Distribution')
plt.show()

cos_pre_ind_rev = conc_mean[year < industrial_rev]
cos_post_ind_rev = conc_mean[year >= industrial_rev]
pi_mean = np.mean(cos_pre_ind_rev)
pi_std = np.std(cos_pre_ind_rev)
post_mean = np.mean(cos_post_ind_rev)
post_std = np.std(cos_post_ind_rev)

plt.hist(cos_pre_ind_rev, bins=100, alpha=0.5, label='Pre Industrial Revolution', density=True)
plt.hist(cos_post_ind_rev, bins=100, alpha=0.5, label='Post Industrial Revolution', density=True)
plt.axvline(pi_mean, color='r', linestyle='solid', linewidth=1, label='Pre Industrial Mean')
plt.axvline(post_mean, color='b', linestyle='solid', linewidth=1, label='Post Industrial Mean')
plt.axvline(pi_mean + pi_std, color='r', linestyle='--', linewidth=1, label='Pre Industrial Std')
plt.axvline(pi_mean - pi_std, color='r', linestyle='--', linewidth=1)
plt.axvline(post_mean + post_std, color='b', linestyle='dotted', linewidth=1, label='Post Industrial Std')
plt.axvline(post_mean - post_std, color='b', linestyle='dotted', linewidth=1)
plt.legend()
plt.xlabel('CO2 Concentration')
plt.ylabel('Frequency')
plt.title('CO2 Concentration Distribution')
plt.show()

