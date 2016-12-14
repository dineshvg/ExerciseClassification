import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

sigma = 4
# x = 2
# y = 2
#
# val = ((x**2) + (y**2))/(2 * (std_dev**2))
#
# part1 = 1/(2*np.pi*(std_dev**2))
# part2 = np.exp(-val)
# print part2
arr=np.zeros((20,20))
arr[0,:]=3
arr[0,0]=20
arr[19,19]=30
arr[10:12,10:12]=10

print arr
filtered_arr=gaussian_filter(arr, sigma)
plt.imshow(filtered_arr)
plt.show()
