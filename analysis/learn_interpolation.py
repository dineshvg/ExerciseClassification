import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# t = [0.5, 1, 3, 6]
# f = [0.6065,    0.3679,    0.0498, 0.5434]
# # plt.figure(1, figsize=(10, 7.5))
# # plt.xlabel('t')
# # plt.ylabel('f(t)')
# # plt.plot(t,f,'o')
# # plt.show()
#
# g = interp1d(t, f,kind='linear',fill_value='extrapolate')
# tnew = np.arange(0.5, 70, 0.01)
# fnew = g(tnew)
# plt.figure(2, figsize=(10, 7.5))
# plt.plot(t, f, 'o', tnew, fnew, '-')
# plt.show()

# a = np.array([1,2,6,2,1,7])
# print a
# R = 3
# a.reshape(-1, R)
#
# a.reshape((-1, R)).mean(axis=0.5)
# print a

from scipy.interpolate import interp1d


x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')


xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

