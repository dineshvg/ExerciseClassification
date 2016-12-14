from scipy import signal
import numpy as np

x = np.linspace(0, 25, 44100*5, endpoint=False)
y = np.sin(-x**2/6.0)
print len(y)
f = signal.resample(y, 44100)
print len(f)
xnew = np.linspace(0, 5, 44100, endpoint=False)

import matplotlib.pyplot as plt
plt.plot(x,y,'go')
plt.plot(xnew, f, '.-')
# plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
plt.legend(['data', 'resampled'], loc='best')
plt.show()