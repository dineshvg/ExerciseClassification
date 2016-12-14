from scipy.signal import butter, freqz
import matplotlib.pyplot as plt
from numpy import pi

fs = 44100.0
lowcut = 18000.0
highcut = 22000.0
order = 3

nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = butter(order, [low, high], btype='band')
w, h = freqz(b, a, worN=2000)
yq = 0.5 * fs

plt.figure(2)
plt.plot((fs * 0.5 / pi) * w, abs(h), label="order = %d" % order)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
plt.show()