from audiogest.utility_methods import get_audio_signal
from scipy import signal
import matplotlib.pyplot as plt

Fs = 44100
filename = '/home/dinesh/PycharmProjects/MasterThesis/signals/shoulder_exercise.wav'
x = get_audio_signal(filename)
x_start = int(0.3*Fs)
x_end = int(0.3*Fs)
x = x[x_start:len(x)]
f, t, Sxx = signal.spectrogram(x, Fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()