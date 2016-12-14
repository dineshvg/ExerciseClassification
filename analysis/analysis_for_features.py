import matplotlib.pyplot as plt
from audiogest.utility_methods import get_audio_signal

filename = '/home/dinesh/PycharmProjects/MasterThesis/signals/0_36_11_216_rec.wav'
signal = get_audio_signal(filename)

print len(signal)

plt.figure('Signal in time domain')
plt.title('Signal in Time domain')
plt.plot(signal)
plt.show()

