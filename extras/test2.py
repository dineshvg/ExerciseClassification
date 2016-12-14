import wave
import struct
import matplotlib.pyplot as plt
import numpy as np
import pylab as pylab

fs = 44100
NFFT = 2048
filename = 'shoulder_exercise.wav'
W = 'coif2'
raw_data = wave.open(filename)
frames = raw_data.readframes(raw_data.getnframes())
# convert binary chunks to short
X = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
X = [float(val) / pow(2, 15) for val in X]

data_points = [x / NFFT for x in X]

fig = plt.figure()
fig.set_size_inches(14, 8)
plt.title('Spectrogram')
plt.xlabel('time (in seconds)')
plt.ylabel('frequency')
data, frequencies, bins, im = plt.specgram(data_points, NFFT=NFFT, Fs=fs)
plt.axis('tight')
fig.colorbar(im).set_label('Intensity [dB]')
plt.show()


# vmin = 20*np.log10(np.max(X)) - 40  # hide anything below -40 dBc
#
# fig, ax = plt.subplots()
# cmap = plt.get_cmap('gist_heat_r')
# min = 20*np.log10(np.max(X)) - 40  # hide anything below -40 dBc
# cmap.set_under(color='k', alpha=None)
#
# NFFT = 256
# pxx,  freq, t, cax = ax.specgram(X(NFFT/2), Fs=44100, mode='magnitude',
#                                  NFFT=NFFT, noverlap=NFFT/2,
#                                  vmin=vmin, cmap=cmap,
#                                  window=plt.window_none)
# fig.colorbar(cax)
#
# print(np.max(pxx)) # should match A




