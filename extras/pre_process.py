import wave
import struct
import matplotlib.pyplot as plt
from numpy import linspace
from scipy.signal import butter, lfilter

from stft import plotstft

filename = '161011173921to161011173928_rcrd.wav'
raw_data = wave.open(filename)
frames = raw_data.readframes(raw_data.getnframes())
# convert binary chunks to short
audio_signal = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
audio_signal = [float(val) / pow(2, 15) for val in audio_signal]

def get_time_axis(x) :
    return linspace(1, len(x), len(x)) / 44100

# show axis in seconds
t = get_time_axis(audio_signal)
# Plot the original signal showing the raw amplitude vs the time axis
plt.figure(1)
plt.plot(t, audio_signal, linewidth=0.5)
plt.title('Raw Signal')
plt.xlabel('time(seconds)')
plt.ylabel('Amplitude of the signal')
plt.grid(True)
plt.show()
#plt.savefig('original_signal.svg', format='svg', dpi=1000)
#plt.close()

# Sample rate and desired cutoff frequencies (in Hz).
fs = 44100.0
lowcut = 18000.0
highcut = 22000.0

# Function to create a butter-worth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

y = butter_bandpass_filter(audio_signal, lowcut, highcut, fs, order=3)
t = get_time_axis(y)
plt.figure(2)
plt.plot(t, y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
#plt.show()
plt.savefig('filtered_signal.png')
plt.close()


signal = audio_signal[1 * fs:9 * fs]
window = sig.hann(2048)

# Resizing the signal
round_ = round(len(signal) / index)
if round_ != 0:
    signal = signal[0:(len(signal) - (len(signal) - (index * round_)))]

# window the signal
windowed_frames = []
signal_frames = np.array_split(signal, round_)
for i in range(len(signal_frames)):
    windowed_frames.append(signal_frames[i]*window)
plotstft(y)
# Wavelet denoising : https://blancosilva.wordpress.com/teaching/mathematical-imaging/denoising-wavelet-thresholding/


