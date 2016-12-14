import struct
import wave

from scipy.signal import find_peaks_cwt, np

from audiogest.utility_methods import butter_bandpass_filter

fs= 44100
# get the file
filename = 'hand_exercises.wav'
# extract the audio signal
raw_data = wave.open(filename)
frames = raw_data.readframes(raw_data.getnframes())
audio_signal = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
audio_signal = [float(val) / pow(2, 15) for val in audio_signal]
# apply the band-pass filter
signal = butter_bandpass_filter(audio_signal, 18000.0, 22000.0, fs, order=3)
signal = signal[1 * fs:9 * fs]
# Peak detection with SndObj
indexes = find_peaks_cwt(signal, np.arange(1, 550))
print(indexes)
