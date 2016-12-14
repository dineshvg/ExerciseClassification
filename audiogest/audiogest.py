import wave
import struct
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
from numpy import linspace
import numpy as np
from scipy.constants import find
from scipy.fftpack import rfft
from utility_methods import plot_specgram, plotstft, get_audio_signal, butter_bandpass_filter


########################################################################################################################

# extract the audio signal
filename = 'signals/shoulder_exercise.wav'
# filename = 'signals/hand_exercises.wav'

# filename = 'signals/Dinesh/pushups/hw/set2/2_6_56_102_rec.wav'

## MOTO-E

# filename = 'signals/Suhas/handscissors/fast/1_22_56_796_rec.wav'
# filename = 'signals/Suhas/handscissors/slow/1_18_20_877_rec.wav'
# filename = 'signals/Suhas/shoulder/fast/1_38_50_31_rec.wav'
# filename = 'signals/Suhas/pushups/3_57_42_792_rec.wav'

filename = 'signals/Dinesh/shoulder/moto/fast/2_26_42_695_rec.wav'
# filename = 'signals/Dinesh/handscissors/fast/1_30_49_112_rec.wav'
# filename = 'signals/Dinesh/handscissors/slow/1_28_1_770_rec.wav'

# filename = 'signals/3_46_15_103_rec.wav'
# filename = 'signals/16_32_46_414_rec.wav'
# filename = 'signals/5_53_56_155_rec.wav'

# NEXUS

# filename = 'signals/Suhas/handscissors/nexus/12_49_28_225_rec.wav'
# filename = 'signals/Suhas/shoulder/nexus/13_4_20_713_rec.wav'
# filename = 'signals/Suhas/squats/nexus/13_16_59_25_rec.wav'
# filename = 'signals/Suhas/pushups/nexus/13_26_15_63_rec.wav'


# filename = 'signals/Dinesh/handscissors/nexus/12_55_46_590_rec.wav'
# filename = 'signals/Dinesh/handscissors/nexus/12_59_31_388_rec.wav' ##fast
# filename = 'signals/Dinesh/shoulder/nexus/13_10_20_437_rec.wav'
# filename = 'signals/Dinesh/squats/nexus/13_21_36_883_rec.wav'
# filename = 'signals/Dinesh/pushups/nexus/13_33_16_218_rec.wav'


fs = 44100
nfft = 4096

audio_signal = get_audio_signal(filename)

# apply the band-pass filter

signal = butter_bandpass_filter(audio_signal, 18000.0, 22000.0, fs, order=3)

# plt.plot(signal)
# plt.show()

# cut the signal for 9 seconds exactly
# signal = signal[1 * fs:9 * fs]

# Checking the spectrogram before windowing
# plot_specgram(signal, nfft, fs)

# Spectrogram after applying window to it
plotstft(signal, fs)

