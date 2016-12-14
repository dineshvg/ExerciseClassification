import struct
import wave

import matplotlib.pyplot as plt
import pywt
import numpy as np
from numpy import linspace, log10, where, shape, transpose
from pandas.algos import int16, float32
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import iterate_structure, generate_binary_structure, binary_erosion
from peakutils.peak import indexes
from extras.stft import logscale_spec, stft

######################################################################
# Number of cells around an amplitude peak in the spectrogram
PEAK_NEIGHBORHOOD_SIZE = 10

######################################################################
# Wavelet being used for analysis
W = 'coif2'

######################################################################
# Minimum amplitude in spectrogram in order to be considered a peak.
DEFAULT_AMP_MIN = 5

def get_time_axis(x) :
    return linspace(1, len(x), len(x)) / 44100

######################################################################
# Peak detection on the spectroram

def getPeaks(arr2D,amp_min=DEFAULT_AMP_MIN) :
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)
    # find local maxima using our fliter shape
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    # Boolean mask of arr2D with True at peaks
    detected_peaks = local_max - eroded_background
    # extract peaks
    amps = arr2D[detected_peaks]
    j, i = where(detected_peaks)
    # filter peaks
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

    # get indices for frequency and time
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]

    if plt:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(time_idx, frequency_idx)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return zip(frequency_idx, time_idx)

filename = '/home/dinesh/PycharmProjects/MasterThesis/signals/Suhas/shoulder/fast/1_38_50_31_rec.wav'
W = 'coif2'
raw_data = wave.open(filename)
frames = raw_data.readframes(raw_data.getnframes())
# convert binary chunks to short
X = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
X = [float(val) / pow(2, 15) for val in X]
WC = pywt.wavedec(X,W,level=6)
NWC = pywt.waverec(WC, W)
# print(WC)


plt.figure('Wavelet transform by coif2')

plt.subplot(231)
plt.title('Level 1')
plt.plot(get_time_axis(WC[1]), WC[1])

plt.subplot(232)
plt.title('Level 2')
plt.plot(get_time_axis(WC[2]), WC[2])

plt.subplot(233)
plt.title('Level 3')
plt.plot(get_time_axis(WC[3]), WC[3])

plt.subplot(234)
plt.title('Level 4')
plt.plot(get_time_axis(WC[4]), WC[4])

plt.subplot(235)
plt.title('Level 5')
plt.plot(get_time_axis(WC[5]), WC[5])

plt.subplot(236)
plt.title('Level 6')
plt.plot(get_time_axis(WC[6]), WC[6])
# indexes = indexes(np.array(vector), thres=7.0/max(vector), min_dist=2)
plt.show()

# getPeaks(WC[5])

def plotstft(samples, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate = 44100
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*log10(abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = shape(ims)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(transpose(ims), origin="lower", aspect="auto", cmap=colormap,
    interpolation="none")
    plt.colorbar()
    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    xlocs = float32(linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in
    ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = int16(round(linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()
        #plt.savefig('stft_signal_without_filter.png', format='png', dpi=800)
    plt.clf()

