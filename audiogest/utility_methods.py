from __future__ import division
import wave
import struct
from numpy import linspace
import scipy.signal as sig
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks


def get_audio_signal(filename):
    raw_data = wave.open(filename)
    print raw_data.getmarkers()
    frames = raw_data.readframes(raw_data.getnframes())
    audio_signal = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
    audio_signal = [float(val) / pow(2, 15) for val in audio_signal]  # normalize the signal to 0-1
    return audio_signal


def get_time_axis(x):
    return linspace(1, len(x), len(x)) / 44100


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    y = sig.lfilter(b, a, data)
    return y


# Source : http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?i=1

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i + 1]], axis=1)
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i + 1]])]
    return newspec, freqs


def plotstft(samples, fs, binsize=2 ** 14, plotpath=None, colormap="jet"):
    # samplerate, samples = wav.read(audiopath)
    samplerate = fs
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) + 10e-6)  # amplitude to decibel

    normal_ims = []
    segmented_ims = []
    thresholded_ims = []

    # 30% more than the mean of the mean
    thresh = 60  # Based on magnitude_analysis
    gaussian_thresh = 500  # not used
    # Frequency normalization
    for mag_list in ims:
        # super_threshold_indices = mag_list < thresh
        # mag_list[super_threshold_indices] = 0
        max_val = max(mag_list)
        # mag_list = [x/max_val for x in mag_list]
        mag_list = [(10 ** (x - max_val)) ** 0.05 for x in mag_list]
        normal_ims.append(mag_list)

    # Audio Signal Segmentation - not required
    for index in range(len(normal_ims)):
        if index != 0:
            old_frame = np.array(ims[index - 1])
            current_frame = np.array(ims[index])
            gradient_change = (current_frame - old_frame) ** 2
            # max_grad = max(gradient_change)
            # gradient_change = [x / max_grad for x in gradient_change]
            segmented_ims.append(gradient_change)

    # Gaussian smoothing - not required
    sigma = 0.1
    filtered_ims = gaussian_filter(segmented_ims, sigma)
    for list in filtered_ims:
        super_threshold_indices = list > gaussian_thresh
        list[super_threshold_indices] = 1000
        thresholded_ims.append(list)

    # Plotting
    timebins, freqbins = np.shape(thresholded_ims)
    fig = plt.figure(1, figsize=(15, 7.5))
    plt.imshow(np.transpose(thresholded_ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.title('Pre-processing')
    plt.xlabel("time (in seconds)")
    plt.ylabel("frequency (in Hz)")
    plt.xlim([0, timebins - 1])
    plt.ylim([0, freqbins])
    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in
                       ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    ## Plot timestamps on the imshow

    ## MOTO-E

    # filename = 'signals/Suhas/handscissors/fast/timestamps.txt'
    # filename = 'signals/Suhas/handscissors/slow/timestamps.txt'
    # filename = 'signals/Suhas/shoulder/fast/1_38_50_31_rec.wav'
    # filename = 'signals/Suhas/pushups/timestamps.txt'

    filename = 'signals/Dinesh/shoulder/moto/fast/2_26_42_695_rec.wav'
    # filename = 'signals/Dinesh/handscissors/fast/timestamps.txt'
    # filename = 'signals/Dinesh/handscissors/slow/1_28_1_770_rec.wav'

    # NEXUS

    # filename = 'signals/Suhas/handscissors/nexus/12_49_28_225.txt'
    # filename = 'signals/Suhas/shoulder/nexus/13_4_20_713.txt'
    # filename = 'signals/Suhas/squats/nexus/13_16_59_25.txt'
    # filename = 'signals/Suhas/pushups/nexus/13_26_15_63.txt'
    #
    #
    # filename = 'signals/Dinesh/handscissors/nexus/12_55_46_590.txt'
    # filename = 'signals/Dinesh/handscissors/nexus/12_59_31_388.txt' ##fast
    # filename = 'signals/Dinesh/shoulder/nexus/13_10_20_437.txt'
    # filename = 'signals/Dinesh/squats/nexus/13_21_36_883.txt'
    # filename = 'signals/Dinesh/pushups/nexus/13_33_16_218.txt'

    # startTime, endTime = get_time_stamps(filename, timebins, binsize, samplerate, len(samples))
    # for xc in startTime:
    #     plt.axvline(x=xc, color='m', linestyle=':', linewidth=2.5)
    #     ax = fig.add_subplot(111)
    #     ax.annotate(xc, xy=(0.50,0), xytext=(xc, -.03),
    #                 arrowprops=dict(facecolor='k', headlength=0.5, shrink=0.5, width=0.5))
    # for xc in endTime:
    #     plt.axvline(x=xc, color='w', linestyle=':', linewidth=2.5)
    #     ax = fig.add_subplot(111)
    #     ax.annotate(xc, xy=(0.50,0), xytext=(xc, -.03),
    #                 arrowprops=dict(facecolor='k', headlength=0.5, shrink=0.5, width=0.5))
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        # plt.ion()
        plt.show()
        # plt.pause(0.05)
        # input("Press [enter] to continue.")
        # plt.savefig('stft_signal.png', format='png', dpi=1200)
    # plt.clf()
    return ims


def plot_specgram(x, nfft, fs, colormap="jet"):
    samplerate = fs
    s = stft(x, nfft)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) + 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    plt.figure(1, figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.title('Spectrogram')
    plt.xlabel("time (in seconds)")
    plt.ylabel("frequency (in Hz)")
    plt.xlim([0, timebins - 1])
    plt.ylim([0, freqbins])
    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in
                       ((xlocs * len(x) / timebins) + (0.5 * nfft)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    plt.show()


def get_time_stamps(filepath, timebins, binsize, samplerate, sample_len):
    activity_start = []
    activity_end = []
    data = open(filepath).read()
    rows = data.split('\n')
    for r in rows:
        val = r.split('\t')
        if len(val[0]) != 0:
            activity_start.append(get_xLoc(float(val[0]), timebins, binsize, samplerate, sample_len))
            activity_end.append(get_xLoc(float(val[1]), timebins, binsize, samplerate, sample_len))
    return activity_start \
        , activity_end


def get_xLoc(timestamp, timebins, binsize, samplerate, sample_len):
    n1 = (timebins * binsize) / 2
    n2 = samplerate * timestamp * timebins
    xlocation = (n1 + n2) / sample_len
    return xlocation

def slidingWindow(sequence, winSize, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1

    # Do the work
    for i in range(0, numOfChunks * step, step):
        print sequence[i:i + winSize]
