import wave
from itertools import islice
import matplotlib.pyplot as plt
import struct

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

Fs=44100
filename = 'shoulder_exercise.wav'
W = 'coif2'
raw_data = wave.open(filename)
frames = raw_data.readframes(raw_data.getnframes())
# convert binary chunks to short
X = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
X = [float(val) / pow(2, 15) for val in X]


Pxx, freqs, bins, im = plt.specgram(X,
                                    NFFT=1028,
                                    Fs=44100,
                                    noverlap=int(128 * 0.5),
                                    cmap=plt.cm.hsv)
#plt.show()
#print(len(Pxx))
#print(len(freqs))
#print(len(bins))
result = tuple(islice(Pxx, 2))
it = iter(Pxx)
for elem in it:
    result = result[1:] + (elem,)
    print(result)
