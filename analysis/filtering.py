import struct
import wave

import matplotlib.pyplot as plt
import pywt

from extras.pre_process import get_time_axis

noiseSigma = 10.0
threshold = noiseSigma*4

thresh = lambda x: pywt.thresholding.soft(x,threshold)

def denoise(X,W,noiseSigma):
    #r = int(np.floor(np.log2(X.shape[0])))
    WC = pywt.wavedec(X,W,level=5)
#    threshold = noiseSigma * np.sqrt(2*np.log2(X.size))
 #   NWC = map(threshold, WC)
    return pywt.waverec(WC, W)

DW = ['sym15','db6', 'bior2.8', 'coif2']

filename = 'shoulder_exercise.wav'
raw_data = wave.open(filename)
frames = raw_data.readframes(raw_data.getnframes())
# convert binary chunks to short
audio_signal = struct.unpack("%ih" % (raw_data.getnframes() * raw_data.getnchannels()), frames)
audio_signal = [float(val) / pow(2, 15) for val in audio_signal]

DLs = {}
for dw in DW:
    DLs[dw] = denoise(audio_signal, dw, noiseSigma)

plt.figure('Wavelet denoising examples')

plt.subplot(231)
plt.title('Lena with noise')
plt.plot(get_time_axis(audio_signal),audio_signal)

plt.subplot(232)
plt.title('Denoised by Sym16')
plt.plot(get_time_axis(audio_signal), DLs['sym15'])

plt.subplot(233)
plt.title('Denoised by Daub6')
plt.plot(get_time_axis(audio_signal), DLs['db6'])

plt.subplot(234)
plt.title('Denoised by Bior2.8')
plt.plot(get_time_axis(audio_signal),DLs['bior2.8'])

plt.subplot(235)
plt.title('Denoised by Coif2')
plt.plot(DLs['coif2'])

plt.tight_layout()
plt.gray()
plt.show()