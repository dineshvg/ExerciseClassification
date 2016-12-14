## Label the data based on the time labels obtained for the signal.

from matplotlib import pyplot as plt

ims = [[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]
timestamps = [0.22058956,0.32058956]
fig = plt.figure(1, figsize=(15, 7.5))
plt.imshow(ims, origin="lower", aspect="auto", cmap="jet", interpolation="none")
plt.colorbar()
plt.xlim([0, 1])
plt.ylim([0, 1])
for xc in timestamps:
    plt.axvline(x=xc,color='w', linestyle='--',linewidth=2.5)
    ax = fig.add_subplot(111)
    ax.annotate(round(xc,2), xy=(xc, 0), xytext=(xc, -.03),
                arrowprops=dict(facecolor='black',headlength=0.5, shrink=0.5,width=0.5))
plt.title('label data in FFT plot of the audio signal')
plt.xlabel("time (in seconds)")
plt.ylabel("frequency (in Hz)")
plt.show()

