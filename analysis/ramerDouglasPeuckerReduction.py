from rdp import rdp
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from audiogest.utility_methods import get_audio_signal

arr = [44, 95, 26, 91, 22, 90, 21, 90,
    19, 89, 17, 89, 15, 87, 15, 86, 16, 85,
    20, 83, 26, 81, 28, 80, 30, 79, 32, 74,
    32, 72, 33, 71, 34, 70, 38, 68, 43, 66,
    49, 64, 52, 63, 52, 62, 53, 59, 54, 57,
    56, 56, 57, 56, 58, 56, 59, 56, 60, 56,
    61, 55, 61, 55, 63, 55, 64, 55, 65, 54,
    67, 54, 68, 54, 76, 53, 82, 52, 84, 52,
    87, 51, 91, 51, 93, 51, 95, 51, 98, 50,
    105, 50, 113, 49, 120, 48, 127, 48, 131, 47,
    134, 47, 137, 47, 139, 47, 140, 47, 142, 47,
    145, 46, 148, 46, 152, 46, 154, 46, 155, 46,
    159, 46, 160, 46, 165, 46, 168, 46, 169, 45,
    171, 45, 173, 45, 176, 45, 182, 45, 190, 44,
    204, 43, 204, 43, 207, 43, 215, 40, 215, 38,
215, 37, 200, 37, 195, 41]
filename = '/home/dinesh/PycharmProjects/MasterThesis/signals/shoulder_exercise.wav'
# audio_signal = get_audio_signal(filename)
# print len(audio_signal)
# signal = np.array(audio_signal).reshape(len(audio_signal)/2, 2)
plt.plot(arr)
print len(arr)
nice_line = np.array(arr).reshape(77, 2)
mask = rdp(nice_line, epsilon=0.5, algo='iter')
arr2 = np.ravel(mask).T
plt.plot(arr2)
print len(mask)
plt.show()

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
    return n / d

def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)