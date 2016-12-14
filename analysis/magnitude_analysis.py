from __future__ import division
import numpy as np
import csv
from matplotlib import pyplot as plt

meanVals = []
maxVals = []
minVals = []
with open('magnitude_shoulder.txt') as inputfile:
    for line in inputfile:
        list = line.strip().split(';')
        max_ = float(list[0])
        min_ = float(list[1])
        if max_ != 0 and min_ != 0:
            maxVals.append(max_)
            minVals.append(min_)
            meanVals.append((max_-min_)/2)

avg = sum(meanVals)/len(meanVals)
print avg
# markerline, stemlines, baseline= plt.stem(meanVals)
plt.plot(meanVals)
# plt.setp(markerline, 'markerfacecolor', 'g')
# plt.setp(baseline, 'color', 'r', 'linewidth', 1)
plt.title('Max magnitude values')
plt.xlabel('time bins')
plt.ylabel('magnitude in freq bins')
plt.show()


# imw = []
# list_of_lists = [[50, 100, 150], [60, 120, 180], [30, 60, 90]]
# for list in list_of_lists:
#     n = max(list)
#     print (list)
#     list = [x/n for x in list]
#     imw.append(list)
#
# print (imw)

# thresh = 3
# a = np.array([2, 23, 15, 7, 9, 11, 17, 19, 5, 3])
# super_threshold_indices = a < thresh
# a[super_threshold_indices] = 0
#
# print a
