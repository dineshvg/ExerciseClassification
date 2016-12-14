import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
print X
y = np.array([1, 1, 2, 2])
# class_1 = np.array([[-1, -1], [-2, -1]])
# class_2 = np.array([[1, 1], [2, 1]])
from matplotlib import pyplot as plt
# plt.plot(class_1,'ro')
# plt.plot(class_2,'g^')
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print(clf.predict([[-0.8, 1]]))

what_class = np.array([[-0.8, 1]])
plt.scatter(what_class[:, 0], what_class[:, 1])
print what_class
# plt.plot(what_class,'yo')
plt.show()