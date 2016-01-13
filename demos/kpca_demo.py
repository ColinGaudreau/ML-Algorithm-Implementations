import sys
sys.path.insert(0, r'../python')

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from pca import KPCA

import numpy as np

import pdb

svc = SVC(kernel='linear')
data = load_digits()
X = np.reshape(data.images, (data.images.shape[0],-1))
y = data.target

N = 200

X_train = X[:N,:]
y_train = y[:N]
X_test = X[N:,:]
y_test = y[N:]

# Basic linear svc on raw data set
svc.fit(X_train, y_train)

result = svc.score(X_test, y_test)

print 'Result with no PCA: ' + str(result)

# Regular linear PCA (using my KCPA with linear kernel)
kpca = KPCA(kernel='linear', components=54)
kpca.fit(X_train)
Xlin_train = kpca.transform(X_train)
Xlin_test = kpca.transform(X_test)

svc.fit(Xlin_train, y_train)

result = svc.score(Xlin_test, y_test)

print 'Result with regular PCA: ' + str(result)

# Kernel PCA (poly deg 3)
kpca = KPCA(kernel='poly', degree=3)
kpca.fit(X_train)
Xpoly_train = kpca.transform(X_train)
Xpoly_test = kpca.transform(X_test)

svc.fit(Xpoly_train, y_train)

result = svc.score(Xpoly_test, y_test)

print 'Result with kernel (poly, deg 3) PCA: ' + str(result)

# Kernel PCA (rbf gamma=10)
gamma = 8.5e-4
kpca = KPCA(kernel='rbf', gamma=gamma)
kpca.fit(X_train)
Xrbf_train = kpca.transform(X_train)
Xrbf_test = kpca.transform(X_test)

svc.fit(Xrbf_train, y_train)

result = svc.score(Xrbf_test, y_test)

print 'Result with kernel (rbf, gamma {}) PCA: {}'.format(gamma, result)