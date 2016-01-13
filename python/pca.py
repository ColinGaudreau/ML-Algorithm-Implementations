import numpy as np
from numpy.linalg import eig
from numpy.matlib import repmat

import pdb

class KPCA():
	'''
	Class for performing kernal PCA, used "http://pca.narod.ru/scholkopf_kernel.pdf" as reference (found it REALLY useful),
	I believe this is actually the original paper.
	'''
	def __init__(self, kernel='rbf', components=0, degree=2, gamma=1):
		self.kernel = kernel
		self.components = components
		self.degree = degree
		self.gamma = gamma
		self.X = None

		self._K_sum = None

		self._eigenvectors = None
		self._eigenvalues = None

		self._set_kernel()

	def fit(self, X):
		if self.components==0:
			self.components=X.shape[0]

		self.X = X
		N = X.shape[0]
		K = np.zeros((N,N))
		for row in range(N):
			for col in range(N):
				K[row,col] = self._kernel_func(X[row,:], X[col,:])

		self._K_sum = np.sum(K)
		K_c = K - repmat(np.reshape(np.sum(K, axis=1), (N,1)), 1, N)/N - repmat(np.sum(K, axis=0), N, 1)/N + self._K_sum/N**2

		self._eigenvalues, self._eigenvectors = eig(K_c)
		self._eigenvalues = np.real(self._eigenvalues)
		self._eigenvectors = np.real(self._eigenvectors)
		key = np.argsort(self._eigenvalues)
		key = key[::-1]
		self._eigenvalues = self._eigenvalues[key]
		self._eigenvectors = self._eigenvectors[:,key]
		self.X = self.X[key,:]


	def transform(self, X):
		if self._eigenvectors is None:
			raise Exception('Model has not been trained yet.')

		newX = np.zeros((X.shape[0], self.components))

		M = X.shape[0]; N = self.X.shape[0]
		K = np.zeros((M,N))
		for row in range(M):
			for col in range(N):
				K[row, col] = self._kernel_func(X[row,:], X[col,:])

		# pdb.set_trace()
		K_c = K - repmat(np.reshape(np.sum(K, axis=1), (M,1)), 1, N)/N - repmat(np.sum(K, axis=0), M, 1)/N + np.sum(K)/N**2

		for row in range(X.shape[0]):
			for col in range(self.components):
				newX[row,col] = np.dot(self._eigenvectors[:,col], K_c[row,:])

		return newX

	def _set_kernel(self):
		if self.kernel is 'rbf':
			self._kernel_func = self._rbf_kern
		elif self.kernel is 'linear':
			self._kernel_func = self._lin_kern
		elif self.kernel is 'poly':
			self._kernel_func = self._poly_kern
		else:
			raise Exception('Invalid kernel type.')

	def _rbf_kern(self, x,y):
		return np.exp(-self.gamma*np.dot(x-y,x-y))

	def _poly_kern(self, x,y):
		return np.power(np.dot(x,y), self.degree)

	def _lin_kern(self, x,y):
		return np.dot(x,y)
