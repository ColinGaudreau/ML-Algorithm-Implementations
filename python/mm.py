import numpy as np
from numpy.matlib import repmat, concatenate
from numpy.linalg import inv
from numpy.random import randn

np.random.seed(0)

from pdb import set_trace

class GaussMM():
	'''
	Class for fitting Gaussian mixture model to data

	components: number of mixtures
	ivalues: initial values (will read more on this topic)
	'''
	def __init__(self, components, ivalues=None, tolerance=1e-6, max_iterations = 1e2):
		self.components = components
		self.ivalues = ivalues
		self.tolerance = tolerance
		self.max_iterations = max_iterations

	def fit(self, X):
		self.X = X
		if self.ivalues is None:
			self._set_ivalues()

		converged = False
		iteration = 0
		d = self.X.shape[1]
		while not converged and iteration < self.max_iterations:
			priors = concatenate([self._prior(component) for component in range(self.components)], axis=1)
			set_trace()
			new_priors = np.asarray([np.sum(priors[:,component])/self.X.shape[0] for component in range(self.components)])
			new_mus = np.asarray([np.sum(priors[:,[component]] * self.X, axis=0)/np.sum(priors[:,component]) for component in range(self.components)])
			new_sigmas = concatenate([sum(np.reshape((np.dot((self.X[[n],:] - self.mus[[component],:]).transpose(), (self.X[[n],:] - self.mus[[component],:]))*priors[n,component]),(d,d,1)) for n in range(X.shape[0])) for component in range(self.components)],axis=2)

			# converged = np.abs(self._log_likelihood(self.X, new_mus, new_sigmas, new_priors) - self._log_likelihood(self.X, self.mus, self.sigmas, self.priors)) < self.tolerance

			self.priors = new_priors
			self.mus = new_mus
			self.sigmas = new_sigmas

			iteration+=1

	def gauss_mixture(self):
		return sum(self.priors[i]*self._gauss(self.X, self.mus[i,:], self.sigmas[:,:,i]) for i in range(self.components))

	def _gauss(self, X, mu, sigma):
		mu = repmat(np.reshape(mu,(1,mu.size)),X.shape[0],1)
		Xmu_dif = X - mu
		sigma_inv = inv(sigma)

		return np.exp(-np.asarray([np.dot(Xmu_dif[i,:],np.dot(sigma_inv, Xmu_dif.transpose()[:,[i]])) for i in range(X.shape[0])])/2)

	def _log_likelihood(self, X, mus, sigmas, priors):
		return np.sum(np.log(sum(priors[component]*self._gauss(X, mus[component,:], sigmas[:,:,component]) for component in range(self.components))))

	def _prior(self, mixture):
		return self.priors[mixture]*self._gauss(self.X, self.mus[mixture,:], self.sigmas[:,:,mixture])/self.gauss_mixture()

	def _set_ivalues(self):
		d = self.X.shape[1]
		self.priors = np.ones(self.components)/self.components
		self.mus = randn(self.components, d)
		self.sigmas = np.tile(np.reshape(np.eye(d),(d,d,1)), [1,1,self.components])
