#include "gaussmm.h"

GaussMM::GaussMM(int mixtures, double tolerance, int maxIterations){
	this->mixtures = mixtures;
	this->tolerance = tolerance;
	this->maxIterations = maxIterations;
	this->didConverge = false;
}

void GaussMM::fit(vector<VectorXd> &X) {
	this->X = X;

	this->initializeVariables();

	VectorXd newPriors;
	vector<VectorXd> newMus;
	vector<MatrixXd> newSigmas;
	VectorXd membershipExpectations;

	for (size_t iteration = 0; iteration < this->maxIterations && !this->didConverge; iteration++) {
		membershipExpectations = GaussMM::membershipExpectations(this->X, this->priors, this->mus, this->sigmas);
		newPriors = this->findNewPriors(membershipExpectations);
		newMus = this->findNewMeans(membershipExpectations);
		newSigmas = this->findNewCov(membershipExpectations);

		if (abs(GaussMM::logLikelihood(this->X, this->priors, this->mus, this->sigmas) - GaussMM::logLikelihood(this->X, newPriors, newMus, newSigmas)) < this->tolerance) {
			this->didConverge = true;
		}

		this->priors = newPriors;
		this->mus = newMus;
		this->sigmas = newSigmas;
	}

	return;
}

void GaussMM::initializeVariables() {
	int d = this->X[0].rows();
	VectorXd priors(this->mixtures);
	vector<VectorXd> mus;
	vector<MatrixXd> sigmas;

	priors -= priors;
	priors = priors.array() + 1.0/this->mixtures;

	srand((unsigned int) time(0));

	for (size_t i = 0; i < this->mixtures; i++) {
		VectorXd mu = VectorXd::Random(d);
		MatrixXd sigma(d,d);
		sigma.setIdentity();
		mus.push_back(mu);
		sigmas.push_back(sigma);
	}

	this->priors = priors;
	this->mus = mus;
	this->sigmas = sigmas;

	return;
}

VectorXd GaussMM::findNewPriors(VectorXd &membershipExpectations) {
	return membershipExpectations/this->X.size();
}

vector<VectorXd> GaussMM::findNewMeans(VectorXd &membershipExpectations) {
	int N = this->X.size();
	vector<VectorXd> mus;
	VectorXd result;

	for (size_t i = 0; i < this->mus.size(); i++) {
		VectorXd result(this->mus[i].rows());
		result -= result; // zero vector

		for (size_t j = 0; j < N; j++) {
			result += GaussMM::membershipPosterior(this->X[j], i, this->priors, this->mus, this->sigmas) * this->X[j];
		}
		result /= membershipExpectations[i];
		mus.push_back(result);
	}

	return mus;
}

vector<MatrixXd> GaussMM::findNewCov(VectorXd &membershipExpectations) {
	int N = this->X.size();
	int d = this->X[0].rows();
	vector<MatrixXd> sigmas;
	VectorXd result;

	for (size_t i = 0; i < this->sigmas.size(); i++) {
		MatrixXd result(d,d);
		result -= result; // zero matrix

		for (size_t j = 0; j < N; j++) {
			result += GaussMM::membershipPosterior(this->X[j], i, this->priors, this->mus, this->sigmas) * (this->X[j] - this->mus[i])*(this->X[j] - this->mus[i]).transpose();
		}
		result /= membershipExpectations[i];
		sigmas.push_back(result);
	}

	return sigmas;
}

double GaussMM::normal(VectorXd &x, VectorXd &mu, MatrixXd &sigma) {
	return exp((-(x  - mu).transpose() * sigma.inverse() * (x - mu)/2.0)(0));
}

double GaussMM::mixture(VectorXd &x, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas) {
	double result = 0;
	for (size_t i = 0; i < priors.rows(); i++) {
		result += priors[i] * GaussMM::normal(x, mus[i], sigmas[i]);
	}
	return result;
}

double GaussMM::membershipPosterior(VectorXd &x, int mixture, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas) {
	return priors[mixture]*GaussMM::normal(x, mus[mixture], sigmas[mixture]) / GaussMM::mixture(x, priors, mus, sigmas);
}

VectorXd GaussMM::membershipExpectations(vector<VectorXd> &X, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas) {
	int N = X.size();
	VectorXd expectations(priors.rows());

	for (size_t i = 0; i < priors.rows(); i++) {
		expectations[i] = 0;
		for (size_t j = 0; j < N; j++) {
			expectations[i] += GaussMM::membershipPosterior(X[j], i, priors, mus, sigmas);
		}
	}

	return expectations;
}

double GaussMM::logLikelihood(vector<VectorXd> &X, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas) {
	int N = X.size();
	double logLikelihood = 0;

	for (size_t i = 0; i < N; i++) {
		logLikelihood += log(GaussMM::mixture(X[i], priors, mus, sigmas));
	}

	return logLikelihood;
}

