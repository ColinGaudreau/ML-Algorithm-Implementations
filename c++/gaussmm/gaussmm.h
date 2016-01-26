#ifndef GAUSSMM_H
#define GAUSSMM_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include "gaussmm.h"

#define TOLERANCE 1e-15
#define MAX_ITERATIONS 100

using namespace Eigen;
using namespace std;

class GaussMM {

public:

	VectorXd priors;
	vector<VectorXd> mus;
	vector<MatrixXd> sigmas;
	vector<VectorXd> X;

	int mixtures;
	float tolerance;
	int maxIterations;
	bool didConverge;

	GaussMM(int mixtures, double tolerance = TOLERANCE, int maxIterations = MAX_ITERATIONS);
	void fit(vector<VectorXd> &X);

	static double normal(VectorXd &x, VectorXd &mu, MatrixXd &sigma);
	static double mixture(VectorXd &x, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas);

private:

	void initializeVariables();

	VectorXd findNewPriors(VectorXd &membershipExpectations);
	vector<VectorXd> findNewMeans(VectorXd &membershipExpectations);
	vector<MatrixXd> findNewCov(VectorXd &membershipExpectations);

	static double membershipPosterior(VectorXd &x, int mixture, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas);
	static VectorXd membershipExpectations(vector<VectorXd> &X, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas);
	static double logLikelihood(vector<VectorXd> &X, VectorXd &priors, vector<VectorXd> &mus, vector<MatrixXd> &sigmas);

};

#endif