#include <iostream>
#include <list>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>
#include "gaussmm.h"

/*
	Basic demo which verifies that my gaussian mixture model
	fits to the data properly.
	 */

using namespace Eigen;
using namespace std;

void printvec(vector<VectorXd> &X) {
	for (size_t i = 0; i < X.size(); i++) {
		cout << X[i] << "\n--------------------\n";
	}
}

void printmat(vector<MatrixXd> &X) {
	for (size_t i = 0; i < X.size(); i++) {
		cout << X[i] << "\n--------------------\n";
	}
}

int main(){
	vector<VectorXd> X;

	default_random_engine gen(0);

	int N1 = 100;
	int N2 = 200;

	normal_distribution<double> d11(-10,0.08);
	normal_distribution<double> d12(-5,0.01);
	normal_distribution<double> d21(10,0.01);
	normal_distribution<double> d22(0,0.03);

	for (size_t i = 0; i < N1; i++) {
		VectorXd tmp(2);
		tmp << d11(gen), d12(gen);

		X.push_back(tmp);
	}
	for (size_t i = 0; i < N2; i++) {
		VectorXd tmp(2);
		tmp << d21(gen), d22(gen);

		X.push_back(tmp);
	}

	GaussMM *mm = new GaussMM(2);
	mm->fit(X);
	cout << "\n -------" << mm->priors <<"-----\n";
	printvec(mm->mus);
	printmat(mm->sigmas);
	delete mm;
}