#pragma once
#include <iostream>
#include <vector>
#include "math.h"

using namespace std;

class svm_test
{
public:
	svm_test(void);
	~svm_test(void);

	// Loads the linear SVM model from file
	bool	loadmodel(const char *cModel) ;
	double	operator()(const vector<double> &vdNewfeature) const ;
	float	operator()(const vector<float> &vdNewfeature) const ;

private:
	double*	linearwt_;
	double	linearbias_;
	long	kernel_type;
	long	poly_degree;
	double	rbf_gamma;
	double	coef_lin;
	double	coef_const;
	long	totwords;
	long	sv_num;
	double*	weight;
	float*	sup_vec;
};
