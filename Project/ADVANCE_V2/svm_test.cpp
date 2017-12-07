#include "StdAfx.h"
#include "svm_test.h"


svm_test::svm_test(void)
{
	linearwt_	= NULL;
	linearbias_ = 0;
	kernel_type = 0;
	poly_degree = 0;
	rbf_gamma	= 0;
	coef_lin	= 0;
	coef_const	= 0;
	totwords	= 0;
	sv_num		= 0;
	weight		= NULL;
	sup_vec		= NULL;
}

svm_test::~svm_test(void)
{
	if (linearwt_)
		delete[] linearwt_;
	if (weight)
		delete[] weight;
	if (sup_vec)
		delete[] sup_vec;
}

bool svm_test::loadmodel(const char *cModel)
{
	/*
	Remarks  	: initialisation for SVM_classify
	Paramters	:
	char *cModel	= directory of the model file
	*/

	FILE *modelfl;
	if ((modelfl = fopen(cModel, "rb")) == NULL)
		return false;											//SG added (05Aug2013)
	//	cerr<<"\n Unable to open the model file \n\n"<<endl;	//SG commented (05Aug2013)


	/* Load SVM parameters for classification from model file */
	fread(&kernel_type, sizeof(long),1,modelfl);
	fread(&poly_degree, sizeof(long),1,modelfl);
	fread(&rbf_gamma, sizeof(double),1,modelfl);
	fread(&coef_lin, sizeof(double),1,modelfl);
	fread(&coef_const, sizeof(double),1,modelfl);
	fread(&totwords, sizeof(long),1,modelfl);
	fread(&sv_num, sizeof(long),1,modelfl);
	fread(&linearbias_, sizeof(double),1,modelfl);

	if(kernel_type == 0) { /* linear kernel */
		/* save linear wts also */
		linearwt_ = new double[totwords+1];
		fread(linearwt_, sizeof(double),totwords+1,modelfl);
	} 
	else 
	{
		weight = new double[sv_num-1];
		sup_vec = new float[(sv_num-1)*totwords];
		for (int i=0; i<sv_num-1; i++)
		{
			fread(&(weight[i]), sizeof(double),1,modelfl);
			for(int j=0; j<totwords; j++)
			{
				fread(&(sup_vec[i*totwords+j]), sizeof(double),1,modelfl);
			}
		}
	}
	fclose(modelfl);
	return true;
}

double svm_test::operator()(const vector<double> &desc) const 
{
	double sum, sum1, sum3, power;
	switch(kernel_type)
	{
	case 0: /*linear*/
		sum = 0;
		for (int i= totwords; i--; ) 
			sum += linearwt_[i]*desc[i]; 
		return sum - linearbias_;
		break;
	case 1: /*polynomial*/
		sum = 0;
		for (int j=sv_num-1; j--; ){
			sum1 = 0;
			for (int i=totwords; i--; )
				sum1 += sup_vec[j*totwords+i]*desc[i];
			power = coef_lin*sum1+coef_const;
			for (int k=poly_degree-1; k--; )
				power = power*(coef_lin*sum1+coef_const);
			sum += weight[j]*power;
		}
		return sum - linearbias_;
		break;
	case 2: /*radial basis function*/
		sum = 0;
		sum3 = 0;
		for (int j=sv_num-1; j--; ){
			sum1 = 0; 
			for (int i=totwords; i--; ){
				sum3 = desc[i] - sup_vec[j*totwords+i];
				sum1 += sum3*sum3;
			}
			sum += weight[j]*exp(-rbf_gamma*sum1);
		}
		return sum - linearbias_;
		break;
	case 3: /*sigmoid neural net*/
		sum = 0;
		for (int j=sv_num-1; j--; ){
			sum1 = 0;
			for (int i=totwords; i--; ){
				sum1 += sup_vec[j*totwords+i]*desc[i];
			}
			sum += weight[j]*tanh(coef_lin*sum1+coef_const);
		}
		return sum - linearbias_;
		break;
	default: /*custom-kernel & unknown option*/
		cerr<<"Unknown kernel function\n"<<endl;
		exit(1);
		break;
	}
}

float svm_test::operator()(const vector<float> &desc) const 
{
	double sum, sum1, sum3, power;
	switch(kernel_type)
	{
	case 0: /*linear*/
		sum = 0;
		for (int i= totwords; i--; ) 
			sum += linearwt_[i]*desc[i]; 
		return sum - linearbias_;
		break;
	case 1: /*polynomial*/
		sum = 0;
		for (int j=sv_num-1; j--; ){
			sum1 = 0;
			for (int i=totwords; i--; )
				sum1 += sup_vec[j*totwords+i]*desc[i];
			power = coef_lin*sum1+coef_const;
			for (int k=poly_degree-1; k--; )
				power = power*(coef_lin*sum1+coef_const);
			sum += weight[j]*power;
		}
		return sum - linearbias_;
		break;
	case 2: /*radial basis function*/
		sum = 0;
		sum3 = 0;
		for (int j=sv_num-1; j--; ){
			sum1 = 0; 
			for (int i=totwords; i--; ){
				sum3 = desc[i] - sup_vec[j*totwords+i];
				sum1 += sum3*sum3;
			}
			sum += weight[j]*exp(-rbf_gamma*sum1);
		}
		return sum - linearbias_;
		break;
	case 3: /*sigmoid neural net*/
		sum = 0;
		for (int j=sv_num-1; j--; ){
			sum1 = 0;
			for (int i=totwords; i--; ){
				sum1 += sup_vec[j*totwords+i]*desc[i];
			}
			sum += weight[j]*tanh(coef_lin*sum1+coef_const);
		}
		return sum - linearbias_;
		break;
	default: /*custom-kernel & unknown option*/
		cerr<<"Unknown kernel function\n"<<endl;
		exit(1);
		break;
	}
}