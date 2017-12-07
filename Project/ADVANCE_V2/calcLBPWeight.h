#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

struct FvPerformance {
	double best;
	vector<double> par;
	int misdetection;
	int falsealarm;
	int accuracy;
};

void getWeightedScore(vector<double> outMean, vector<double> outStd, vector<double> outLBP);