/*
This code calculates the weights for different feature vectors (existing ones are mean/intensity, std, and LBP).
Input data are taken from FloodMonitor.cpp in .txt file format.
Frames when flood occurred and ended are manually labeled.
*/

#include <iostream>
#include <fstream>
#include <string>
#include "math.h"

#include <opencv2/opencv.hpp>
#include "calcLBPWeight.h"

using namespace std;
using namespace cv;

// Define parameters for Commonwealth Lane
int flood_frame_start = 800;
int flood_frame_end = 2360;
int num_of_images = 2359;
vector<int> labels(num_of_images,0);

string outMeanFile = "D:/PUB/Codes/floodEventsDetection/floodEventsDetection/2013-02-08 Commonwealth Lane - 01.20 PM - 02.30 PM OutMean.txt";
string outStdFile = "D:/PUB/Codes/floodEventsDetection/floodEventsDetection/2013-02-08 Commonwealth Lane - 01.20 PM - 02.30 PM OutStd.txt";
string outLBPFile = "D:/PUB/Codes/floodEventsDetection/floodEventsDetection/2013-02-08 Commonwealth Lane - 01.20 PM - 02.30 PM OutLBP.txt";

vector<double> getFileContent(string filename);

//int main ()
//{
//	// create labels/ground truth
//	for (int nn = 1; nn < num_of_images; nn++)
//	{
//		if ((nn < flood_frame_start) || (nn > flood_frame_end))
//		{
//			labels.at(nn) = 0;
//		}
//		else
//		{
//			labels.at(nn) = 1; // flood
//		}
//	}	
//	
//	// get data from files
//	vector<double> outMean = getFileContent(outMeanFile);
//	vector<double> outStd = getFileContent(outStdFile);
//	vector<double> outLBP = getFileContent(outLBPFile);
//
//	FvPerformance results;
//	getWeightedScore(outMean, outStd, outLBP);
//
//	return 0;
//}


double StrToDbl(string s) {
     double d;
     stringstream ss(s); //turn the string into a stream
     ss >> d; //convert
     return d;
}

vector<double> getFileContent(string filename)
{
	string line;
	vector<double> outValue;
	ifstream fileStream(filename.c_str());
	int idx = 0;

	if (fileStream.is_open())
	{
		while (getline(fileStream,line)) 
		{		
			outValue.push_back(StrToDbl(line));			
			// outValue.push_back(stod(line));	
			
		}
		fileStream.close();
	}	
	
	else cout << "Unable to open file"; 

	return outValue;
}

// create a function to get the best weights for each feature

void getWeightedScore(vector<double> outMean, vector<double> outStd, vector<double> outLBP)
{
	FvPerformance performance;
	performance.best = 0;
	performance.misdetection = 0;
	performance.falsealarm = 0;
	performance.accuracy = 0;
	performance.par.resize(4,0.0);
	double thre = 0.5; // threshold for flood detection
	double min = 0.1;
	double inc = 0.1;
	double max = 1.0;

	vector<double> DetScoreAll;	
	vector<int> Output;

	for (double w1 = min; w1 < max; w1 = w1+inc)
	{		
		for (double w2 = min; w2 < max; w2 = w2+inc)
		{
			if ((w1+w2) > 1.0)
				break;
			
			for (double w3 = min; w3 < max; w3 = w3+inc)
			{
				if ((w1+w2+w3) != 1.0)
				{
					break;
				}

				addWeighted(outMean,w1,outStd,w2,0,DetScoreAll,-1);
				addWeighted(DetScoreAll,1.0,outLBP,w3,0,DetScoreAll,-1);				

				Output.resize(num_of_images,0);

				// calculate accuracy using these weights

				for (int nn = 0; nn < num_of_images ; nn++)
				{
					if (DetScoreAll.at(nn) > thre)					
						Output.at(nn) = 1;					
					else
						Output.at(nn) = 0;
					
					if ((labels.at(nn) == 1) & (Output.at(nn) == 0))
						performance.misdetection++;
					else if ((labels.at(nn) == 0) & (Output.at(nn) == 1))
						performance.falsealarm++;
					else if (((labels.at(nn) == 1) & (Output.at(nn) == 1)) || (labels.at(nn) == 0) & (Output.at(nn) == 0))
						performance.accuracy ++;
				}

				if (performance.accuracy > performance.best)
				{
					performance.best = performance.accuracy;
					performance.par.at(0) = w1;
					performance.par.at(1) = w2;
					performance.par.at(2) = w3;
					performance.par.at(3) = thre;
				}
			}		
		}
	}

	addWeighted(outMean, performance.par.at(1), outStd, performance.par.at(2),0,DetScoreAll, -1);
	addWeighted(DetScoreAll, 1.0, outLBP, performance.par.at(3), 0, DetScoreAll, -1);
	
	Output.resize(num_of_images,0);	

	// re-initialize???
	performance.misdetection = 0;
	performance.falsealarm = 0;
	performance.accuracy = 0;

	for (int ii = 0; ii < num_of_images ; ii++)
	{
		if (DetScoreAll.at(ii) > performance.par.at(3))
			Output.at(ii) = 1;

		// calculate accuracy
		if ((labels.at(ii) == 1) & (Output.at(ii) == 0))
			performance.misdetection++;
		else if ((labels.at(ii) == 0) & (Output.at(ii) == 1))
			performance.falsealarm++;
		else if (((labels.at(ii) == 1) & (Output.at(ii) == 1)) || (labels.at(ii) == 0) & (Output.at(ii) == 0))
			performance.accuracy ++;
	}

	cout << "Misdetection = " << (double)performance.misdetection/num_of_images << endl;
	cout << "FalseAlarm = " << (double)performance.falsealarm/num_of_images << endl;
	cout << "Accuracy = " << (double)performance.accuracy/num_of_images << endl;
	cout << "Best Weight = [ " << performance.par.at(1) << " " << performance.par.at(2) << " " << performance.par.at(3) << " ]" << endl;
	cout << "Threshold used " << performance.par.at(3) << endl;

}
