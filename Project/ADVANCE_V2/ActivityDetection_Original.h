#include <boost/filesystem.hpp>
#include <stdlib.h>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm> 
#include <functional> 
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <direct.h>
#include <windows.h>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

#pragma once
#define			StartFrame       5
#define			DefaultWidth	 320
#define			DefaultHeight	 256
#define			FrameBufferSize  300  // number of frames 
#define			DetInterval		 600  // number of frames between valid detections
class ActivityDetection
{
public:
	ActivityDetection(void);
	virtual ~ActivityDetection(void);

public:
	void ActivityDetection::OptFlowLK(Mat oldFrame,Mat newFrame, int index, string CurrFrameName);
	void ActivityDetection::startActDetection(Mat &inputMat, Rect &selected_roi, string OutputPath, string OneFrameName);
	void GetPreFrameBuffer(vector<Mat> &savePreFrameBuffer);
	void GetPreFrameString(vector<string> &savePreFrameString);
	bool IsFrameRecorded( void );
	
	//30 Oct 2013 SG
	void	InitThresMode(int nOpt);
	
private:
	void OptFlowEnergy_Frame();
	void OptFlowEnergy_Window(Mat Energy_Frame);
	void FixedThreshold();
	void Detection();
	void CheckCurrentFrame();
	void Inititalization();
	string convertInt(int number);	

	//25 Oct 2013 SG
	void	SumAngleFlow(vector<Mat> &vmSumAngle, double& dSumAngle);
	void	AngleFlow(vector<Mat>& vmFlowChannel, Mat& mAngle);
	
private:
	vector<double>  ObservationBuf,ProbBuf;		           
	vector<double>  ThresholdBuf;
	vector<double> EnergyVect;
	vector<double> confidFrame,confidWindow;
    vector<vector<double>>	FeatureVect,ThresholdBufWin;	
	vector<Mat>     VelYBuff,VelXBuff;
	vector<vector<Point>> FeatureIndx;
	vector<Mat> PeakMap;
	Mat currMat,preMat;
	int WinSize,WinSize2,ThresBuf;
	int ObsWindow,ProbWindow;
	int num_NNBours;
	double Threshold,ThresholdWin;
	double weight;
	
	char framePath[250];

	// 11 Dec 2012 - Ee Sin
	int DetectionCount;	
	int PostFramesWritten;
	int PrevDetFrame;
	Mat InputImg;
	Mat resizeImg;
	string FrameNames;

	bool m_bIsFrameRecorded;
	
	//25 Oct 2013 - SG
	int	nAngleDiffBuff;
	vector<Mat> vmSumAngle;

public:

	double detectScoreFrame,detectScoreWindow;
	int frameNum ;
	int saveframeNum;
	Rect rectROI;
	
	// by TMH at 17 Jan 2013
	vector<Mat>		PreFramesBuffer;
	vector<Mat>     PostFrameBuffer;
	vector<string>  PreFramesString;
	Size sm_Size; 
	

	int DetectionFlag;
	int PostDetFlag;
	int savePostFrameFlag;
	char PostStr[100];
	string outputpath;
	string tmpstring;
	string imgnamestr; // the name of the current image

	// 30 Sep 2013 - ES
	int SmoothFrames;
	int ThresMode;
	// ThresMode = 1 - Adaptive
	// ThresMode = 2 - Fixed
	double ThresScale;
	double probSeize;
	int TopK; 			// approximately half
	int updatethres; 	// the number of frames to pass before updating thres

	//25 Oct 2013 - SG
	int		nThresMethod;	// 1 - Median, else - Mean
	int 	nAngleMethod;	// 1 - Use Angle Different, else - not using Angle Different
	double 	dThresAngle;	//
	double	dDetSumAngle;
	
	Mat prev_flow_mag;
	Mat curr_flow_mag;

};
