#include "StdAfx.h"
#include "ActivityDetection.h"


ActivityDetection::ActivityDetection(void)
: m_bIsFrameRecorded( false )
{
	Inititalization();
}

ActivityDetection::~ActivityDetection(void)
{
	
}
void ActivityDetection::startActDetection(Mat &inputMat, Rect &selected_roi, string OutputPath, string OneFrameName)
{
	frameNum += 1;	
	outputpath = OutputPath;
	if(frameNum == 1)
	{
		// WD
		// --->>>
		//sm_Size.height = (int)(inputMat.rows/2);
		//sm_Size.width  = (int)(inputMat.cols/2);
		sm_Size.height = (int)(inputMat.rows);
		sm_Size.width  = (int)(inputMat.cols);
		// <<<---
		currMat = inputMat(selected_roi);		
	}
	else
	{			
		inputMat.copyTo(InputImg);
		currMat.copyTo(preMat);
		currMat = inputMat(selected_roi);		
		imshow("Video Data",inputMat);
		OptFlowLK(preMat,currMat,frameNum,OneFrameName);
		waitKey(1);
	}	
}
void ActivityDetection::InitThresMode(int nOpt)
{
	ThresMode = nOpt;	//1 - Adaptive, 2 - Fixed
}

void ActivityDetection::Inititalization()
{
	VelYBuff.clear();VelYBuff.begin();
	VelXBuff.clear();VelXBuff.begin();
	PeakMap.clear();PeakMap.begin();
	
	// tweak parameters
		WinSize				= 32;
		WinSize2			= 15;
		ObsWindow			= 300;	//Original: 300; SG:120
		ProbWindow			= 30;
		ThresBuf			= 1500;	//Original 60; 1500
		num_NNBours 		= 20;
		weight      		= 1;
		detectScoreFrame 	= 0;
		detectScoreWindow 	= 0;
		frameNum 			= 0;
	int numwin 				= 0;
	cv::Point  indPoint;
	vector<Point> tmpPoint;
	FeatureIndx.clear();FeatureIndx.begin();
	for(int r=0; r<DefaultHeight; r+=WinSize)
	{
		for(int c=0; c<DefaultWidth; c+=WinSize)
		{
			tmpPoint.clear();tmpPoint.begin();
			for(int rr= r; rr<(r+WinSize); rr++)
			{
				for(int cc=c; cc<(c+WinSize); cc++)
				{
					indPoint.x = rr; 
					indPoint.y = cc;
					tmpPoint.push_back(indPoint);
					
				}
			}
			FeatureIndx.push_back(tmpPoint);
			numwin++;			
		}
	}
	
	// 11 Dec 2012 - Ee Sin
	DetectionCount		 	= 0;
	DetectionFlag			= 0 ;
	PostDetFlag 			= 0;
	PostFramesWritten 		= 0;
	savePostFrameFlag 		= 0;
	PrevDetFrame 			= 0;
	PreFramesBuffer.clear(); PreFramesBuffer.begin();
	//FrameNames.clear(); FrameNames.begin();
	

	// 30 Sep 2013 - Ee Sin
	SmoothFrames			= 15;
	//ThresMode				= 1;		//1 - Adaptive, 2 - Fixed
	ThresScale				= 1.0;
	probSeize				= 0.85;
	TopK					= 40000;	// approximately half
	updatethres				= 600;		// Original 600;
	
	// 25 Oct 2013 - SG
	nThresMethod			= 1;		// 1-Median, else-Mean
	nAngleMethod			= 1;		// 1-Use Angle Different, else - Not using Angle Different
	nAngleDiffBuff			= 15;
	dThresAngle				= 10000000;
	dDetSumAngle			= 0;
	vmSumAngle.clear();
}
void ActivityDetection::OptFlowLK(Mat oldFrame,Mat newFrame, int index, string CurrFrameName)
{
	// determine the name of the image:
	// needs to be varied according to the image name:
	// here we assume year (4) _month (3) _day (3) _hour (3) _minutes (3) _seconds (3) _mseconds(3) .jpg (4) = 26 from end:
	string tmpnamestr (CurrFrameName.end()-26,CurrFrameName.end());// sets a global variable
	unsigned foundfile = CurrFrameName.find_last_of("/\\");
	imgnamestr = CurrFrameName.substr(foundfile+1);
	//imgnamestr = tmpnamestr;

	int buffsize;
	//Mat VelY,VelX;	//25 Oct 2013 SG
	Mat PreMat,CurrMat;
	Mat PreMatGray,CurrMatGray;
	Mat FlowField = Mat::zeros(oldFrame.rows,oldFrame.cols,CV_32FC2);
	vector<Mat> flowchannels(2);
	
	Size RImSize; RImSize.width = DefaultWidth;
	RImSize.height = DefaultHeight;
	
	cvtColor(oldFrame,PreMatGray,CV_RGB2GRAY);
	cvtColor(newFrame,CurrMatGray,CV_RGB2GRAY);
	resize(PreMatGray,PreMat,RImSize,0,0,INTER_NEAREST);
	resize(CurrMatGray,CurrMat,RImSize,0,0,INTER_NEAREST);
	normalize(PreMat,PreMat,0,255,NORM_MINMAX,-1);
	normalize(CurrMat,CurrMat,0,255,NORM_MINMAX,-1);
	// 11 Dec 2012
	//FrameNames.push_back(CurrFrameName);
	// PreFramesBuffer.push_back(CurrMat);
	FrameNames = CurrFrameName;
	if(PreFramesBuffer.size() < FrameBufferSize)
	{
		resize(InputImg,resizeImg,sm_Size,0,0,INTER_LINEAR);
		PreFramesBuffer.push_back(resizeImg);
	}
	else
	{
		PreFramesBuffer.erase(PreFramesBuffer.begin());
		resize(InputImg,resizeImg,sm_Size,0,0,INTER_LINEAR);
		PreFramesBuffer.push_back(resizeImg);
	}
	
		
	// cvCalcOpticalFlowLK(PreMatGray,CurrMatGray,winOPF,ipl_VelX,ipl_VelY);
	calcOpticalFlowFarneback(PreMat, CurrMat, FlowField, 0.5, 3, 8, 2, 3, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
	split(FlowField,flowchannels);
	//VelX = flowchannels[0];	//25 Oct 2013 SG
	//VelY = flowchannels[1];	//25 Oct 2013 SG

	if((int)VelYBuff.size()>=SmoothFrames)
	{
		VelYBuff.erase(VelYBuff.begin());
		VelXBuff.erase(VelXBuff.begin());
	}
	if(index >= StartFrame)
	{
		//VelYBuff.push_back(VelY);
		//VelXBuff.push_back(VelX);
		////Ee Sin 26 Nov
		//VelY.release();
		//VelX.release();
		
		//25 Oct 2013 - SG -->>
		VelYBuff.push_back(flowchannels[1]);
		VelXBuff.push_back(flowchannels[0]);
		//<<--
	}

	if(index >= (StartFrame + WinSize2))
	{
		OptFlowEnergy_Frame();
		int obs_buf_size = int(ObservationBuf.size());
		
		//25 Oct 2013 SG -->>
		if(nAngleMethod == 1)
		{	
			Mat mAngle	= Mat::zeros(Size(flowchannels[0].cols, flowchannels[0].rows),CV_32FC1);
			AngleFlow(flowchannels, mAngle);
			if((int)vmSumAngle.size() >= nAngleDiffBuff)
			{
				vmSumAngle.erase(vmSumAngle.begin());
				vmSumAngle.push_back(mAngle);
			}
			else
				vmSumAngle.push_back(mAngle);

			if(vmSumAngle.size()==nAngleDiffBuff)
				SumAngleFlow(vmSumAngle,dDetSumAngle);
		}
		//<<--
		
		if(int(ObservationBuf.size())>ObsWindow)
		{
			ObservationBuf.erase(ObservationBuf.begin());
			FeatureVect.erase(FeatureVect.begin());			
		}		
		if(int(ThresholdBuf.size())>ThresBuf)
		{
			ThresholdBuf.erase(ThresholdBuf.begin());
			ThresholdBufWin.erase(ThresholdBufWin.begin());
		}
		if(int(PeakMap.size())>ProbWindow)
		{
			PeakMap.erase(PeakMap.begin());
		}

		// 11 Dec 2012 - Ee Sin
		if(int(PreFramesBuffer.size())>FrameBufferSize)
		{
			PreFramesBuffer.erase(PreFramesBuffer.begin());
			FrameNames.erase(FrameNames.begin());
		}
	}
	if(index ==(StartFrame + WinSize2 + ThresBuf ))
	{
		FixedThreshold();	
	}
	if(index > (StartFrame + WinSize2 + (max( ObsWindow, ThresBuf)) ))
	{
		Detection();
	}
	
	if(index > (StartFrame + WinSize2 + (max( ObsWindow, ThresBuf)) +ProbWindow))
	{
		updatethres--;
		CheckCurrentFrame();
	}
	
	resizeImg.release();	// WD: must be put after CheckCurrentFrame cos it's still needed there
}

//25 Oct 2013 SG: Calculate Sum Angle Flow -->>
void ActivityDetection::SumAngleFlow(vector<Mat> &vmSumAngle, double& dSumAngle)
{
	Mat mSumAngle = Mat::zeros(vmSumAngle[0].rows,vmSumAngle[0].cols,CV_32FC1);
	//Sum the Angle into one Mat
	int nFrame = vmSumAngle.size();
	for(int i =0; i < nFrame; i++)
		mSumAngle+=vmSumAngle[i];

	dSumAngle = sum(mSumAngle).val[0];
}

//25 Oct 2013 SG: Calculate Angle Flow -->>
void ActivityDetection::AngleFlow(vector<Mat>& vmFlowChannel, Mat& mAngle)
{
	//Per Frame 
	Mat mTmpMag = Mat::zeros(Size(VelXBuff[0].cols, VelXBuff[0].rows),CV_32FC1);
	cartToPolar(vmFlowChannel[0], vmFlowChannel[1], mTmpMag, mAngle, TRUE);
	
	//SG: Angle Different, correct the opposite angles
	for (int j = 0; j < (int)mAngle.rows; j++)
	{
		float* fPtr_AngIm 	= mAngle.ptr<float>(j);
		float* fPtr_MagIm	= mTmpMag.ptr<float>(j);
		for (int i = 0; i < (int)mAngle.cols; i++)
		{
			float ftmp = floor(fPtr_MagIm[i]);
			if(ftmp <= 0.00001)
				fPtr_AngIm[i] = 0;
			else
			{
				if (fPtr_AngIm[i] >= 180)	
					fPtr_AngIm[i] = -(fPtr_AngIm[i] - 180);
			}
		}
	}
}
//<<---
void ActivityDetection::OptFlowEnergy_Frame()
{
	Mat mVx_sum 		= Mat::zeros(Size(VelXBuff[0].cols, VelXBuff[0].rows),CV_32FC1);
	Mat	mVy_sum 		= Mat::zeros(Size(VelXBuff[0].cols, VelXBuff[0].rows),CV_32FC1);
	Mat mEnergy_Frame	= Mat::zeros(Size(VelXBuff[0].cols, VelXBuff[0].rows),CV_32FC1);
	Mat mTmpAngle		= Mat::zeros(Size(VelXBuff[0].cols, VelXBuff[0].rows),CV_32FC1);
	
	for(int i=0; i < (int)VelXBuff.size(); i++)
	{
		mVx_sum = mVx_sum + VelXBuff.at(i);
		mVy_sum = mVy_sum + VelYBuff.at(i) ;
	}
	mVx_sum = mVx_sum/WinSize2;
	mVy_sum = mVy_sum/WinSize2;
	cartToPolar(mVx_sum, mVy_sum, mEnergy_Frame, mTmpAngle, TRUE);

	// Debug:
	if (prev_flow_mag.empty()){
		mEnergy_Frame.copyTo(prev_flow_mag);
		mEnergy_Frame.copyTo(curr_flow_mag);
	} else {
		Mat mConEnergy_Frame;
		double dMin=0.00, dMax = 0.00;
		minMaxLoc(mEnergy_Frame, &dMin, &dMax, 0);
		mEnergy_Frame.convertTo(mConEnergy_Frame,CV_8U,255.0/(dMax-dMin),0);
		imshow("mEnergy_Frame",mTmpAngle);
		imshow("mConEnergy_Frame",mConEnergy_Frame);
		waitKey(1);
		float fEnergy = sum(mConEnergy_Frame).val[0];  

		curr_flow_mag.copyTo(prev_flow_mag);
		mEnergy_Frame.copyTo(curr_flow_mag);
		
		
		ObservationBuf.push_back(fEnergy);
		ThresholdBuf.push_back(fEnergy);   
		OptFlowEnergy_Window(mConEnergy_Frame);
		mEnergy_Frame.release();
		 mConEnergy_Frame.release();
	   
	}

	//Mat mConEnergy_Frame;
	// double dMin=0.00, dMax = 0.00;
	// minMaxLoc(mEnergy_Frame, &dMin, &dMax, 0);
	// mEnergy_Frame.convertTo(mConEnergy_Frame,CV_8U,255.0/(dMax-dMin),0);
	// imshow("mEnergy_Frame",mEnergy_Frame);
	// imshow("mConEnergy_Frame",mConEnergy_Frame);
	// waitKey(1);
	// float fEnergy = sum(mConEnergy_Frame).val[0];  
	// vdObservationBuf.push_back(fEnergy);
	// vdThresholdBuf.push_back(fEnergy);   
	// OptFlowEnergy_Window(mConEnergy_Frame);
	// mEnergy_Frame.release();
	// mConEnergy_Frame.release();
		
}


void ActivityDetection::OptFlowEnergy_Window(Mat Energy_Frame)
{
	// Mat smallFrame;
	vector<double> EnergyVect;
	double sum_element;
	//Mat tmpEngFrame;
	Mat tmpEngFrame(Energy_Frame.rows,Energy_Frame.cols,Energy_Frame.type(),cv::Scalar::all(0));
	Energy_Frame.copyTo(tmpEngFrame);
	int num_window = 0;
	Mat smallFrame(WinSize,WinSize,tmpEngFrame.type(),cv::Scalar::all(0));

	for(int r=0; r<(tmpEngFrame.rows -WinSize+1); r+=WinSize)
	{
		for(int c=0; c<(tmpEngFrame.cols-WinSize+1); c+=WinSize)
		{	
			//Ee Sin 26 Nov
			smallFrame = Mat::zeros(WinSize,WinSize,tmpEngFrame.type());
			
			tmpEngFrame(Range(r,r+WinSize),Range(c,c+WinSize)).copyTo(smallFrame);
			sum_element = ((double)(sum(smallFrame).val[0]));
			sum_element = sum_element/((double)(WinSize*WinSize));
			EnergyVect.push_back(sum_element);
			num_window ++;			
		}
	}

	ThresholdBufWin.push_back(EnergyVect);
	FeatureVect.push_back(EnergyVect);
	//Ee Sin 26 Nov
	//EnergyVect.release();
	smallFrame.release();

	// 30 Sep 2013:
	tmpEngFrame.release();
	Energy_Frame.copyTo(tmpEngFrame);
	double base_thres = 2.0*std::sqrt(2.0);
	threshold(tmpEngFrame,tmpEngFrame,base_thres,1,THRESH_BINARY);
	// cout << "channels" << tmpEngFrame.channels()<<endl;
	
	//for(int r=0; r<(tmpEngFrame.rows -WinSize+1); r+=WinSize)
	//{
	//	for(int c=0; c<(tmpEngFrame.cols-WinSize+1); c+=WinSize)
	//	{	
	//		//Ee Sin 26 Nov
	//		smallFrame = Mat::zeros(WinSize,WinSize,tmpEngFrame.type());
	//		
	//		tmpEngFrame(Range(r,r+WinSize),Range(c,c+WinSize)).copyTo(smallFrame);
	//		sum_element = ((double)(sum(smallFrame).val[0]));
	//		sum_element = sum_element/((double)(WinSize*WinSize));
	//		EnergyVect.push_back(sum_element);
	//		num_window ++;			
	//	}
	//}
	
	
}

string ActivityDetection::convertInt(int number)
{
	std::stringstream ss;	//create a stringstream
	ss << number;			//add number to the stream
	return ss.str();		//return a string with the contents of the stream
}

void ActivityDetection::FixedThreshold()
{
	//From EeSin
	// additional thresholds:
	// low sensitivity - default mode:
	double min_ThresWin;
	double min_ThresFrame;
	if (ThresMode == 2) 
	{
		min_ThresWin    = 0.70; 	// 0.2; 0.35; 0.70
		min_ThresFrame  = 60000; 	// 15000; 25000; 60000
	} else 
	{
		min_ThresWin 	= 0;
		min_ThresFrame 	= 0;
	}
	
	//SG 25 Oct 2013 -->>
	int 	nSize 			= 0;
	double 	dMedianValue	= 0.00;
	Scalar 	sMean, sStdDev;
	
	//==== Frame ====//
	vector<double> vdSortThresholdBuff;
	meanStdDev((Mat)ThresholdBuf,sMean,sStdDev); 
	
	switch(nThresMethod)
	{
		case 1://Median
		cv::sort(ThresholdBuf,vdSortThresholdBuff,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		nSize = vdSortThresholdBuff.size();
		if (nSize%2==0)
			dMedianValue = 0.5*(vdSortThresholdBuff[nSize/2-1] + vdSortThresholdBuff[nSize/2]);
		else
			dMedianValue = vdSortThresholdBuff[int(floor(double(nSize)/2.0))];
		Threshold = dMedianValue + ThresScale*sStdDev.val[0];
		break;
		
		default://Mean
		Threshold = sMean.val[0] + ThresScale*sStdDev.val[0];
		break;
	}
	
	//==== Window ====//
	vector<double> vdSumThresBufWin((ThresholdBufWin[0]).size(),0);		//Sum
	vector<double> vdSumThresBuffWin2(num_NNBours,0); 					//Store top num_NNBours of vdSumThresBuffWin
	
	for (int i = 0; i < ThresholdBufWin.size(); i++)
		transform(vdSumThresBufWin.begin(),vdSumThresBufWin.end(),ThresholdBufWin[i].begin(),vdSumThresBufWin.begin(),std::plus<double>());
	transform(vdSumThresBufWin.begin(),vdSumThresBufWin.end(),vdSumThresBufWin.begin(),std::bind2nd(std::divides<double>(),ThresBuf));
	cv::sort(vdSumThresBufWin,vdSumThresBufWin,CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	std::copy(vdSumThresBufWin.begin(),vdSumThresBufWin.begin()+ num_NNBours-1,vdSumThresBuffWin2.begin());
	meanStdDev((Mat)vdSumThresBuffWin2,sMean,sStdDev); 

	switch(nThresMethod)
	{
		case 1://Median
		nSize = vdSumThresBuffWin2.size();
		if (nSize%2==0)
			dMedianValue = 0.5*(vdSumThresBuffWin2[nSize/2-1] + vdSumThresBuffWin2[nSize/2]);
		else
			dMedianValue = vdSumThresBuffWin2[int(floor(double(nSize)/2.0))];
		ThresholdWin = dMedianValue + ThresScale*sStdDev.val[0];
		break;
		
		default://Mean
		ThresholdWin = sMean.val[0] + ThresScale*sStdDev.val[0];
		break;
	}
	//<<--
	
	//SG:17Oct2013 
	//Change the saving directory to the output folder instead of the solution folder 
	FILE *tmpFile; 
	//tmpFile = fopen("threshold.txt","a+");
	tmpstring = outputpath + "threshold.txt";
	tmpFile = fopen(tmpstring.c_str(),"a+");
	
	fprintf(tmpFile,"Threshold = %.05f, ThresholdWin = %.05f\n",Threshold,ThresholdWin);	
	fclose(tmpFile);

	// Optional: To ensure that the thresholds are above a minimum level.
	ThresholdWin 	= max(ThresholdWin,min_ThresWin);
	Threshold 		= max(Threshold,min_ThresFrame);
}

//Original Codes From EeSin-->>
/*void ActivityDetection::FixedThreshold()
{
	double diff,sum_var;
	double meanValue,stdvariation;
	// Ee Sin 26 Nov
	vector<int> Index;
	vector<double> TempVect;
	double tempV;

	// additional thresholds:
	// low sensitivity - default mode:
	double min_ThresWin;
	double min_ThresFrame;
	if (ThresMode == 2) {
		min_ThresWin    = 0.35; // 0.2
		min_ThresFrame  = 25000; // 15000
	} else {
		min_ThresWin = 0;
		min_ThresFrame = 0;
	}
	sum_var = 0;
	sum_var = cv::sum(ThresholdBuf).val[0];
	meanValue = sum_var/(double)(ThresholdBuf.size());
	sum_var = 0;
    for(int i=0; i<ThresholdBuf.size(); i++)
	{
		sum_var = sum_var + pow((ThresholdBuf.at(i)- meanValue),2);
		
	}
	stdvariation = (double)sqrt((double)(sum_var/(double)(ThresholdBuf.size())));
	Threshold = meanValue + ThresScale*stdvariation;	

	vector<double> TempBuff (80,0); 

	// Ee Sin - do for threshold of windows:
    for(int i=0; i<ThresholdBuf.size(); i++)
	{
		TempVect = ThresholdBufWin.at(i);
		for(int ii=0; ii<TempVect.size(); ii++)
		{
			tempV = TempBuff.at(ii) + TempVect.at(ii)/ThresBuf;
			TempBuff.at(ii) = tempV;
		}	
	}

	//cv::sort(TempBuff,TempBuff,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
	cv::sort(TempBuff,TempBuff,CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	sum_var     = 0;
	stdvariation = 0; 
	vector<double> TempBuff2 (num_NNBours,0); 
	
	std::copy(TempBuff.begin(),TempBuff.begin()+ num_NNBours-1,TempBuff2.begin());
	meanValue = mean(TempBuff2).val[0];

	for(int i=0; i<TempBuff2.size();i++)
	{
		sum_var = sum_var + pow((TempBuff2.at(i) - meanValue),2);
	}
	stdvariation = sqrt(sum_var/TempBuff2.size());
	ThresholdWin = meanValue + ThresScale*stdvariation;
	

	//SG:17Oct2013 
	//Change the saving directory to the output folder instead of the solution folder 
	FILE *tmpFile; 
	//tmpFile = fopen("threshold.txt","a+");
	tmpstring = outputpath + "threshold.txt";
	tmpFile = fopen(tmpstring.c_str(),"a+");
	
	fprintf(tmpFile,"Threshold = %.05f, ThresholdWin = %.05f\n ",Threshold,ThresholdWin);	
	fclose(tmpFile);
	
	// Optional: To ensure that the thresholds are above a minimum level.
	ThresholdWin = max(ThresholdWin,min_ThresWin);
	Threshold = max(Threshold,min_ThresFrame);
}
*/
void ActivityDetection::Detection()
{
	Mat IndexBuff,tmp_Buff,tmp_Buff_win,IndexBuff_win;
	Mat meanFeatVect,ThresholdMat,ThresholdMatWin;
	vector<vector<double>> tmp_FeatureVect;
	vector<double> tmp_ObservationBuff;
	
	
	int length_index;
	double prob;
	int rows = (int)ObservationBuf.size();
	
	tmp_Buff = Mat(rows,1,CV_32F);
	IndexBuff = Mat(rows,1,CV_8UC1);	
	tmp_Buff = Mat(ObservationBuf);
	
	ThresholdMat = Mat::ones(tmp_Buff.rows,tmp_Buff.cols,CV_64F)*Threshold;
	// compare the energy values of a input frame with threshold matrix
	cv::compare(tmp_Buff,ThresholdMat,IndexBuff,CMP_GT);
	
	length_index = countNonZero(IndexBuff);
	prob = ((double)length_index)/((double)ObsWindow);
	// ProbBuf.push_back(prob);
	confidFrame.push_back(prob);
    // Store the maximum of tmp_Buff:
	double tmp_score = 0.0;
	for (int ii = 0;ii<tmp_Buff.rows;ii++){
		tmp_score = tmp_score + tmp_Buff.at<double>(ii,0)/rows;
	}
	tmpstring = outputpath + "raw_frame_score.txt";
	FILE *tmpFile_det2;
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%.05f\n",tmp_score);
	fclose(tmpFile_det2);   

	prob = 0;
	length_index = 0;
	
	IndexBuff.setTo(0);
	ThresholdMat.release();
	int nframe = FeatureVect.size();
	tmp_Buff_win = Mat(tmp_Buff.size(),tmp_Buff.type());
	tmp_Buff_win.setTo(0);
	// Ee Sin 26 Nov
	ThresholdMatWin = Mat::ones(tmp_Buff.rows,tmp_Buff.cols,CV_64F) *ThresholdWin;
	
	tmp_FeatureVect = FeatureVect;	
	
	double meanvalue;	
	double initvalue = 0;
	tmp_score = 0.0;
	for(int ii=0; ii<nframe; ii++)
	{
		meanvalue = double(mean(FeatureVect.at(ii)).val[0]);
		tmp_Buff_win.at<double>(ii,0) = meanvalue;
		tmp_score = tmp_score + meanvalue/nframe;

	}
    		
	tmpstring = outputpath + "raw_win_score.txt";	
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%.05f\n",tmp_score);
	fclose(tmpFile_det2);   


	cv::compare(tmp_Buff_win,ThresholdMatWin,IndexBuff,CMP_GT);
	ThresholdMatWin.release();
	
	length_index = countNonZero(IndexBuff);	
	prob = ((double)length_index)/((double)ObsWindow);
	confidWindow.push_back(prob);

	
	int num_window;
	
	num_window = (int)FeatureVect.at(0).size();
	meanFeatVect = Mat::zeros(num_window,1,CV_64F);
	
	// Ee Sin 26 Nov
	vector<double> tmp_vectorBuff(num_window,0);
	
	tmp_Buff.release();
	tmp_Buff_win.release();
	tmp_Buff = Mat(num_window,1,CV_32F);
	for(int i=0; i<(int)(FeatureVect.size());i++)
	{
		tmp_vectorBuff = FeatureVect.at(i);
		tmp_Buff = Mat(tmp_vectorBuff);		
		meanFeatVect  = meanFeatVect + tmp_Buff;
	}


	double scale = (double)FeatureVect.size();
	cv::divide(scale,meanFeatVect,meanFeatVect,-1);	


	Mat sortIndex ;
	int index;

	cv::sortIdx(meanFeatVect,sortIndex,CV_SORT_EVERY_COLUMN+ CV_SORT_DESCENDING);
    index = (int) sortIndex.at<float>(0,0);

	cv::Point stpoint,endpoint;	 
	Mat matWindow = Mat::ones(WinSize,WinSize,CV_8UC1)*255;
	Mat currentMap = Mat::zeros(DefaultHeight,DefaultWidth,CV_8UC1);

	for(int i=0; i<num_NNBours; i++)
	{
		index =  sortIndex.at<int>(i,0);
		stpoint = ((FeatureIndx.at(index)).front());
		endpoint = (FeatureIndx.at(index)).back();
		matWindow.copyTo(currentMap(Range(stpoint.x,endpoint.x+1),Range(stpoint.y,endpoint.y+1)));	
		
	}

	PeakMap.push_back(currentMap);	
	
	if(confidFrame.size() > ProbWindow)
	{
		confidFrame.erase(confidFrame.begin());
		confidWindow.erase(confidWindow.begin());
	}
	
	// Ee Sin 26 Nov
	currentMap.release();
	tmp_Buff.release();
	tmp_Buff_win.release();
	matWindow.release();
	
}
void ActivityDetection::CheckCurrentFrame()
{
	double meanConfidFrame,meanConfidWindow;
	double detScoreOut;
	double intScore;
	Mat IntMap,SumMap;
    
	detScoreOut = 0;
	detectScoreFrame = 0;
	detectScoreWindow = 0;
	meanConfidFrame = mean(confidFrame).val[0];
	meanConfidWindow = mean(confidWindow).val[0];

	IntMap = Mat::zeros(PeakMap.at(0).rows,PeakMap.at(0).cols,PeakMap.at(0).type());
	SumMap = Mat::zeros(PeakMap.at(0).rows,PeakMap.at(0).cols,PeakMap.at(0).type());
	cv::bitwise_and(PeakMap.at(0),PeakMap.at(1),IntMap);
	cv::bitwise_or(PeakMap.at(0),PeakMap.at(1),SumMap);
	
	for(int kk=2; kk<ProbWindow; kk++)
	{
		cv::bitwise_and(PeakMap.at(kk),IntMap,IntMap);
		cv::bitwise_or(PeakMap.at(kk),SumMap,SumMap);
	}
	intScore = (double)((sum(IntMap).val[0])/(sum(SumMap).val[0]));
	detectScoreFrame	= (weight * meanConfidFrame) + ((1 - weight)*intScore) ;
	detectScoreWindow	= (weight * meanConfidWindow) + ((1 - weight)*intScore);
	IntMap.release();
	SumMap.release();

	string ImgStr;
	int	 PreBufferSize = 0;
	detScoreOut = detectScoreFrame * 0.1  + detectScoreWindow * 0.9;
	
	//25 Oct 2013 SG: Check Angle Different --> 
	if((nAngleMethod == 1) && (abs(dDetSumAngle) > dThresAngle) && (detScoreOut >= probSeize))
		detScoreOut = probSeize;
	//<<--

	tmpstring = outputpath + "detection_score.txt";
	FILE *tmpFile_det2;
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%.05f\n",detScoreOut);
	fclose(tmpFile_det2);
    savePostFrameFlag = 0;
	// isolate the individual components:
	tmpstring = outputpath + "global_score.txt";
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%.05f\n",meanConfidFrame);
	fclose(tmpFile_det2);
	tmpstring = outputpath + "local_score.txt";
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%.05f\n",meanConfidWindow);
	fclose(tmpFile_det2);
	tmpstring = outputpath + "int_score.txt";
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%.05f\n",intScore);
	fclose(tmpFile_det2);
	tmpstring = outputpath + "SumAngleFlow.txt";
	tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
	fprintf(tmpFile_det2,"%6.3f\n",dDetSumAngle);
	fclose(tmpFile_det2);

	if (detScoreOut >= probSeize)
	{			

		tmpstring = outputpath + "Debug_detectedscores.txt";
		tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
		fprintf(tmpFile_det2,".05f\n",detScoreOut);
		fclose(tmpFile_det2);

		// Check if it is the start of a new detection:
		if((PostDetFlag ==0)&& (DetectionFlag == 0))
		{
			DetectionCount++;
			PostDetFlag 		= 1;
			DetectionFlag 		= 1;
			savePostFrameFlag 	= 1;
			saveframeNum 		= 1;
			PostFramesWritten 	= 0;
			string detstring 	= convertInt(DetectionCount);
			tmpstring = outputpath + "Detection_" + detstring + "/";	
			mkdir(tmpstring.c_str());
			// LPCWSTR dirstring;
			// dirstring = (LPCWSTR) tmpstring.c_str();
			// CreateDirectory(dirstring,NULL);
			PreBufferSize = (int)PreFramesBuffer.size();
			ImgStr = tmpstring + imgnamestr;
			imwrite(ImgStr,resizeImg);
			resizeImg.release();
			PrevDetFrame = frameNum;
		}
		else
		{
			if((saveframeNum)< DetInterval )
			{
				//save the image 
				
				PostFramesWritten ++;
				saveframeNum ++;
				PreBufferSize = (int)PreFramesBuffer.size();
				string detstring = convertInt(DetectionCount);
				tmpstring = outputpath + "Detection_" + detstring + "/";
			    ImgStr = tmpstring + imgnamestr;
				imwrite(ImgStr,resizeImg);
			}
			else 
			{
				PrevDetFrame = 0;
				savePostFrameFlag = 1;
				saveframeNum = 1;
				DetectionCount++;
				//create the new folder and save the first image.
				PostFramesWritten = 0;
				string detstring = convertInt(DetectionCount);
				tmpstring = outputpath + "Detection_" + detstring + "/";
				if (!exists(tmpstring)){
					mkdir(tmpstring.c_str());
				}
				ImgStr = tmpstring + imgnamestr;
				PreBufferSize = (int)PreFramesBuffer.size();

				imwrite(ImgStr,resizeImg);
				
			}
		}
		m_bIsFrameRecorded = true;
	}
	else
	{
		DetectionFlag = 0;  //DetectionFlag == 0;
		if((PostDetFlag > 0)&&(PostFramesWritten < DetInterval))
		{
			// save the images continuous 
			PostFramesWritten ++;
			saveframeNum ++;
			PreBufferSize = (int)PreFramesBuffer.size();
			string detstring = convertInt(DetectionCount);
			tmpstring = outputpath + "Detection_" + detstring + "/";
			ImgStr = tmpstring + imgnamestr;
			imwrite(ImgStr,resizeImg);

			m_bIsFrameRecorded = true;

		}
		else
		{
			PostDetFlag = 0;
			PostFramesWritten = 0;
			m_bIsFrameRecorded = false;
			// 30 Sep 2013
			// Perform adaptive updates of threshold:
			if (updatethres <= 0) {
				FixedThreshold();
				updatethres = 1500;
			}

		}		
	}
}

void ActivityDetection:: GetPreFrameBuffer( vector<Mat> &savePreFrameBuffer)
{
  
	savePreFrameBuffer.clear();
	savePreFrameBuffer.begin();
	savePreFrameBuffer = PreFramesBuffer ;	
}
void ActivityDetection::GetPreFrameString(vector<string> &savePreFrameString)
{
	savePreFrameString.clear();
	savePreFrameString.begin();
	savePreFrameString = PreFramesString;
}


bool ActivityDetection::IsFrameRecorded( void )
{
	return m_bIsFrameRecorded;
}
