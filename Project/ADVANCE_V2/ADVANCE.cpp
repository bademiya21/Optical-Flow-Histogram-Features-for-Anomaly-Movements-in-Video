//#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
//#include <direct.h> // to create new folder
#include <iostream>
#include <fstream>
#include <string>
//#include <vector>
//#include "MouseSelect.h";
//#include "genLBP.h" // ********* used in LBP **********
#include "ActivityDetection.h"

using namespace std;
//using namespace boost::filesystem;
using namespace cv;
/*//Original Codes:
// -------------------------- Global variables -----------------------------------
Rect Sel_rect;
int initflag = 1;
int SmoothWin = 5; // number of frames to smooth the optical flow field
int DefaultHeight = 256;
int DefaultWidth = 320;
int WinSize = 64;
int NumBinsLBP = 0;
int opmode; // mode of operation
Mat PrevFrame, CurrFrame;
vector<Mat> Vx_Buff;
vector<Mat> Vy_Buff;
vector<Point> WinPoint;
vector<Mat> AllTrainFv,PosTrainFv,NegTrainFv,TestFv;
vector<Mat> SmoothFlowField_x,SmoothFlowField_y;
Mat TrainFv;
string root_dir;
Mat CodeBook;
vector<vector< CvBoost*>> DataModels;
vector<int> WinSizes;
int ProbFrames = 30;
int ObservationFrames;
vector<vector<double>> ProbScores;
string OutputPath,DataPath;
string sImgFilename;
// --------------------------- Functions -----------------------------------------

string convertInt(int number){
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

void NormaliseImg(Mat& Img, Mat& ImgOut){
	// Note: Images must be greyscale:	
	Mat Img32F;
	Img.convertTo(Img32F,CV_32F);
	normalize(Img32F,ImgOut,0.0,1.0,NORM_MINMAX);
}

void InitSubWindows2(int winsize,int overlapwin, int input_h, int input_w){
	// 26 Sep 2013
	// with overlap:
	cv::Point  indPoint;
	cv::Point  indPoint2;
	vector<Point> tmpPoint;
	int numwin = 0;	
	if (overlapwin == 1){		
		for(int r=0; r<input_h; r+=winsize/2){
			for(int c=0; c<input_w; c+=winsize/2){	
				if (r+winsize < input_h && c+winsize < input_w){
					indPoint2.x = c;
					indPoint2.y = r;
					WinPoint.push_back(indPoint2);			
					WinPoint.push_back(Point(c,r));	
					numwin++;			
				}
			}
		}
	} else {
		for(int r=0; r<input_h; r+=winsize){
			for(int c=0; c<input_w; c+=winsize){	
				indPoint2.x = c;
				indPoint2.y = r;
				WinPoint.push_back(indPoint2);			
				numwin++;			
			}
		}

	}
}

void ExtractWinFV_Params(Mat &img, int numbins, int min_r, int max_r, Mat &outfeatures){
	// basic function to get features:
	int winsize_x = DefaultWidth; //WinSize
	int winsize_y = DefaultHeight;
	for (int ii=0;ii<WinPoint.size();ii++){
		Rect winRoi = Rect(WinPoint.at(ii).x,WinPoint.at(ii).y,winsize_x,winsize_y);
		Mat roiImg = img(winRoi);
		Mat OneHist;
		vector<Mat> ImgData;
		ImgData.push_back(roiImg);
		int channels[] = { 0 };
		float range[] = { min_r,max_r };
		const float *ranges[] = { range };
		int hsize[] = { numbins };
		calcHist(&ImgData[0],1,channels,Mat(),OneHist,1,hsize,ranges,true,false);
//		double minVal; double maxVal; Point minLoc; Point maxLoc;
		//minMaxLoc(OneHist,&minVal,&maxVal,&minLoc,&maxLoc,Mat());
		//cout << "minval " << minVal << "    max val "<< maxVal << endl;
		// normalize(OneHist,OneHist,1,0,NORM_L2);		
		normalize(OneHist,OneHist,1,0,NORM_L2);		
		// normalize(OneHist,OneHist,1,0,NORM_L1);	
//		minMaxLoc(OneHist,&minVal,&maxVal,&minLoc,&maxLoc,Mat());
//		cout << "minval " << minVal << "    max val "<< maxVal << endl;
		Mat TempHist;
		TempHist = OneHist.t();
		outfeatures.push_back(TempHist);		
		ImgData.clear();
	}
}


void ExtractLBPFeatures(Mat &Img, Mat &FVect, int numbins){

	int winsize_x = DefaultWidth; //WinSize
	int winsize_y = DefaultHeight;

	uchar*** pppArray = NULL;
	int tlength = 1;
	int i, j, k;

	pppArray = new uchar**[tlength];
	if(!pppArray)
	{
		printf("Allocating memory failed.\n");
	}
	int width = WinSize;
	int height = WinSize;

	for(i=0; i<tlength; i++)
	{		
		pppArray[i] = new uchar*[height];
		if(!pppArray[i])
		{
			printf("Allocating memory failed.\n");			
		}		
			
		for(j=0; j<height; j++)
		{
			pppArray[i][j] = new uchar[width];
			if(!pppArray[i][j])
			{
				printf("Allocating memory failed.\n");				
			}
			
			for(k=0; k<width; k++)
			{
				//write data into the 3d array
				pppArray[i][j][k] = 0;
			}
		}		
	}

	// LBP setting
	LBP lbp;	
	lbp.width = width; lbp.height = height; lbp.tlength = 1;        	
	lbp.R.xR = 2; lbp.R.yR = 2;	//range
	lbp.SN.xy = 8; //number of points
	lbp.uni = 1; //uniform lbp
	lbp.interp = 1; //interpolation for points
	lbp.norm = 1; //normalization of histogram
	lbp.riu2 = 1; // riu2 lbp

	//Extract LBP for each sub-window
	for (int ii=0;ii<WinPoint.size();ii++)
	{
		Rect winRoi = Rect(WinPoint.at(ii).x,WinPoint.at(ii).y,winsize_x,winsize_y);
		Mat roiImg = Img(winRoi);
		
		// copy Img into pppArray data;
		for(i=0; i<tlength; i++)
		{					
			for(j=0; j<height; j++)
			{
				for(k=0; k<width; k++)
				{
					//write data into the 3d array
					pppArray[i][j][k] = roiImg.at<float>(j,k);
					// pppArray[i][j][k] = roiImg.at<char>(j,k);
				}
			}		
		}

		// create LBP
		lbp.CreateHistogram(pppArray, 0);	// calculate LBP histogram
		
		// print LBP histogram for debug
		//if ((lbp.riu2==1) || (lbp.uni==1))
		//{
		//	printf("The histogram is:\n");
		//	for(i=0; i<lbp.uni_bin.xy+1; i++)
		//	{
		//		printf("%.4f ", lbp.uni_hist.pHist_xy[i]);
		//	}
		//	printf("\n");
		//	printf("The bin number is: %d\n", lbp.uni_bin.xy+1);	
		//	printf("\n\n");	
		//}

		NumBinsLBP = lbp.uni_bin.xy+1;

		Mat TempHist = Mat::zeros(1,NumBinsLBP,CV_32FC1);;

		for(i=0; i<lbp.uni_bin.xy+1; i++)
		{
			TempHist.at<float>(i)= lbp.uni_hist.pHist_xy[i];
		}
		normalize(TempHist,TempHist,1,0,NORM_L2);
		FVect.push_back(TempHist);
		TempHist.release();
	}

	lbp.CreateHistogram(pppArray, -1);  // release LBP histogram

	// Release pppArray data
	if(pppArray)
	{
		for(i=0; i<tlength; i++)
		{
			if(pppArray[i])
			{
				for(j=0; j<height; j++)
				{
					if(pppArray[i][j])
					{
						delete []pppArray[i][j];
					}
				}
				
				delete []pppArray[i];
			}
		}
		
		delete []pppArray;					
	}
}

void SmoothField(vector<Mat>& vx_buff, vector<Mat>& vy_buff, Mat& sm_field){
	// function to obtain smoothed optical flow fields
	Mat vx_temp 		= Mat::zeros(Size(vx_buff[0].cols, vx_buff[0].rows),CV_32FC1);
	Mat	vy_temp 		= Mat::zeros(Size(vy_buff[0].cols, vy_buff[0].rows),CV_32FC1);
	sm_field		    = Mat::zeros(Size(vx_buff[0].cols, vx_buff[0].rows),CV_32FC1);
	Mat mTmpAngle		= Mat::zeros(Size(vx_buff[0].cols, vx_buff[0].rows),CV_32FC1);

	for (int ii = 0; ii<vx_buff.size();ii++){
		vx_temp = vx_temp + vx_buff[ii];
		vy_temp = vy_temp + vy_buff[ii];
	}
	vx_temp = vx_temp/SmoothWin;
	vy_temp = vy_temp/SmoothWin;
	cartToPolar(vx_temp, vy_temp, sm_field, mTmpAngle);
}

void GetStdDevImg(Mat& Img, Mat& StdImg){
	// Get standard deviation image using blur filters
	Mat mu;
	Mat Img32F;
	Img.convertTo(Img32F,CV_32F);
	blur(Img32F,mu,Size(3,3));
	Mat mu2;
	blur(Img32F.mul(Img32F),mu2,Size(3,3));
	Mat sigma;
	cv::sqrt(mu2-mu.mul(mu),sigma);
	normalize(sigma,StdImg,0.0,1.0,NORM_MINMAX);
}

void ExtractFeatures(Mat& sm_field, Mat& flow_fv){
	// Function to extract flow features:
	// define two types of flow features -
	// 1. intensity distribution
	// 2. texture distribution
	Mat tmp_flow_fv;
	flow_fv.release();
	int f_bins = 64;
	int f_min = 0;
	int f_max = 1;

	// cout << sm_field.channels() << endl;

	// Intensity - normalised to 0 - 1
	Mat norm_field;
	Mat f_features;
	NormaliseImg(sm_field, norm_field);
	ExtractWinFV_Params(norm_field,f_bins,f_min,f_max,f_features);
	f_features.copyTo(tmp_flow_fv);
	f_features.release();

	// Texture - LBP
	ExtractLBPFeatures(sm_field, f_features, 0);	
	hconcat(tmp_flow_fv,f_features,tmp_flow_fv);	
	f_features.release();

	// Texture - STD:
	Mat StdImg;
	GetStdDevImg(sm_field, StdImg);
	ExtractWinFV_Params(StdImg,f_bins,f_min,f_max,f_features);	
	hconcat(tmp_flow_fv,f_features,tmp_flow_fv);
	// FrameFv.push_back(FVect);
	f_features.release();

	// Pull and straighten the flow_fv:
	flow_fv = tmp_flow_fv.reshape(0,1);
	tmp_flow_fv.release();
}


void GetTestCodebookIdx(vector<Mat>& PosTrainFv,Mat& codebook,Mat& poscodewords){
	// function to get codeword idx:
	vector<vector<DMatch>> knnmatches;
	BFMatcher matcher(cv::NORM_L2);
	for (int ii=0;ii<PosTrainFv.size();ii++){
		Mat one_feature_seq = PosTrainFv.at(ii);
		Mat one_tmp_fv = Mat::zeros(one_feature_seq.rows,1,CV_32FC1);
		for (int rr=0;rr<one_feature_seq.rows;rr++){	
			matcher.knnMatch(one_feature_seq.row(rr),codebook,knnmatches,1);			
			int tempidx = knnmatches.at(0).at(0).trainIdx;
			one_tmp_fv.at<float>(rr,0) = tempidx;
			// one_tmp_fv.at<float>(0,tempidx) = one_tmp_fv.at<float>(0,tempidx) + 1;								
		}
		if (poscodewords.empty()){
			one_tmp_fv.copyTo(poscodewords);
		} else {
			hconcat(poscodewords,one_tmp_fv,poscodewords);
		}
		// poscodewords.push_back(one_tmp_fv);
	}
}

void TestFlowFeatures(vector<Mat> &testfv,string outputpath,vector<double>& WinScores){
	// function to test flow features:
	// Convert testfv into a Mat for sampling:	
	int cbk_size = CodeBook.rows;
	Mat testcodewords;
	GetTestCodebookIdx(testfv,CodeBook,testcodewords);
	vector<CvBoost*> ClassifierList(WinSizes.size());
	ClassifierList = DataModels.at(0);
	
	// sampling begins:
	for (int ww=0;ww<WinSizes.size();ww++){
		int tmp_winsize = WinSizes.at(ww);
		for (int ii=testcodewords.cols-tmp_winsize;ii<testcodewords.cols;ii+=tmp_winsize){
			// one test-seq for each window size:
			Mat one_seq = testcodewords.colRange(ii,ii+tmp_winsize);
			// Mat one_seq = tmp_one_seq.t();

			Mat tmp_hist;					
			vector<Mat> tmpdata;
			tmpdata.push_back(one_seq);
			int channels[] = { 0 };
			float range[] = { 0 , tmp_winsize };
			const float *ranges[] = { range };
			int hsize[] = { cbk_size };
			calcHist(&tmpdata[0],1,channels,Mat(),tmp_hist,1,hsize,ranges,true,false);
			// normalise histogram to l2:
			normalize(tmp_hist,tmp_hist,1,0,NORM_L2);
			Mat testfv  = tmp_hist.t();

			double classout = ClassifierList.at(ww)->predict(testfv,Mat(),Range::all(),false,true);			
			WinScores.push_back(classout);
		}				
	}
}

void GetFlowFeatures(Mat& prevframe, Mat& currframe, Rect sel_roi, int nOpt){
	// function to calculate dense optical flow, and to get flow features

	
	Mat sm_field;	
	Mat flow_features;
	vector<Mat> flowchannels(2);

	Size fixedSize(DefaultWidth,DefaultHeight);
	Mat FlowField = Mat::zeros(DefaultHeight,DefaultWidth,CV_32FC2);

	Mat prevframe_gray,currframe_gray;
	cvtColor(prevframe,prevframe_gray,CV_RGB2GRAY);
	cvtColor(currframe,currframe_gray,CV_RGB2GRAY);

	Mat prev_mat = prevframe_gray(sel_roi);
	Mat curr_mat = currframe_gray(sel_roi);
	resize(prev_mat,prev_mat,fixedSize);
	resize(curr_mat,curr_mat,fixedSize);


	//SG 10 Feb 2014-->>
	normalize(prev_mat,prev_mat,0,255,NORM_MINMAX,-1);
	normalize(curr_mat,curr_mat,0,255,NORM_MINMAX,-1);

	//double dMin1 = 0.00, dMin2 = 0.00, dMax1 = 0.00, dMax2 = 0.00;
	//minMaxLoc(prev_mat,&dMin1,&dMax1,0);
	//minMaxLoc(curr_mat,&dMin2,&dMax2,0);
	//normalize(prev_mat,prev_mat,dMin1,dMax1,NORM_MINMAX,-1);
	//normalize(curr_mat,curr_mat,dMin2,dMax2,NORM_MINMAX,-1);
	//<<--


	calcOpticalFlowFarneback(prev_mat, curr_mat, FlowField, 0.5, 3, 8, 2, 3, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
	split(FlowField,flowchannels);
	
	Vx_Buff.push_back(flowchannels[0]);
	Vy_Buff.push_back(flowchannels[1]);

	if (Vx_Buff.size() > SmoothWin){
		// to maintain flow fields at SmoothWin
		Vy_Buff.erase(Vy_Buff.begin());
		Vx_Buff.erase(Vx_Buff.begin());
	}

	if (Vx_Buff.size() == SmoothWin){
		// calculate smooth optical flow:
		SmoothField(Vx_Buff,Vy_Buff,sm_field);
		ExtractFeatures(sm_field,flow_features);
		if (nOpt == 1){
			// stack the flow_features:
			if (TrainFv.empty()){
				flow_features.copyTo(TrainFv);
			} else {
				vconcat(TrainFv,flow_features,TrainFv);
			}
			
		} else if (nOpt == 2)
		{
			// Get codeword and test:
			TestFv.push_back(flow_features);
			vector<double> WinScores;
			ObservationFrames = WinSizes.at(2); // Longest window:
			if (TestFv.size() > ObservationFrames)
			{
				WinScores.clear();
				// enough frames to test:
				TestFv.erase(TestFv.begin());
				TestFlowFeatures(TestFv,OutputPath,WinScores);
				
				for (int ww=0;ww<WinSizes.size();ww++){
					// Write data to files:
					int tmp_winsize = WinSizes.at(ww);
					string win_str = convertInt(tmp_winsize);
					string tmpstring;

					tmpstring = OutputPath + "raw_det_score_"+win_str+".txt";
					FILE *tmpFile_det2;
					tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
					fprintf(tmpFile_det2,"%.05f\n",WinScores.at(ww));
					fclose(tmpFile_det2);

					// Smoothing of scores: 
					// store in ProbScores:
					vector<double> tmp_score;
					if (ProbScores.size()<WinSizes.size()){
						if (WinScores.at(ww)>1){
							tmp_score.push_back(1.0);
						} else {
							tmp_score.push_back(0.0);
						}
						ProbScores.push_back(tmp_score);
					} else {
						tmp_score = ProbScores.at(ww);
						if (WinScores.at(ww)>1){
							tmp_score.push_back(1.0);
						} else {
							tmp_score.push_back(0.0);
						}					

						if (tmp_score.size()>ProbFrames){
							tmp_score.erase(tmp_score.begin());
							// Calculate probability and store:
							double probscore = 0.0;
							for (int kk=0;kk<tmp_score.size();kk++){
								probscore = probscore + tmp_score.at(kk)/ProbFrames;
							}
							tmpstring = OutputPath + "prob_score_"+win_str+".txt";
							FILE *tmpFile_det3;
							tmpFile_det3 = fopen(tmpstring.c_str(),"a+");
							fprintf(tmpFile_det3,"%s %.05f\n",sImgFilename.c_str(), probscore);
							fclose(tmpFile_det3);

						}
						ProbScores.at(ww) = tmp_score;					
					}
					tmp_score.clear();
				}						
			}
		}
	}
}

void GetTestData(const path& basepath,int offline_mode){
	// TEST
	// Function to buffer and get test data from video
	
	if (offline_mode == 1 )
	{
		// gets data 
		for (directory_iterator iter = directory_iterator(basepath); iter != directory_iterator(); iter++)
		{
			directory_entry entry = *iter;
			if (is_directory(entry.path())) 
			{
				cout << "Processing directory " << entry.path().string() << endl;
				root_dir = entry.path().string();
				GetTestData(entry.path(), offline_mode);
			}
			else
			{
				path entryPath = entry.path();
				if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg")
				{
					cout << " Processing image: " << entryPath.string() << endl;
					CurrFrame = imread(entryPath.string());				
					if (initflag == 1)
					{
						string roifile = root_dir+"/ROI.xml";
						if (!exists(roifile)){
							// Read first image and then select ROI					
							Sel_rect = SelectRoi(CurrFrame,0);
							initflag = 0;
							PrevFrame.release(); // initialise
						} else {
							// Read from .xml:						
							FileStorage fs_w(roifile,FileStorage::READ);	
							fs_w["ROIx"] >> Sel_rect.x;
							fs_w["ROIy"] >> Sel_rect.y;
							fs_w["ROIh"] >> Sel_rect.height;
							fs_w["ROIw"] >> Sel_rect.width;
							fs_w.release();
							initflag = 0;
						}
					}
					// Begin processing the images using optical flow:
					if (PrevFrame.empty())
					{
						CurrFrame.copyTo(PrevFrame);
					} else 
					{
						//Get the image filename
						string		sImgPath	 = entryPath.string();
						unsigned	uTmp 	  	 = sImgPath.find_last_of("/\\");
									sImgFilename = sImgPath.substr(uTmp+1);

						// sufficient number of frames to start:
						GetFlowFeatures(PrevFrame,CurrFrame,Sel_rect,opmode);
						CurrFrame.copyTo(PrevFrame);
					}		
					
				}
			}
		}

	} 
	else if (offline_mode == 0)
	{
		// To-do: online mode
	}
}

void GetRawTrainData(const path& basepath) 
{
	// Recursive function that process files in sub directories and runs activity detection on it
	for (directory_iterator iter = directory_iterator(basepath); iter
			!= directory_iterator(); iter++)
	{
		directory_entry entry = *iter;
		if (is_directory(entry.path())) 
		{
			cout << "Processing directory " << entry.path().string() << endl;
			root_dir = entry.path().string();
			GetRawTrainData(entry.path());
			// Note: This only works if training data are separated into sub-folders, one-level down from input-folder
			if (!TrainFv.empty()){
				// finished gathering data for one sub-folder:
				AllTrainFv.push_back(TrainFv);
				TrainFv.release();
			}
			if (opmode == 1){
				initflag = 1;
			}
		} else 
		{
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg") 
			{
				// process the images:
				cout << " Processing image: " << entryPath.string() << endl;
				CurrFrame = imread(entryPath.string());				
				if (initflag == 1)
				{
					string roifile = root_dir+"/ROI.xml";
					if (!exists(roifile)){
						// Read first image and then select ROI					
						Sel_rect = SelectRoi(CurrFrame,0);
						initflag = 0;
						PrevFrame.release(); // initialise
					} else {
						// Read from .xml:						
						FileStorage fs_w(roifile,FileStorage::READ);	
						fs_w["ROIx"] >> Sel_rect.x;
						fs_w["ROIy"] >> Sel_rect.y;
						fs_w["ROIh"] >> Sel_rect.height;
						fs_w["ROIw"] >> Sel_rect.width;
						fs_w.release();
						initflag = 0;
					}
				}

				// Begin processing the images using optical flow:
				if (PrevFrame.empty()){
					CurrFrame.copyTo(PrevFrame);
				} else {
					// sufficient number of frames to start:
					GetFlowFeatures(PrevFrame,CurrFrame,Sel_rect,opmode);
					CurrFrame.copyTo(PrevFrame);
				}									
			}
		}
	}
}

void PrepareFiles(vector<Mat>& alltrainfv,Mat& trainfv){
	// function to convert vector<Mat> to vconcat Mat
	trainfv.release();
	for (int ii = 0;ii<alltrainfv.size();ii++){
		Mat tmpfv = alltrainfv.at(ii);
		if(trainfv.empty()){
			tmpfv.copyTo(trainfv);
		} else {
			vconcat(trainfv,tmpfv,trainfv);
		}
	}
}

void GetSeqCodebookIdx(vector<Mat>& PosTrainFv,Mat& codebook,vector<Mat>& poscodewords){
	// function to get codeword idx:
	vector<vector<DMatch>> knnmatches;
	BFMatcher matcher(cv::NORM_L2);
	for (int ii=0;ii<PosTrainFv.size();ii++){
		Mat one_feature_seq = PosTrainFv.at(ii);
		Mat one_tmp_fv = Mat::zeros(one_feature_seq.rows,1,CV_32FC1);
		for (int rr=0;rr<one_feature_seq.rows;rr++){	
			matcher.knnMatch(one_feature_seq.row(rr),codebook,knnmatches,1);			
			int tempidx = knnmatches.at(0).at(0).trainIdx;
			one_tmp_fv.at<float>(rr,0) = tempidx;
			// one_tmp_fv.at<float>(0,tempidx) = one_tmp_fv.at<float>(0,tempidx) + 1;								
		}
		poscodewords.push_back(one_tmp_fv);
	}
}

void GetCodeBooks(const path& negpath, const path& pospath,string outputpath){
	// function to collect all raw data and train codebooks:
	Mat PosFv,NegFv;
	int codebook_size = 100;
	GetRawTrainData(negpath);	
	NegTrainFv = AllTrainFv;
	AllTrainFv.clear();
	PrepareFiles(NegTrainFv,NegFv);
	Mat negcodebook;
	Mat labels;
	kmeans(NegFv, codebook_size, labels,
		   TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
		   1, KMEANS_PP_CENTERS, negcodebook);	

	labels.release();

	GetRawTrainData(pospath);
	PosTrainFv = AllTrainFv;
	AllTrainFv.clear();
	PrepareFiles(PosTrainFv,PosFv);
	Mat poscodebook;
	kmeans(PosFv, codebook_size, labels,
		   TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
		   1, KMEANS_PP_CENTERS, poscodebook);

	PosFv.release();
	NegFv.release();
	
	Mat codebook;
	poscodebook.copyTo(codebook);
	vconcat(codebook,negcodebook,codebook);

	vector<Mat> poscodewords;
	GetSeqCodebookIdx(PosTrainFv,codebook,poscodewords);
	vector<Mat> negcodewords;
	GetSeqCodebookIdx(NegTrainFv,codebook,negcodewords);

	int cbk_size = codebook.rows;
	string tmp_path = outputpath + "seize_codebook.xml";
	FileStorage fs_w(tmp_path,FileStorage::WRITE);	
	fs_w << "POSCBK" << poscodebook;
	fs_w << "NEGCBK" << negcodebook;
	fs_w.release();

	tmp_path = outputpath + "pos_sequence.xml";
	FileStorage fs_w1(tmp_path,FileStorage::WRITE);	
	fs_w1 << "Data" << PosTrainFv;
	fs_w1 << "Codewords" << poscodewords;
	fs_w1 << "CBKSize" << cbk_size;
	fs_w1.release();

	tmp_path = outputpath + "neg_sequence.xml";
	FileStorage fs_w2(tmp_path,FileStorage::WRITE);	
	fs_w2 << "Data" << NegTrainFv;
	fs_w2 << "Codewords" << negcodewords;
	fs_w2 << "CBKSize" << cbk_size;
	fs_w2.release();
	
	NegTrainFv.clear();
	PosTrainFv.clear();
	poscodewords.clear();
	negcodewords.clear();
	poscodebook.release();
	negcodebook.release();
}

void SampleSeq(vector<int>& winsize,vector<Mat>& codewords,int cbk_size,vector<Mat>& train_data,vector<Mat>& train_labels,int labelmode){
	// Samples sequence according to winsize:
	for (int ww=0;ww<winsize.size();ww++){
		int tmp_win = winsize.at(ww);	
		Mat one_win_train_data;
		for (int ii=0;ii<codewords.size();ii++){
			Mat one_seq = codewords.at(ii);
			int seq_len = one_seq.rows;
			if (seq_len > tmp_win){
				for (int rr=0;rr<seq_len-tmp_win;rr+=tmp_win/2){
					Mat tmp_sample = one_seq.rowRange(rr,rr+tmp_win);
					// convert into histogram:
					Mat tmp_hist;					
					vector<Mat> tmpdata;
					tmpdata.push_back(tmp_sample);
					int channels[] = { 0 };
					float range[] = { 0 , tmp_win };
					const float *ranges[] = { range };
					int hsize[] = { cbk_size };
					calcHist(&tmpdata[0],1,channels,Mat(),tmp_hist,1,hsize,ranges,true,false);
					// normalise histogram to l2:
					normalize(tmp_hist,tmp_hist,1,0,NORM_L2);
					Mat one_hist = tmp_hist.t();
					if (one_win_train_data.empty()){
						one_hist.copyTo(one_win_train_data);						
					} else {
						vconcat(one_win_train_data,one_hist,one_win_train_data);
					}
				}
			}
		}
		Mat one_win_train_labels = labelmode*Mat::ones(one_win_train_data.rows,1,CV_32FC1);
		train_data.push_back(one_win_train_data);
		train_labels.push_back(one_win_train_labels);
		one_win_train_data.release();
		one_win_train_labels.release();
	}
}

void TrainClassifiers(string outputpath){
	// function to train classifiers:
	vector<int> winsize;
	winsize.push_back(60); winsize.push_back(90); winsize.push_back(120);
	vector<Mat> neg_train_data,pos_train_data;
	vector<Mat> neg_train_labels,pos_train_labels;
	int cbk_size;
	// load neg-data:
	vector<Mat> negcodewords;
	string tmp_path = outputpath + "neg_sequence.xml";
	FileStorage fs_w2(tmp_path,FileStorage::READ);	
	fs_w2["Codewords"] >> negcodewords;
	fs_w2["CBKSize"] >> cbk_size;
	fs_w2.release();

	// start sampling data:
	cout<< "Sampling Negative Training Data" <<endl;
	SampleSeq(winsize,negcodewords,cbk_size,neg_train_data,neg_train_labels,-1);

	// load pos-data:
	vector<Mat> poscodewords;
	tmp_path = outputpath + "pos_sequence.xml";
	FileStorage fs_w1(tmp_path,FileStorage::READ);	
	fs_w1["Codewords"] >> poscodewords;
	fs_w1["CBKSize"] >> cbk_size;
	fs_w1.release();

	cout<< "Sampling Positive Training Data" <<endl;
	SampleSeq(winsize,poscodewords,cbk_size,pos_train_data,pos_train_labels,1);

	// Train Classifiers:
	for (int ww=0;ww<winsize.size();ww++){
		Mat train_data = pos_train_data.at(ww);
		Mat train_labels = pos_train_labels.at(ww);
		vconcat(train_data,neg_train_data.at(ww),train_data);
		vconcat(train_labels,neg_train_labels.at(ww),train_labels);
		string winstr_size = convertInt(winsize.at(ww));
		string classifiername = outputpath + "trained_classifier" + "_" + winstr_size + ".xml";
		CvBoost boostclassifier;
		boostclassifier.train(train_data,CV_ROW_SAMPLE,train_labels);
		boostclassifier.save(classifiername.c_str(),"boost");	
		train_data.release();
		train_labels.release();
	}

}

void SelectAllRois(const path& tmp_path){
	// function to be used during training to select ROIs
	// Recursive function that process files in sub directories and runs activity detection on it
	for (directory_iterator iter = directory_iterator(tmp_path); iter
			!= directory_iterator(); iter++)
	{
		directory_entry entry = *iter;
		if (is_directory(entry.path())) 
		{
			cout << "Processing directory " << entry.path().string() << endl;
			root_dir = entry.path().string();
			SelectAllRois(entry.path());			
			
		} else 
		{
			path entryPath = entry.path();
			string roi_file = root_dir+"/ROI.xml";
			if (!exists(roi_file)){
				if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg") 
				{
					// process the images:
					cout << " Processing image: " << entryPath.string() << endl;
					CurrFrame = imread(entryPath.string());
					// Read first image and then select ROI					
					Rect sel_rect = SelectRoi(CurrFrame,0);					
					FileStorage fs_w1(roi_file,FileStorage::WRITE);	
					fs_w1<< "ROIx" << sel_rect.x;
					fs_w1<< "ROIy" << sel_rect.y;
					fs_w1 << "ROIh" << sel_rect.height;
					fs_w1 << "ROIw" << sel_rect.width;
					fs_w1.release();													
				}
			}
		}
	}
}

void LoadDataModels(vector<int>& WinSizes){
	// Loads all classifiers associated to WinSizes:
	vector< CvBoost*> ClassifierList(WinSizes.size());
	for (int ww=0;ww<WinSizes.size();ww++){
		int winsize = WinSizes.at(ww);
		string win_str = convertInt(winsize);
		string classifiername =  DataPath + "trained_classifier_"+win_str+".xml";
		ClassifierList[ww] = new CvBoost();
		assert(ClassifierList[ww] != NULL);
		ClassifierList[ww]->load(classifiername.c_str(),"boost");
	}
	DataModels.push_back(ClassifierList);
	ClassifierList.clear();
}

void TestFiles(string TestPath){
	// To test files in path:
	WinSizes.push_back(60); WinSizes.push_back(90); WinSizes.push_back(120);
	// Load codebook first: (make it faster)
	Mat negcodebook;
	string tmp_path = DataPath + "seize_codebook.xml";
	FileStorage fs_w(tmp_path,FileStorage::READ);	
	fs_w["POSCBK"] >> CodeBook;
	fs_w["NEGCBK"] >> negcodebook;
	fs_w.release();
	vconcat(CodeBook,negcodebook,CodeBook);
	negcodebook.release();

	// Load classifiers next: 
	LoadDataModels(WinSizes);
	GetTestData(path(TestPath),1);
	
}

int main(int argc, char ** argv) {
	// This is an 'off-line' version of SEIZE's algorithm, does not require GUI
	// Allows for the tests to be done on 64-bit machines.

	// init params:
	SmoothFlowField_x.clear();
	SmoothFlowField_y.clear();

	cout << "Select mode of operation: 1-Training, 2-Testing:  "<<endl;
	cin >> opmode;

	string TestPath,NegTestPath,PosTestPath;
	
	if (opmode == 1)
	{
		
		//cout << "\nPath containing negative training data: "<<endl;
		//cin >> NegTestPath;
		//cout << "\nPath containing positive training data: "<<endl;
		//cin >> PosTestPath;
		//cout << "\nPath containing trained data: "<<endl;
		//cin >> OutputPath;
		
		NegTestPath = "E:/sgong/From Others/Ee Sin/TrainingData/Negative";
		PosTestPath = "E:/sgong/From Others/Ee Sin/TrainingData/Positive/New";
		OutputPath = "E:/sgong/From Others/Ee Sin/TrainingData/Output_Trial/";
		InitSubWindows2(WinSize,1,DefaultHeight,DefaultWidth);
		WinPoint.clear();
		// To use whole image:
		cv::Point  indPoint2;
		indPoint2.x = 0;
		indPoint2.y = 0;
		WinPoint.push_back(indPoint2);

		if (!exists(OutputPath)){
			// assume that if trained data folder is not available, need to collect training data, learn codebook. 
			mkdir(OutputPath.c_str());	
			SelectAllRois(path(NegTestPath));
			SelectAllRois(path(PosTestPath));
			GetCodeBooks(path(NegTestPath),path(PosTestPath),OutputPath);

		} else {
			string tmp_file = OutputPath + "seize_codebook.xml";			
			if (!exists(tmp_file)){
				SelectAllRois(path(NegTestPath));
				SelectAllRois(path(PosTestPath));
				GetCodeBooks(path(NegTestPath),path(PosTestPath),OutputPath);
			}
		}

		// Sample and Train:
		TrainClassifiers(OutputPath);




	} else if (opmode == 2){

		//cout << "\n Test Path: " << endl;
		//cin >> TestPath;
		// getline (cin, TestPath);
		//cout << "\n Output Path: " << endl;
		//cin >> OutputPath;

		DataPath	= "E:/sgong/From Others/Ee Sin/TrainingData/Models/Output_5/";
		//TestPath	= "E:/sgong/From Others/Ee Sin/TrainingData/Positive/HDD08_P20_20101108_TP1/";
		TestPath	= "G:/FAMS_Capture_1/00/2010/06/10/P06_20100610_00to08/";
		OutputPath	= "G:/FAMS_Capture_1/00/2010/06/10/P06_20100610_00to08-output_Advance2/";
		
		InitSubWindows2(WinSize,1,DefaultHeight,DefaultWidth);
		WinPoint.clear();
		// To use whole image:
		cv::Point  indPoint2;
		indPoint2.x = 0;
		indPoint2.y = 0;
		WinPoint.push_back(indPoint2);

		if (!exists(OutputPath)){
			mkdir(OutputPath.c_str());		
		} else {
			remove(OutputPath.c_str());
			mkdir(OutputPath.c_str());
		}
		// Offline testing mode:
		// To do: include an online testing one
		TestFiles(TestPath);
	}
	
}
*/

int main()
{
	int nMode;
	cout<< "1 = Training, 2 = Testing: " << endl;
	cin >> nMode;
	cin.ignore();

	ActivityDetection ActDetection;
	ActDetection.initialisation(nMode);

	if(nMode ==1)
	{
		string PosTrainPath, NegTrainPath, ModelPath;
		//cout << "Postive Train Path: \n";
		//getline (cin, PosTrainPath);	
		//
		//cout << "\nNegative Train Path: \n";
		//getline (cin, NegTrainPath);
		//
		//cout << "\nModel Output Path: \n";
		//getline (cin, ModelPath);

		PosTrainPath	= "E:/Seizure Data/TrainingData/Positive_Trimmed";
		NegTrainPath	= "E:/Seizure Data/TrainingData/Negativev2.8";
		ModelPath		= "E:/Seizure Data/Models/Output";
		
		ActDetection.initialTrain(PosTrainPath, NegTrainPath, ModelPath);
		ActDetection.SelectAllRois(path(PosTrainPath));
		ActDetection.SelectAllRois(path(NegTrainPath));
		ActDetection.SaveTrainData(path(NegTrainPath), path(PosTrainPath));
	}
	else 
	{
		string ModelPath, TestPath, OutputPath;
		//cout << "Model Path:\n";
		//getline (cin, ModelPath);	
	
		
		ModelPath	= "E:/Seizure Data/Models/Output/";
		//TestPath	= "E:/Seizure Data/TestingData/Full Data/20101221_00to07_P23/";
		//OutputPath	= "E:/Seizure Data/Output_1/Full Data/20101221_00to07_P23/";
		cout << "Test Path:\n";
		getline (cin, TestPath);
		
		cout << "\nOutput Path:\n";
		getline (cin, OutputPath);

		ActDetection.initialTest(ModelPath, TestPath, OutputPath);
		ActDetection.loadTestDataOffline(TestPath);

		/*TestPath	= "E:/Seizure Data/TestingData/Full Data/20101110_00to09_P20/";
		OutputPath	= "E:/Seizure Data/Output_1/Full Data/20101110_00to09_P20/";

		ActDetection.initialTest(ModelPath, TestPath, OutputPath);
		ActDetection.loadTestDataOffline(TestPath);*/
	}

	system("pause");

	return 0;

}