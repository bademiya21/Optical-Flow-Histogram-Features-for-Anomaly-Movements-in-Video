#pragma once
#include <cv.h>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <direct.h>
#include "MouseSelect.h"
#include "genLBP.h"					// ********* used in LBP *********//
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include "svm_common.h"
#include "svm_learn.h"
#include "svm_test.h"
using namespace std;
using namespace cv;
using namespace boost::filesystem;

class ActivityDetection
{
public:

	ActivityDetection(void);
	~ActivityDetection(void);

	//Train and Test
	void			initialisation(int nOpt);

	//Train
	void			initialTrain(string _sPosTrainPath, string _sNegTrainPath, string _sModelPath);
	void			SelectAllRois(const path& tmp_path);
	void			SaveTrainData(const path& negpath, const path& pospath);

	//Test
	void			initialTest(string _sModelPath, string _sTestPath, string _sOutputhPath);
	void			loadTestDataOffline(const path& basepath);

private:
	//Train and Test
	//Parameters:
	int				nSmoothWin;
	int				nWinSize;
	int				DefaultHeight;
	int				DefaultWidth;

	//
	int				nModeOpt;			//1 = Training, else = Testing
	int				nInitFlag;			//1 = First Frame, else = not first frame
	int				nNumBinsLBP;
	Mat				mCurrFrame;
	Mat				mPrevFrame;
	Rect			rSelectROI;
	string			sRootDir;			//Root Directory
	string			sImgFilename;		//Image filename
	string			sModelPath;
	vector<int>		vnWindSize;
	vector<Mat>		vmVx_Buff;
	vector<Mat>		vmVy_Buff;
	vector<Mat>		vmVx_Buff_old;
	vector<Mat>		vmVy_Buff_old;
	vector<Point>	vpWinPoint;

	//Train
	string			sPosTrainPath;
	string			sNegTrainPath;
	Mat				mTrainFv;
	vector<Mat>		vmAllTrainFv;
	vector<Mat>		vmPosTrainFv;
	vector<Mat>		vmNegTrainFv;
	char			restartfile[255];       /* file with initial alphas */

	//Test
	//Parameters used in saving image - Detection mode
	int					nDetInterval;
	vector<int>			vnDetectionCount;
	vector<int>			vnSubDetectionCount;
	vector<int>			vnPostDetFlag;
	vector<int>			vnDetectionFlag;
	vector<int>			vnSaveDetFameNum;
	vector<int>			vnPostDetFrameWritten;
	vector<svm_test*>* 	vSVM;
	
	int				nObservationFrames;
	int				nProbFrames;
	double			dProbSeize;
	string			sTestPath;
	string			sOutputPath;
	Mat				mCodeBook;
	vector<Mat>		vmTestFv;
	vector<vector<CvBoost*>>	DataModels;
	vector<vector<double>>		vvdProbScore;


	//=============================================================================================================
	//Train and Test
	string			convertInt(int number);
	void			InitSubWindows2(int nWinsize,int nOverlapwin, int nInput_h, int nInput_w);
	void			SmoothField(vector<Mat>& vx_buff, vector<Mat>& vy_buff, vector<Mat> &sm_field);
	void			GradOptFlow(vector<Mat> &sm_field, vector<Mat> &gradopt_field);
	void			NormaliseImg(Mat& Img, Mat& ImgOut);
	void			GetStdDevImg(Mat& Img, Mat& StdImg);
	void			ExtractOptFlowHist(Mat &m_field, Mat &ori_field, int numbins, Mat &outfeatures);
	void			ExtractLBPFeatures(Mat &Img, Mat &FVect, int numbins);
	void			ExtractLBPFeatures2(vector<Mat> &Img, Mat &FVect, int numbins);
	void			ExtractFeatures(vector<Mat> &sm_field, vector<Mat> &gradopt_field, Mat& flow_fv);
	void			GetFlowFeatures();

	//Train
	void			GetTrainFlowFeatures(Mat& mFlowFeatures);
	void			GetRawTrainData(const path& basepath);
	void			trainFeature(string featureEx, string modelfile, double weight);
	void			set_parameters(char *restartfile,long *verb, LEARN_PARM *learn_parm,KERNEL_PARM *kernel_parm);
	
	//Test 
	void			loadSVMModels();
	void			TestFlowFeatures(vector<double>& vdWinScores);
	void			GetTestFlowFeatures(Mat& mFlowFeatures);
};