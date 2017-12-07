#include "ActivityDetection.h"

ActivityDetection::ActivityDetection(void)
{
}

ActivityDetection::~ActivityDetection(void)
{
}

//================================ Train and Test Functions ===================================
void ActivityDetection::initialisation(int nOpt)
{
	nModeOpt		= nOpt;
	nInitFlag		= 1;
	nNumBinsLBP		= 0;

	//Parameters:
	vnWindSize.clear();
	vnWindSize.push_back(60);
	vnWindSize.push_back(90);
	vnWindSize.push_back(120);	// push back in increasing order

	DefaultHeight	= 256;
	DefaultWidth	= 320;
	nSmoothWin		= 5;		// number of frames to smooth the optical flow field
	nWinSize		= 64;

	InitSubWindows2(nWinSize, 1, DefaultHeight, DefaultWidth);

}

void ActivityDetection::set_parameters(char *restartfile,long *verb, LEARN_PARM *learn_parm,KERNEL_PARM *kernel_parm)
{
	char type[100];
	char msg[512];


	/* set default */
	strcpy (learn_parm->predfile, "trans_predictions");
	strcpy (learn_parm->alphafile, "");
	strcpy (restartfile, "");

	/*Set default parameters*/ /*change these to suit the program*/
	(*verb)=1;
	strcpy(type,"c"); /* c - classification, r - regression, p - ranking, o - optimization, s - optimization with slack*/
	learn_parm->svm_costratio=3.0; /* weighting of positive samples wrt negative samples i.e. if no of pos 3x less than neg, then cost is 3 but this is user defined. Any weight will do. Does not need to be in propotion to training samples.*/
	kernel_parm->kernel_type=0; /* 0 - linear svm, 1 - polynomial, 2 - RBF, 3 - sigmoid neural net, 4 - custom*/

	/* This controls the C paramter value for SVM. By default, the program decides this automatically. If necessary, change manually. 0 means automatic.*/
	learn_parm->svm_c=0.0;

	/*Below not necessary to change if linear SVM is desired*/
	learn_parm->biased_hyperplane=1;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=-9999;
	learn_parm->maxiter=100000;
	learn_parm->kernel_cache_size=40;
	learn_parm->eps=0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	strcpy(kernel_parm->custom,"empty");

	if(learn_parm->svm_iter_to_shrink == -9999) {
		if(kernel_parm->kernel_type == LINEAR) 
			learn_parm->svm_iter_to_shrink=2;
		else
			learn_parm->svm_iter_to_shrink=100;
	}
	if(strcmp(type,"c")==0) {
		learn_parm->type=CLASSIFICATION;
	}
	else if(strcmp(type,"r")==0) {
		learn_parm->type=REGRESSION;
	}
	else if(strcmp(type,"p")==0) {
		learn_parm->type=RANKING;
	}
	else if(strcmp(type,"o")==0) {
		learn_parm->type=OPTIMIZATION;
	}
	else if(strcmp(type,"s")==0) {
		learn_parm->type=OPTIMIZATION;
		learn_parm->sharedslack=1;
	}
	else {
		printf(msg,"\nUnknown type '%s': Valid types are 'c' (classification), 'r' regession, and 'p' preference ranking.\n",type);
	}    
	if((learn_parm->skip_final_opt_check) && (kernel_parm->kernel_type == LINEAR)) {
		printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
		learn_parm->skip_final_opt_check=0;
	}    
	if((learn_parm->skip_final_opt_check) && (learn_parm->remove_inconsistent)) {
		cerr<<"It is necessary to do the final optimality check when removing inconsistent \nexamples.\n"<<endl;

	}    
	if((learn_parm->svm_maxqpsize<2)) {
		cerr<<"\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n"<<learn_parm->svm_maxqpsize<<endl; 
	}
	if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
		cerr<<"Maximum size of QP-subproblems [%ld] must be larger than the number of\n"<<learn_parm->svm_maxqpsize<<endl;
		cerr<<"new variables [%ld] entering the working set in each iteration.\n"<<learn_parm->svm_newvarsinqp<<endl; 
	}
	if(learn_parm->svm_iter_to_shrink<1) {
		cerr<<"\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n"<<learn_parm->svm_iter_to_shrink<<endl;
	}
	if(learn_parm->svm_c<0) {
		cerr<<"\nThe C parameter must be greater than zero!\n\n"<<endl;
	}
	if(learn_parm->transduction_posratio>1) {
		cerr<<"\nThe fraction of unlabeled examples to classify as positives must be less than 1.0 !!!\n\n"<<endl;
	}
	if(learn_parm->svm_costratio<=0) {
		cerr<<"\nThe COSTRATIO parameter must be greater than zero!\n\n"<<endl;
	}
	if(learn_parm->epsilon_crit<=0) {
		cerr<<"\nThe epsilon parameter must be greater than zero!\n\n"<<endl;
	}
	if(learn_parm->rho<0) {
		cerr<<"\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n"<<endl;
		cerr<<"be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n"<<endl;
		cerr<<"Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n"<<endl;
		cerr<<"ending.\n"<<endl;
	}
	if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
		cerr<<"\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n"<<endl;
		cerr<<"for switching to the conventional xa/estimates described in T. Joachims,\n"<<endl;
		cerr<<"Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n"<<endl;
		cerr<<"ending\n"<<endl;
	}
}

void ActivityDetection::InitSubWindows2(int nWinsize,int nOverlapwin, int nInput_h, int nInput_w)
{
	// 26 Sep 2013
	// with overlap:
	/*	cv::Point  indPoint2;
	if (nOverlapwin == 1)
	{		
	for(int r=0; r<nInput_h; r+=nWinsize/2){
	for(int c=0; c<nInput_w; c+=nWinsize/2){	
	if (r+nWinsize < nInput_h && c+nWinsize < nInput_w){
	indPoint2.x = c;
	indPoint2.y = r;
	vpWinPoint.push_back(indPoint2);			
	vpWinPoint.push_back(Point(c,r));	
	}
	}
	}
	} 
	else {
	for(int r=0; r<nInput_h; r+=nWinsize){
	for(int c=0; c<nInput_w; c+=nWinsize){	
	indPoint2.x = c;
	indPoint2.y = r;
	vpWinPoint.push_back(indPoint2);			
	}
	}
	}
	*/

	//To use whole image:
	vpWinPoint.clear();
	vpWinPoint.push_back(Point(0,0));
}

string ActivityDetection::convertInt(int number)
{
	stringstream ss;	//create a stringstream
	ss << number;	//add number to the stream
	return ss.str();	//return a string with the contents of the stream
}

void ActivityDetection::SmoothField(vector<Mat>& vx_buff, vector<Mat>& vy_buff, vector<Mat> &sm_field)
{
	sm_field.clear();

	// function to obtain smoothed optical flow fields
	Mat vx_temp 		= Mat::zeros(Size(vx_buff[0].cols, vx_buff[0].rows),CV_32FC1);
	Mat	vy_temp 		= Mat::zeros(Size(vy_buff[0].cols, vy_buff[0].rows),CV_32FC1);
	
	for (int ii = 0; ii< (int)vx_buff.size();ii++)
	{
		vx_temp = vx_temp + vx_buff[ii];
		vy_temp = vy_temp + vy_buff[ii];
	}
	vx_temp = vx_temp/nSmoothWin;
	//medianBlur(vx_temp, vx_temp, 5);
	sm_field.push_back(vx_temp);
	vy_temp = vy_temp/nSmoothWin;
	//medianBlur(vy_temp, vy_temp, 5);
	sm_field.push_back(vy_temp);
	
	vx_temp.release();
	vy_temp.release();
}

void ActivityDetection::GradOptFlow(vector<Mat> &sm_field, vector<Mat> &gradopt_field)
{
	Mat temp    = Mat::zeros(Size(sm_field[0].cols, sm_field[0].rows),CV_32FC1);
	Rect ROI(1, 1, sm_field[0].cols-2, sm_field[0].rows-2);
	Mat kernelx = (Mat_<float>(1,3)<<1, 0, -1);
	Mat kernely = (Mat_<float>(3,1)<<1, 0, -1);

	for (int ii=0; ii < (int)sm_field.size(); ii++){
		//Sobel(sm_field[ii], temp, -1, 1, 0, 1);
		filter2D(sm_field[ii], temp, -1, kernelx);
		gradopt_field.push_back(temp(ROI));
		temp.release();
		//Sobel(sm_field[ii], temp, -1, 0, 1, 1);
		filter2D(sm_field[ii], temp, -1, kernely);
		gradopt_field.push_back(temp(ROI));
		temp.release();
	}

	kernelx.release();
	kernely.release();
}

void ActivityDetection::NormaliseImg(Mat& Img, Mat& ImgOut)
{
	// Note: Images must be greyscale:	
	Mat Img32F;
	Img.convertTo(Img32F,CV_32F);
	normalize(Img32F,ImgOut,0.0,1.0,NORM_MINMAX);
}

void ActivityDetection::GetStdDevImg(Mat& Img, Mat& StdImg)
{
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

void ActivityDetection::ExtractOptFlowHist(Mat &m_field, Mat &ori_field, int numbins, Mat &outfeatures)
{
	// basic function to get features:
	// quantize ori_field into histogram bin representation
	Mat temp_bin, temp;
	float bin_range = 2*M_PI/numbins;
	temp = ori_field/bin_range;
	temp_bin = Mat::zeros(Size(ori_field.cols,ori_field.rows),CV_8UC1);
	for (int rows = 0; rows < ori_field.rows; rows++){
		for (int cols = 0; cols < ori_field.cols; cols++){
			temp_bin.at<uchar>(rows,cols) = cvFloor(temp.at<float>(rows,cols));
			if (temp_bin.at<uchar>(rows,cols) == numbins)
				temp_bin.at<uchar>(rows,cols) = 0;
		}
	}
	temp.release();

	outfeatures = Mat::zeros(1,numbins, CV_32FC1);
	
	//Scalar m_field_mean,m_field_std;
	//meanStdDev(m_field,m_field_mean,m_field_std);
	//Scalar m_field_thres = mean(m_field);

	for (int ii=0; ii < (int)vpWinPoint.size();ii++){
		for (int rows = 0; rows < m_field.rows; rows++){
			for (int cols = 0; cols < m_field.cols; cols++){
				//if (m_field.at<float>(rows,cols) >= (m_field_mean[0]+m_field_std[0]))
					outfeatures.at<float>(0,temp_bin.at<uchar>(rows,cols)) += m_field.at<float>(rows,cols);
			}
		}
	}
	temp_bin.release();
}

void ActivityDetection::ExtractLBPFeatures(Mat &Img, Mat &FVect, int numbins)
{

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
	int width = nWinSize;
	int height = nWinSize;

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
	lbp.R.xR = 1; lbp.R.yR = 1;	//range
	lbp.SN.xy = 8; //number of points
	lbp.uni = 1; //uniform lbp
	lbp.interp = 1; //interpolation for points
	lbp.norm = 0; //normalization of histogram
	lbp.riu2 = 0; // riu2 lbp

	//Extract LBP for each sub-window
	for (int ii=0; ii < (int)vpWinPoint.size();ii++)
	{
		Rect winRoi = Rect(vpWinPoint.at(ii).x,vpWinPoint.at(ii).y,winsize_x,winsize_y);
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
		roiImg.release();

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

		nNumBinsLBP = lbp.uni_bin.xy+1;

		Mat TempHist = Mat::zeros(1,nNumBinsLBP,CV_32FC1);;

		for(i=0; i<lbp.uni_bin.xy+1; i++)
		{
			TempHist.at<float>(i)= lbp.uni_hist.pHist_xy[i];
		}
		//normalize(TempHist,TempHist,1,0,NORM_L2);
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

void ActivityDetection::ExtractLBPFeatures2(vector<Mat> &Img, Mat &FVect, int numbins)
{

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
	int width = nWinSize;
	int height = nWinSize;

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
	lbp.R.xR = 1; lbp.R.yR = 1;	//range
	lbp.SN.xy = 8; //number of points
	lbp.uni = 1; //uniform lbp
	lbp.interp = 1; //interpolation for points
	lbp.norm = 0; //normalization of histogram
	lbp.riu2 = 0; // riu2 lbp

	Mat tempImg, TempHist;

	for (int buff=0;buff<Img.size();buff++){
		tempImg = Img[buff];
		//Extract LBP for each sub-window
		for (int ii=0; ii < (int)vpWinPoint.size();ii++)
		{
			Rect winRoi = Rect(vpWinPoint.at(ii).x,vpWinPoint.at(ii).y,winsize_x,winsize_y);
			Mat roiImg = tempImg(winRoi);

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

			nNumBinsLBP = lbp.uni_bin.xy+1;
			if (buff==0){
				TempHist = Mat::zeros(Size(nNumBinsLBP,1),CV_32FC1);
			}

			for(int l=0; l<lbp.uni_bin.xy+1; l++)
			{
				TempHist.at<float>(l) += lbp.uni_hist.pHist_xy[l];
			}
		}
	}
	//normalize(TempHist,TempHist,1,0,NORM_L1);
	//cv::sqrt(TempHist,TempHist);
	FVect.push_back(TempHist);
	TempHist.release();

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

void ActivityDetection::ExtractFeatures(vector<Mat> &sm_field, vector<Mat> &gradopt_field, Mat& flow_fv)
{
	// Function to extract flow features:
	flow_fv.release();
	Mat f_features;
	int f_bins = 36;
	double maxvalue = 0.0;
	Mat mag_field;//	= Mat::zeros(Size(sm_field[0].cols, sm_field[0].rows),CV_32FC1);
	Mat ori_field;//	= Mat::zeros(Size(sm_field[0].cols, sm_field[0].rows),CV_32FC1);

	// Histogram of optical flow
	cartToPolar(sm_field[0], sm_field[1], mag_field, ori_field);
	ExtractOptFlowHist(mag_field,ori_field,f_bins,f_features);
	//normalize(f_features,f_features,1,0,NORM_L2);
	//minMaxLoc(f_features,NULL,&maxvalue,NULL,NULL);
	//f_features /= maxvalue;
	f_features.copyTo(flow_fv);
	f_features.release();
	mag_field.release();
	ori_field.release();

	/*Visualize optical flow*/
	/*translate magnitude to range [0;1]*/

	//cartToPolar(sm_field[0], sm_field[1], mag_field, ori_field, true);

	//double mag_max = 0.00;
	//minMaxLoc(mag_field, 0, &mag_max);
	//mag_field.convertTo(mag_field, -1, 1.0/mag_max);

	////build hsv image
	//Mat _hsv[3], hsv;
	//_hsv[0] = ori_field;
	//_hsv[1] = Mat::ones(ori_field.size(), CV_32F);
	//_hsv[2] = mag_field;
	//merge(_hsv, 3, hsv);

	////convert to BGR and show
	//Mat bgr;//CV_32FC3 matrix
	//cvtColor(hsv, bgr, COLOR_HSV2BGR);
	//imshow("optical flow", bgr);
	//waitKey(500);

	//mag_field.release();
	//ori_field.release();
	/*Visualization done*/

	// Motion Boundary Histograms
	cartToPolar(gradopt_field[0], gradopt_field[1], mag_field, ori_field);
	ExtractOptFlowHist(mag_field,ori_field,f_bins,f_features);
	//normalize(f_features,f_features,1,0,NORM_L2);
	//minMaxLoc(f_features,NULL,&maxvalue,NULL,NULL);
	//f_features /= maxvalue;
	hconcat(flow_fv,f_features,flow_fv);
	f_features.release();
	mag_field.release();
	ori_field.release();

	cartToPolar(gradopt_field[2], gradopt_field[3], mag_field, ori_field);
	ExtractOptFlowHist(mag_field,ori_field,f_bins,f_features);
	//normalize(f_features,f_features,1,0,NORM_L2);
	//minMaxLoc(f_features,NULL,&maxvalue,NULL,NULL);
	//f_features /= maxvalue;
	hconcat(flow_fv,f_features,flow_fv);
	f_features.release();
	mag_field.release();
	ori_field.release();

	// Internal Motion Histograms
	//cartToPolar(gradopt_field[0], gradopt_field[2], mag_field, ori_field);
	//ExtractOptFlowHist(mag_field,ori_field,f_bins,f_features);
	////normalize(f_features,f_features,1,0,NORM_L2);
	////minMaxLoc(f_features,NULL,&maxvalue,NULL,NULL);
	////f_features /= maxvalue;
	//hconcat(flow_fv,f_features,flow_fv);
	//f_features.release();
	//mag_field.release();
	//ori_field.release();

	//cartToPolar(gradopt_field[1], gradopt_field[3], mag_field, ori_field);
	//ExtractOptFlowHist(mag_field,ori_field,f_bins,f_features);
	////normalize(f_features,f_features,1,0,NORM_L2);
	////minMaxLoc(f_features,NULL,&maxvalue,NULL,NULL);
	////f_features /= maxvalue;
	//hconcat(flow_fv,f_features,flow_fv);
	//f_features.release();
	//mag_field.release();
	//ori_field.release();

	// Texture - LBP
	//ExtractLBPFeatures(sm_field, f_features, 0);	
	//hconcat(tmp_flow_fv,f_features,tmp_flow_fv);	
	//f_features.release();

	// Texture - STD:
	//Mat StdImg;
	//GetStdDevImg(sm_field, StdImg);
	//ExtractWinFV_Params(StdImg,f_bins,f_min,f_max,f_features);	
	//hconcat(tmp_flow_fv,f_features,tmp_flow_fv);
	//// FrameFv.push_back(FVect);
	//f_features.release();

	// Pull and straighten the flow_fv:
	//flow_fv = tmp_flow_fv.reshape(0,1).clone();
	//tmp_flow_fv.release();
	//minMaxLoc(flow_fv,NULL,&maxvalue,NULL,NULL);
	//flow_fv /= maxvalue;*/
	
}


void ActivityDetection::GetFlowFeatures()
{
	// function to calculate dense optical flow, and to get flow features
	Mat prevframe_gray,currframe_gray;
	Mat mFlow_features;
	Mat FlowField;
	vector<Mat> mSM_field;
	vector<Mat> gradopt_field;
	vector<Mat> flowchannels(2);

	//blur(mPrevFrame, mPrevFrame, Size(3,3));
	cvtColor(mPrevFrame,prevframe_gray,CV_RGB2GRAY);
	//blur(mCurrFrame, mCurrFrame, Size(3,3));
	cvtColor(mCurrFrame,currframe_gray,CV_RGB2GRAY);

	FlowField = Mat::zeros(currframe_gray.rows,currframe_gray.cols,CV_32FC2);

	//imshow("Previous Frame",prevframe_gray);
	//waitKey(0);
	//imshow("Current Frame (before adjustment)",currframe_gray);
	//waitKey(0);

	//SG 10 Feb 2014-->> 
	//normalize(prevframe_gray,prevframe_gray,0,255,NORM_MINMAX,-1);
	//normalize(currframe_gray,currframe_gray,0,255,NORM_MINMAX,-1);
	double dMin1 = 0.00, dMin2 = 0.00, dMax1 = 0.00, dMax2 = 0.00;
	double bright_adjust = 0.00;
	minMaxLoc(prevframe_gray,&dMin1,&dMax1);
	minMaxLoc(currframe_gray,&dMin2,&dMax2);
	//normalize(prevframe_gray,prevframe_gray,dMin1,dMax1,NORM_MINMAX,-1);
	prevframe_gray = (prevframe_gray - dMin1)*255/(dMax1 - dMin1);
	//normalize(currframe_gray,currframe_gray,dMin2,dMax2,NORM_MINMAX,-1);
	currframe_gray = (currframe_gray - dMin2)*255/(dMax2 - dMin2);
	//<<--
	//imshow("Previous Frame (after adjustment)",prevframe_gray);
	//waitKey(0);
	//imshow("Current Frame (after adjustment)",currframe_gray);
	//waitKey(0);

	//calcOpticalFlowFarneback(prevframe_gray, currframe_gray, FlowField, 0.5, 3, 8, 2, 3, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);// Original
	calcOpticalFlowFarneback(prevframe_gray, currframe_gray, FlowField, 0.5, 5, 9, 5, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
	split(FlowField,flowchannels);

	//medianBlur(flowchannels[0], flowchannels[0], 5);
	//medianBlur(flowchannels[0], flowchannels[0], 5);
	//medianBlur(flowchannels[0], flowchannels[0], 5);
	vmVx_Buff.push_back(flowchannels[0](rSelectROI));
	//medianBlur(flowchannels[1], flowchannels[1], 5);
	//medianBlur(flowchannels[1], flowchannels[1], 5);
	//medianBlur(flowchannels[1], flowchannels[1], 5);
	vmVy_Buff.push_back(flowchannels[1](rSelectROI));

	if (vmVx_Buff.size() == nSmoothWin)
	{
		mSM_field.clear();
		gradopt_field.clear();
		
		// calculate smooth optical flow:
		SmoothField(vmVx_Buff,vmVy_Buff,mSM_field);
		GradOptFlow(mSM_field,gradopt_field); // Compute differentials of optical flow for x and y components
		ExtractFeatures(mSM_field,gradopt_field,mFlow_features);

		if (nModeOpt == 1) 	GetTrainFlowFeatures(mFlow_features);
		else 				GetTestFlowFeatures(mFlow_features);

		// to maintain flow fields at SmoothWin
		vmVy_Buff.erase(vmVy_Buff.begin());
		vmVx_Buff.erase(vmVx_Buff.begin());

	}

	mFlow_features.release();
	FlowField.release();
	prevframe_gray.release();
	currframe_gray.release();
	flowchannels.clear();
}

//====================================== Train Functions ======================================= 
void ActivityDetection::initialTrain(string _sPosTrainPath, string _sNegTrainPath, string _sModelPath)
{
	sPosTrainPath	= _sPosTrainPath;
	sNegTrainPath	= _sNegTrainPath;
	sModelPath		= _sModelPath + "/" ;
}

void ActivityDetection::SelectAllRois(const path& tmp_path)
{
	// function to be used during training to select ROIs
	// Recursive function that process files in sub directories and runs activity detection on it
	for (directory_iterator iter = directory_iterator(tmp_path); iter != directory_iterator(); iter++)
	{
		directory_entry entry = *iter;
		if (is_directory(entry.path())) 
		{
			cout << "Processing directory " << entry.path().string() << endl;
			sRootDir = entry.path().string();
			SelectAllRois(entry.path());			

		} else 
		{
			path 	entryPath = entry.path();
			string	roi_file = sRootDir+"/ROI.xml";
			if (!exists(roi_file))
			{
				if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg") 
				{
					// process the images:
					cout << " Processing image: " << entryPath.string() << endl;
					mCurrFrame = imread(entryPath.string());
					// Read first image and then select ROI					
					Rect sel_rect = SelectRoi(mCurrFrame,0);					
					FileStorage fs_w1(roi_file,FileStorage::WRITE);	
					fs_w1 << "ROIx" << sel_rect.x;
					fs_w1 << "ROIy" << sel_rect.y;
					fs_w1 << "ROIh" << sel_rect.height;
					fs_w1 << "ROIw" << sel_rect.width;
					fs_w1.release();													
				}
			}
		}
	}
}

// Added by AS - 18 Mar 2014 *For SVM training*
void ActivityDetection::SaveTrainData(const path& negpath, const path& pospath)
{
	/*Function to collect all raw data and save for SVM Training*/
	vmNegTrainFv.clear();
	vmPosTrainFv.clear();

	//Positive
	vmAllTrainFv.clear();
	GetRawTrainData(pospath);
	vmPosTrainFv = vmAllTrainFv;

	//Negative
	vmAllTrainFv.clear();
	GetRawTrainData(negpath);	
	vmNegTrainFv = vmAllTrainFv;
	
	vmAllTrainFv.clear();

	vector<vector<float>>	traindata;
	vector<int>				traindatalabel;
	vector<float>			temp;
	int						pos_num, neg_num, tmp_win, seq_len, TargetType, LabelType, nTotImg, nFeatDim;
	double					weight, norm_factor, maxvalue;
	string					model_path, tmp_path;
	FILE					*m_featureExt;
	Mat						one_seq, tmp_sample, tmp_sample_win, output_hist, output_hist_temp, tmp;

	tmp_path = sModelPath + "Data/";
	path folder_pos(tmp_path);
	create_directory(folder_pos);

	for (int ww=0; ww < (int)vnWindSize.size();ww++)
	{
		tmp_win = vnWindSize[ww];
		pos_num = 0;
		neg_num = 0;

		// Samples positive sequence according to winsize:
		for (int ii = 0; ii < (int)vmPosTrainFv.size();ii++)
		{
			one_seq = vmPosTrainFv[ii].clone();
			seq_len = one_seq.rows;
			if (seq_len > tmp_win-1)
			{
				for (int rr = vnWindSize[2]-tmp_win; rr < seq_len-tmp_win;rr++)
				{
					tmp_sample = one_seq.rowRange(rr,rr+tmp_win).clone();
					for (int tw = 0; tw < (tmp_win-9) ; tw+=5){
						tmp_sample_win = tmp_sample.rowRange(tw,tw+10).clone();						
						reduce(tmp_sample_win, output_hist_temp, 0, CV_REDUCE_SUM, -1);
						//normalize(output_hist_temp,output_hist_temp,1,0,NORM_L2);
						if(output_hist.empty())
							output_hist_temp.copyTo(output_hist);
						else
							hconcat(output_hist,output_hist_temp,output_hist);
												
						tmp_sample_win.release();
						output_hist_temp.release();
					}
					normalize(output_hist,output_hist,1,0,NORM_L2);
					/*for (int i=0; i < output_hist.cols; i++){
						if (output_hist.at<float>(0,i) > 0.065)
							output_hist.at<float>(0,i) = 0.065;
					}
					normalize(output_hist,output_hist,1,0,NORM_L2);*/
					//minMaxLoc(output_hist,NULL,&maxvalue,NULL,NULL);
					//output_hist /= maxvalue;
					
					// Copy histogram to vector
					const float* p1 = output_hist.ptr<float>(0);
					temp.insert(temp.begin(),p1, p1 + output_hist.cols);
					traindata.push_back(temp);
					traindatalabel.push_back(1);
					pos_num++;

					output_hist.release();
					tmp_sample.release();
					temp.clear();
				}
			}
			one_seq.release();
		}

		// Samples negative sequence according to winsize:
		for (int ii = 0; ii < (int)vmNegTrainFv.size();ii++)
		{
			one_seq = vmNegTrainFv[ii].clone();
			seq_len = one_seq.rows;
			if (seq_len > tmp_win-1)
			{
				for (int rr = 0; rr < seq_len-tmp_win;rr++)
				{
					tmp_sample = one_seq.rowRange(rr,rr+tmp_win).clone();
					for (int tw = 0; tw < (tmp_win-9) ; tw+=5){
						tmp_sample_win = tmp_sample.rowRange(tw,tw+10).clone();						
						reduce(tmp_sample_win, output_hist_temp, 0, CV_REDUCE_SUM, -1);
						//normalize(output_hist_temp,output_hist_temp,1,0,NORM_L2);
						if(output_hist.empty())
							output_hist_temp.copyTo(output_hist);
						else
							hconcat(output_hist,output_hist_temp,output_hist);
												
						tmp_sample_win.release();
						output_hist_temp.release();
					}
					normalize(output_hist,output_hist,1,0,NORM_L2);
					/*for (int i=0; i < output_hist.cols; i++){
						if (output_hist.at<float>(0,i) > 0.065)
							output_hist.at<float>(0,i) = 0.065;
					}
					normalize(output_hist,output_hist,1,0,NORM_L2);*/
					//minMaxLoc(output_hist,NULL,&maxvalue,NULL,NULL);
					//output_hist /= maxvalue;

					// Copy histogram to vector
					const float* p1 = output_hist.ptr<float>(0);
					temp.insert(temp.begin(),p1, p1 + output_hist.cols);
					traindata.push_back(temp);
					traindatalabel.push_back(-1);
					neg_num++;

					output_hist.release();
					tmp_sample.release();
					temp.clear();
				}
			}
			one_seq.release();
		}

		tmp_path = sModelPath + "Data/" + "trainingdata_" + convertInt(tmp_win) + ".blt";

		TargetType = 4; /*5 - double values, 4 - float values, 3 - int values  */
		LabelType = 3; /*3 - int labelling of class types i.e. 1, -1, 2, 3; 4 - float labelling of classes i.e. 2.0, 1.0,-1.0...; 5 - double type labelling of classes*/
		nTotImg = (int)traindata.size();
		nFeatDim = (int)traindata[0].size();

		m_featureExt = fopen(tmp_path.c_str(), "wb");
		fwrite(&TargetType,sizeof(int),1,m_featureExt); //read double type target value
		fwrite(&LabelType,sizeof(int),1,m_featureExt);
		fwrite(&nTotImg,sizeof(int),1,m_featureExt);
		fwrite(&nFeatDim,sizeof(int),1,m_featureExt);
		for (int ii=0; ii < nTotImg; ii++)
		{
			fwrite(&traindatalabel[ii],sizeof(int),1,m_featureExt);
			fwrite(&traindata[ii][0], sizeof(float), nFeatDim, m_featureExt);
		}
		fclose(m_featureExt);

		traindata.clear();
		traindatalabel.clear();

		model_path = sModelPath + "trained_classifier" + "_" + convertInt(tmp_win) + ".xml";

		//Get model train file
		//weight = (double)neg_num/(double)pos_num;
		weight = 1.0;
		printf("Training Linear SVM classifier...\n");
		trainFeature(tmp_path, model_path, weight);
	}
}


void ActivityDetection::trainFeature(string featureEx, string modelFile, double weight)
{
	/*
	Remarks  	: Pass feature file to SVM, to train and get .model file
	Paramters	: -
	*/

	DOC **docs;  /* training examples */
	long totwords,totdoc,i;
	double *target;
	double *alpha_in=NULL;
	KERNEL_CACHE *kernel_cache;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	MODEL *model=(MODEL *)my_malloc(sizeof(MODEL));
	long verb;

	set_parameters(restartfile,&verb,&learn_parm,&kernel_parm);
	learn_parm.svm_costratio=weight;
	printf("Read all input parameters!\n");

	read_binary_documents(featureEx.c_str(),&docs,&target,&totwords,&totdoc, &verb);
	printf("Read all examples into memory!\n");

	///* TODO modify to accept this array */
	if(restartfile[0]) alpha_in=read_alphas(restartfile,totdoc);

	if(kernel_parm.kernel_type == LINEAR) { /* don't need the cache */
		kernel_cache=NULL;
	}
	else {
		/* Always get a new kernel cache. It is not possible to use the
		same cache for two different training runs */
		kernel_cache=kernel_cache_init(totdoc,learn_parm.kernel_cache_size);
	}

	if(learn_parm.type == CLASSIFICATION) {
		svm_learn_classification(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,kernel_cache,model,alpha_in);

	}
	else if(learn_parm.type == REGRESSION) {
		svm_learn_regression(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,&kernel_cache,model);
	}
	else if(learn_parm.type == RANKING) {
		svm_learn_ranking(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,&kernel_cache,model);
	}
	else if(learn_parm.type == OPTIMIZATION) {
		svm_learn_optimization(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,kernel_cache,model,alpha_in);
	}
	else {
		cerr<<"\n unknown learning parameter type!\n"<<endl;
	}

	if(kernel_cache) {
		/* Free the memory used for the cache. */
		kernel_cache_cleanup(kernel_cache);
	}

	write_binary_model(modelFile.c_str(),model);

	free(alpha_in);
	free_model(model,0);
	for (i=0;i<totdoc;i++)
		free_example(docs[i],1);
	free(docs);
	free(target);
}

void ActivityDetection::GetRawTrainData(const path& basepath) 
{
	// Recursive function that process files in sub directories and runs activity detection on it
	for (directory_iterator iter = directory_iterator(basepath); iter != directory_iterator(); iter++)
	{
		directory_entry entry = *iter;
		if (is_directory(entry.path())) 
		{
			cout << "Processing directory " << entry.path().string() << endl;
			sRootDir = entry.path().string();
			GetRawTrainData(entry.path());
			// Note: This only works if training data are separated into sub-folders, one-level down from input-folder
			if (!mTrainFv.empty())
			{
				// finished gathering data for one sub-folder:
				vmAllTrainFv.push_back(mTrainFv);
				vmVx_Buff.clear();
				vmVy_Buff.clear();
				mTrainFv.release();
			}
			nInitFlag = 1;
		} 
		else 
		{
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg") 
			{
				// process the images:
				cout << " Processing image: " << entryPath.string() << endl;
				mCurrFrame = imread(entryPath.string());				
				if (nInitFlag == 1)
				{
					string roifile = sRootDir+"/ROI.xml";
					if (!exists(roifile)){
						// Read first image and then select ROI					
						rSelectROI = SelectRoi(mCurrFrame,0);
						nInitFlag = 0;
						mPrevFrame.release(); // initialise
					} else {
						// Read from .xml:						
						FileStorage fs_w(roifile,FileStorage::READ);	
						fs_w["ROIx"] >> rSelectROI.x;
						fs_w["ROIy"] >> rSelectROI.y;
						fs_w["ROIh"] >> rSelectROI.height;
						fs_w["ROIw"] >> rSelectROI.width;
						fs_w.release();
						nInitFlag = 0;
					}
				}

				// Begin processing the images using optical flow:
				if (mPrevFrame.empty())
				{
					mCurrFrame.copyTo(mPrevFrame);
				} 
				else 
				{
					// sufficient number of frames to start:
					GetFlowFeatures();
					mCurrFrame.copyTo(mPrevFrame);
				}									
			}
		}
	}
}


void ActivityDetection::GetTrainFlowFeatures(Mat& mFlowFeatures)
{
	if (mTrainFv.empty())	
		mFlowFeatures.copyTo(mTrainFv);
	else
		vconcat(mTrainFv,mFlowFeatures,mTrainFv);
}


//====================================== Test Functions ======================================= 
void ActivityDetection::initialTest(string _sModelPath, string _sTestPath, string _sOutputhPath)
{
	sModelPath			= _sModelPath;
	sTestPath			= _sTestPath;
	sOutputPath			= _sOutputhPath;
	
	//Check if the Output Path exist
    if(!boost::filesystem::exists(sOutputPath))
        boost::filesystem::create_directories(sOutputPath);

	nObservationFrames	= vnWindSize[(vnWindSize.size()-1)];	// Longest window:
	nProbFrames			= 30;
	dProbSeize			= 0.9;

	vSVM = new vector<svm_test*>;
	for(int i=0;i<(int)vnWindSize.size();i++){
		(*vSVM).push_back(new svm_test);
	}
	loadSVMModels();

	//===Initialise Saving Detection Results ===//
	//Create Output Folders for each vnWindSize
	int nNumWind = (int)vnWindSize.size();
	for(int i = 0; i < nNumWind; i++)
	{
		string sWin = convertInt(vnWindSize[i]);
		string sTmp = sOutputPath + "WindSize_" + sWin + "/";
		mkdir(sTmp.c_str());
	}
	nDetInterval	= 600;
	vnDetectionCount.assign(nNumWind,0);
	vnSubDetectionCount.assign(nNumWind,0);
	vnPostDetFlag.assign(nNumWind,0);
	vnDetectionFlag.assign(nNumWind,0);
	vnSaveDetFameNum.assign(nNumWind,0);
	vnPostDetFrameWritten.assign(nNumWind,0);
}

void ActivityDetection::loadSVMModels()
{
	int		nload;
	string	classifiername;

	// Loads all classifiers associated to WinSizes:
	for (int ww = 0; ww < (int)vnWindSize.size(); ww++)
	{
		classifiername	= sModelPath + "trained_classifier_"+convertInt(vnWindSize[ww])+".xml";
		nload = (vSVM->at(ww))->loadmodel(classifiername.c_str());
	}
}

void ActivityDetection::loadTestDataOffline(const path& basepath)
{
	for (directory_iterator iter = directory_iterator(basepath); iter != directory_iterator(); iter++)
	{
		directory_entry entry = *iter;
		if (is_directory(entry.path())) 
		{
			cout << "Processing directory \n" << entry.path().string() << endl;
			cout << "____________________________________________________________" << endl; 
			sRootDir = entry.path().string();
			loadTestDataOffline(entry.path());
		}
		else
		{
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg")
			{
				cout << "Processing image: " << entryPath.string() << endl;
				mCurrFrame = imread(entryPath.string());				
				if (nInitFlag == 1)
				{
					string roifile = sRootDir+"/ROI.xml";
					if (!exists(roifile))
					{
						// Read first image and then select ROI					
						rSelectROI = SelectRoi(mCurrFrame,0);
						nInitFlag = 0;
						mPrevFrame.release(); // initialise
					} else 
					{
						// Read from .xml:						
						FileStorage fs_w(roifile,FileStorage::READ);	
						fs_w["ROIx"] >> rSelectROI.x;
						fs_w["ROIy"] >> rSelectROI.y;
						fs_w["ROIh"] >> rSelectROI.height;
						fs_w["ROIw"] >> rSelectROI.width;
						fs_w.release();
						nInitFlag = 0;
					}
				}
				// Begin processing the images using optical flow:
				if (mPrevFrame.empty())
				{
					mCurrFrame.copyTo(mPrevFrame);
				} 
				else 
				{
					//Get the image filename
					string		sImgPath	 = entryPath.string();
					unsigned	uTmp 	  	 = sImgPath.find_last_of("/\\");
					sImgFilename = sImgPath.substr(uTmp+1);

					// sufficient number of frames to start:
					GetFlowFeatures();
					mCurrFrame.copyTo(mPrevFrame);
				}
			}
		}
	}
}

void ActivityDetection::TestFlowFeatures(vector<double>& vdWinScores)
{
	// function to test flow features:
	Mat tmp, tmp_sample, tmp_sample_win, output_hist, output_hist_temp;
	int tmp_win;
	vector<float> temp;
	double dProbEst, norm_factor, weight, maxvalue;

	// sampling begins:
	for (int ww=0; ww< (int)vnWindSize.size(); ww++)
	{
		tmp_win = vnWindSize[ww];

		temp.clear();

		// Samples sequence according to winsize:
		for (int ii = vmTestFv.size()-tmp_win;ii<(int)vmTestFv.size();ii+=tmp_win)
		{
			// one test-seq for each window size:
			for (int rr=ii;rr<ii+tmp_win;rr++){
				tmp = vmTestFv[rr].clone();
				tmp_sample.push_back(tmp);
				tmp.release();
			}
			for (int tw = 0; tw < (tmp_win-9) ; tw+=5){
				tmp_sample_win = tmp_sample.rowRange(tw,tw+10).clone();						
				reduce(tmp_sample_win, output_hist_temp, 0, CV_REDUCE_SUM, -1);
				//normalize(output_hist_temp,output_hist_temp,1,0,NORM_L2);
				if(output_hist.empty())
					output_hist_temp.copyTo(output_hist);
				else
					hconcat(output_hist,output_hist_temp,output_hist);
										
				tmp_sample_win.release();
				output_hist_temp.release();
			}
			normalize(output_hist,output_hist,1,0,NORM_L2);
			/*for (int i=0; i < output_hist.cols; i++){
				if (output_hist.at<float>(0,i) > 0.065)
					output_hist.at<float>(0,i) = 0.065;
			}
			normalize(output_hist,output_hist,1,0,NORM_L2);*/
			//minMaxLoc(output_hist,NULL,&maxvalue,NULL,NULL);
			//output_hist /= maxvalue;

			// Copy histogram to vector
			const float* p1 = output_hist.ptr<float>(0);
			temp.insert(temp.begin(),p1, p1 + output_hist.cols);
			
			dProbEst = vSVM->at(ww)->operator()(temp);
			vdWinScores.push_back(dProbEst);

			output_hist.release();
			tmp_sample.release();
			temp.clear();
		}
	}
}

void ActivityDetection::GetTestFlowFeatures(Mat& mFlowFeatures)
{
	string win_str, tmpstring;
	double probscore;
	vector<double> WinScores;
	vector<double> tmp_score;
	FILE *tmpFile_det2, *tmpFile_det3;

	vmTestFv.push_back(mFlowFeatures);
		
	if ((int)vmTestFv.size() > (nObservationFrames-1))
	{
		WinScores.clear();
		// enough frames to test:
		TestFlowFeatures(WinScores);
			
		for (int ww = 0; ww < (int)vnWindSize.size(); ww++)
		{
			// Write data to files:
			win_str = convertInt(vnWindSize.at(ww));
			
			tmpstring = sOutputPath + "raw_det_score_"+win_str+".txt";
			tmpFile_det2 = fopen(tmpstring.c_str(),"a+");
			fprintf(tmpFile_det2,"%.05f\n",WinScores.at(ww));
			fclose(tmpFile_det2);

			// Smoothing of scores: 
			// store in vvdProbScore:
			if (vvdProbScore.size()<vnWindSize.size())
			{
				if (WinScores.at(ww)>0)
					tmp_score.push_back(1.0);
				else 
					tmp_score.push_back(0.0);
				
				vvdProbScore.push_back(tmp_score);
			} 
			else 
			{
				tmp_score = vvdProbScore.at(ww);
				if (WinScores.at(ww)>0)
					tmp_score.push_back(1.0);
				else 
					tmp_score.push_back(0.0);
										
				if ((int)tmp_score.size() > (nProbFrames-1))
				{					
					// Calculate probability and store:
					probscore = 0.0;
					for (int kk = 0; kk < (int)tmp_score.size();kk++){
						probscore += tmp_score.at(kk)/nProbFrames;
					}
						
					tmpstring = sOutputPath + "prob_score_"+win_str+".txt";
					tmpFile_det3 = fopen(tmpstring.c_str(),"a+");
					fprintf(tmpFile_det3,"%s %.05f\n",sImgFilename.c_str(), probscore);
					fclose(tmpFile_det3);

					//SG 11 Feb 2014 Saving of Detected Images -->>
					string tmpstring, ImgStr;
					if (probscore >= dProbSeize)
					{			
						// Check if it is the start of a new detection:
						if((vnPostDetFlag[ww] ==0)&& (vnDetectionFlag[ww] == 0))
						{
							vnDetectionCount[ww] 		= vnDetectionCount[ww] + 1;
							vnSubDetectionCount[ww]     = 0;
							vnPostDetFlag[ww] 			= 1;
							vnDetectionFlag[ww] 		= 1;
							vnSaveDetFameNum[ww]		= 1;
							vnPostDetFrameWritten[ww]	= 0;
							
							tmpstring 				= sOutputPath + "WindSize_" + convertInt(vnWindSize[ww]) + "/" + "Detection_" + convertInt(vnDetectionCount[ww]) + "_" + convertInt(vnSubDetectionCount[ww]) + "/";
							ImgStr 					= tmpstring + sImgFilename;
							
							mkdir(tmpstring.c_str());
							imwrite(ImgStr,mCurrFrame);
						}
						else
						{
							if((vnSaveDetFameNum[ww])< nDetInterval )
							{
								vnPostDetFrameWritten[ww] = vnPostDetFrameWritten[ww] + 1;
								vnSaveDetFameNum[ww]	  = vnSaveDetFameNum[ww] + 1;
								
								tmpstring 				= sOutputPath + "WindSize_" + convertInt(vnWindSize[ww]) + "/" + "Detection_" + convertInt(vnDetectionCount[ww]) + "_" + convertInt(vnSubDetectionCount[ww]) + "/";
								ImgStr 					= tmpstring + sImgFilename;
								imwrite(ImgStr,mCurrFrame);
							}
							else 
							{
								//create the new folder and save the first image.
								vnSubDetectionCount[ww]     = vnSubDetectionCount[ww] + 1;
								vnSaveDetFameNum[ww] 		= 1;
								vnPostDetFrameWritten[ww] 	= 0;
								
								tmpstring 				= sOutputPath + "WindSize_" + convertInt(vnWindSize[ww]) + "/" + "Detection_" + convertInt(vnDetectionCount[ww]) + "_" + convertInt(vnSubDetectionCount[ww]) + "/";
								ImgStr 					= tmpstring + sImgFilename;
								
								if (!exists(tmpstring))
									mkdir(tmpstring.c_str());
								imwrite(ImgStr,mCurrFrame);
							}
						}
					}
					else
					{
						vnDetectionFlag[ww] = 0;
						if((vnPostDetFlag[ww] > 0)&&(vnPostDetFrameWritten[ww] < nDetInterval))
						{
							// save the images continuous 
							vnPostDetFrameWritten[ww] 	= vnPostDetFrameWritten[ww] + 1;
							vnSaveDetFameNum[ww]  		= vnSaveDetFameNum[ww] + 1;
							
							tmpstring 				= sOutputPath + "WindSize_" + convertInt(vnWindSize[ww]) + "/" + "Detection_" + convertInt(vnDetectionCount[ww]) + "_" + convertInt(vnSubDetectionCount[ww]) + "/";
							ImgStr 					= tmpstring + sImgFilename;
							imwrite(ImgStr,mCurrFrame);
						}
						else
						{
							vnPostDetFlag[ww] 			= 0;
							vnPostDetFrameWritten[ww] 	= 0;							
						}		
					}
					//<<--	
					tmp_score.erase(tmp_score.begin());
				}
				vvdProbScore.at(ww) = tmp_score;
			}
			tmp_score.clear();
		}						
		vmTestFv.erase(vmTestFv.begin());
	}
}