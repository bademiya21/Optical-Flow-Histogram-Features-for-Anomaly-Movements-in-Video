#include <vector>
#include <map>

using namespace std;

typedef unsigned char uchar;
typedef unsigned int uint;

typedef struct Number
{
	int xy;
	int xt;
	int yt;
}
Number;  //numbers of sample points

typedef struct Radius
{
	int xR;
	int yR;
	int tR;
}
Radius;  //sampling radii

typedef struct Hist
{
	float* pHist_xy;
	float* pHist_xt;
	float* pHist_yt;
}
Hist;   //LBP based histograms

typedef struct Code
{
	uint xy;  //x-y plane; previous frame
	uint xt;  //x-t plane; current frame
	uint yt;  //y-t plane; post frame
}
Code;  //values of LBP of a pixel

typedef struct Point3D
{
	int x;
	int y;
	int t;
}
Point3D;  //3d point

typedef struct Point3D32f
{
	float x;
	float y;
	float t;
}
Point3D32f;  //3d float point

class LBP
{
public:
	LBP();
	~LBP();
	
	void CreateHistogram(uchar*** pppArray, int algorithm, int type=0);
	
private:
	struct Code LBPCode(uchar*** pppArray, int algorithm, int x, int y, int t);
	
	void RIU2Histogram(); 
	int CalcRIU2Hist(float** pphist1, float* phist2, int num, int r1, int r2, int len1, int len2, int seq_num);	
	void Normalization();
	void CalcNormHist(float** pphist, int bin);
	
	void Release();
	
public:    
	int width;     //width of the input image(s)
	int height;    //height of the input image(s)
	int tlength;   //number of the input images

	struct Radius R;	     //the sampling radii
	struct Number SN;        //the number of the sample points
	
	bool uni;        //1 if only using the uniform patterns; 0 if not
	bool interp;     //1 if using bilinear interpolation; 0 if not
	bool norm;       //1 if normalized histogram is calculated; 0 if not
	bool riu2;		 //1 if riu2 histogram is calculated; 0 if not

	struct Number basic_bin;	//number of the basic patterns
	struct Number uni_bin;      //number of uniform patterns
	struct Hist basic_hist;     
	struct Hist uni_hist;
	
	float* pHist_ri;        //(uniform)RIVLBP based histogram
	int ri_bin;             //number of the (uniform)RIVLBP patterns
};

uchar bittest(uchar num,uchar bit);
uchar bitclr(uchar num,uchar bit);
uchar bitset(uchar num,uchar bit);
uchar bitcpl(uchar num,uchar bit);
uchar bitshift(uchar num,char bit);
uchar bitxor(uchar num1,uchar num2);
uchar bitget(uchar num,uchar bit);
void LBPgetmapping(int* mapping, int samples, int mode);

