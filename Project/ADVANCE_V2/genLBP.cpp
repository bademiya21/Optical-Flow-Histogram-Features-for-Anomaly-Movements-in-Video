
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "genLBP.h"

using namespace std;

#define PI 3.1415926535897932

#define	BOUND(x, lowerbound, upperbound)  { (x) = (x) > (lowerbound) ? (x) : (lowerbound); \
                                            (x) = (x) < (upperbound) ? (x) : (upperbound); };                                          


uchar bittest(uchar num,uchar bit) //check certain bit
{ 
	if(((num>>bit) & 0x01) == 1) 
		return 1; 
	else 
		return 0; 
} 
uchar bitclr(uchar num,uchar bit)  // clear
{ 
	uchar bit_value[]={1,2,4,8,16,32,64,128}; 
	return num&~bit_value[bit]; 
}

uchar bitset(uchar num,uchar bit) // set a value
{ 
	uchar bit_value[]={1,2,4,8,16,32,64,128}; 
	return num|bit_value[bit]; 
} 

uchar bitcpl(uchar num,uchar bit) //reverse  
{ 
	uchar bit_value[]={1,2,4,8,16,32,64,128}; 
	if(((num>>bit)&0x01)==1) 
		return num&~bit_value[bit]; 
	else 
		return num|bit_value[bit]; 
} 
uchar bitshift(uchar num,char bit) //shift
{
	if(bit>0)// to left
		return num<<abs(bit);
	if(bit<0)// to right
		return num>>abs(bit);
	
	return num;
}

uchar bitxor(uchar num1,uchar num2) //xor
{
	return num1^num2;
}

uchar bitget(uchar num,uchar bit)  //get certain bit value 
{ 
	uchar bit_value[]={1,2,4,8,16,32,64,128}; 
	if(num&bit_value[bit])
		return 1;
	else
		return 0;
} 


void LBPgetmapping(int* mapping, int samples, int mode) //calculate the mapping table
{

	if (mode == 1 ) //'u2'
	{
		int length=1<<samples;	
		int newMax=samples*(samples-1)+3;
		int index=0;
		int i=0,j=0;
		for(i=0;i<length;i++)
		{
			uchar tp1=bitget(i,samples-1);
			if(tp1)
				j=bitset(bitshift(i,1),0);
			else
				j=bitclr(bitshift(i,1),0);
			uchar tp2=bitxor(i,j);
			uchar sum=0;
			for(int k=0;k<samples;k++)
				sum+=bitget(tp2,k);
			uchar numt=sum;
			if(numt<=2)
			{
				mapping[i]=index;
				index+=1;
			}
			else
				mapping[i]=newMax-1;
		}
	}
	else if (mode==3) //'riu2'
	{
		int length=1<<samples;
		int newMax = samples + 2;
		int index=0;
		int i=0,j=0;
		for(i=0;i<length;i++)
		{
			uchar tp1=bitget(i,samples-1);
			if(tp1)
				j=bitset(bitshift(i,1),0);
			else
				j=bitclr(bitshift(i,1),0);
			uchar tp2=bitxor(i,j);
			uchar sum=0;
			for(int k=0;k<samples;k++)
				sum+=bitget(tp2,k);
			uchar numt=sum;

			if (numt <=2)
			{
				sum=0;
				for(int k=0;k<samples;k++)
				{
					sum+=bitget(i,k);
				}
				mapping[i] = sum;
			}else
			{
				mapping[i] = samples+1;
			}
		}
	}	
}



/*********************constructor****************************/
/************************************************************/
LBP::LBP()
{
	R.xR = 0;
	R.yR = 0;
	R.tR = 0;
	
	SN.xy = 0;
	SN.xt = 0;
	SN.yt = 0;
	
	uni = 0;
	interp = 0;
	norm = 0;
	riu2 = 0;
	
	basic_bin.xy = 0;
	basic_bin.xt = 0;
	basic_bin.yt = 0;	
	
	uni_bin.xy = 0;
	uni_bin.xt = 0;
	uni_bin.yt = 0;
	
	basic_hist.pHist_xy = NULL;
	basic_hist.pHist_xt = NULL;
	basic_hist.pHist_yt = NULL;
	
	uni_hist.pHist_xy = NULL;
	uni_hist.pHist_xt = NULL;
	uni_hist.pHist_yt = NULL;	
	
	pHist_ri = NULL;
	ri_bin = 0;
}

/*******************deconstructor****************************/
/************************************************************/
LBP::~LBP()
{
	Release();
}

/****calculate the LBP based histogram of the input image****/
/* pppArray:   the input data of image(s)
/* algorithm:  0 for histogram of LBP
/*             1 for histograms of LBP-TOP
/*             2 for histogram of RIVLBP
/* type:       0(default) for old rotation invariant descriptor published in ECCV workshop 2006
/*             1 for new rotation invariant descriptor published in PAMI 2007
/************************************************************/
void LBP::CreateHistogram(uchar*** pppArray, int algorithm, int type)
{	
	if(!pppArray)
	{
		cout<<"No input data!"<<endl;
		return;
	}

	int bin;
	int x, y, t;
	struct Code res_code ={0,0,0};	
	
	if(algorithm == 0) //LBP based histogram
	{
		//check size
		if(width<=2*R.xR || height<=2*R.yR)
		{
			cout<<"Can not calculate the histogram. Parameters don't match."<<endl;
			return;
		}	
		
		//clear 
		Release();
		
		//allocate memory
		basic_bin.xy = (int)pow(2, (double)SN.xy);
		bin = basic_bin.xy;
		basic_hist.pHist_xy = new float[bin];	memset(basic_hist.pHist_xy, 0, 4*bin);	
		
		//calculate the histogram
		t = 0;
		for(y=R.yR; y<height-R.yR; y++)
		{
			for(x=R.xR; x<width-R.xR; x++)
			{
				res_code = LBPCode(pppArray, algorithm, x, y, t);
				basic_hist.pHist_xy[res_code.xy]++;
			}
		}		
			
		if ((uni==1) || (riu2==1))
		{
			RIU2Histogram();
		}

		if (norm==1)  //normalize the histogram
		{
			Normalization();		
		}
	}

	if(algorithm == -1) // realease memory
		Release();
}	


/********calculate the histogram of RIU2 patterns*********/
/* pphist1:    histogram of uniform patterns
/* phist2:     histogram of all patterns
/* num:        number of sample points
/* r1:         sampling radius of axis1
/* r2:         sampling radius of axis2
/* len1:       size of data along axis1
/* len2:	   size of data along axis2
/* seq_num:    number of axis1-axis2 planes
/* return:     the number of uniform patterns
/************************************************************/
int LBP::CalcRIU2Hist(float** pphist1, float* phist2, int num, int r1, int r2, int len1, int len2, int seq_num)
{	
	int uni_number;
	
	int i, bin;
	
	if(phist2)
	{
		int* mapping = NULL;
		int length=1<<num;
		mapping = new int[length];
		memset((int*)mapping,0,length*sizeof(int));
		
		int lbp_mode = 0; //1:ri; 3: riu2 mode

		if ((uni==1) && (riu2==1))
		{
			lbp_mode = 3;
		}
		else if ((uni==1) && (riu2==0))
		{
			lbp_mode = 1;
		}

		LBPgetmapping(&mapping[0], num, lbp_mode);

		if (lbp_mode==1)
		{
			bin = num*(num-1)+3; 
		}
		else if (lbp_mode ==3)
		{
			bin = num+2; 
		}
				
		uni_number = bin-1;  //consistent with others
		
		//calculate the histogram of uniform patterns		
		*pphist1 = new float[bin];
		
		for(i=0; i<bin; i++)
		{
			(*pphist1)[i] = 0;
		}

		for(i=0; i<length; i++)
		{
			(*pphist1)[mapping[i]] = (*pphist1)[mapping[i]] + phist2[i];
		}				
		
		delete[] mapping;
		mapping = NULL;
	}

	return uni_number;
}

/*******calculate the histograms of RIU2  patterns*********/
/* for LBP
/************************************************************/
void LBP::RIU2Histogram()
{
	uni_bin.xy = CalcRIU2Hist(&uni_hist.pHist_xy, basic_hist.pHist_xy, SN.xy, R.xR, R.yR, width, height, 1);	
}	

/*****************normalize the histograms*******************/
/************************************************************/
void LBP::Normalization()
{
	if((uni==1)||(riu2==1))
	{
		if(uni_hist.pHist_xy)
		{
			CalcNormHist(&uni_hist.pHist_xy, uni_bin.xy+1);
		}
		
		if(uni_hist.pHist_xt)
		{
			CalcNormHist(&uni_hist.pHist_xt, uni_bin.xt+1);
		}
		
		if(uni_hist.pHist_yt)
		{
			CalcNormHist(&uni_hist.pHist_yt, uni_bin.yt+1);
		}				
	}
	else
	{	
		if(basic_hist.pHist_xy)
		{
			CalcNormHist(&basic_hist.pHist_xy, basic_bin.xy);
		}
		
		if(basic_hist.pHist_xt)
		{
			CalcNormHist(&basic_hist.pHist_xt, basic_bin.xt);
		}
		
		if(basic_hist.pHist_yt)
		{
			CalcNormHist(&basic_hist.pHist_yt, basic_bin.yt);
		}
	}
	
	if(pHist_ri)
	{
		CalcNormHist(&pHist_ri, ri_bin);   
	}
}

/*****************normalize the histograms*******************/
/*pphist:  the histogram
/*bin:     bin number of the histogram
/************************************************************/
void LBP::CalcNormHist(float** pphist, int bin)
{
	int i;
	
	float total=0;
	for(i=0; i<bin; i++)
	{
		total += (*pphist)[i];
	}
	
	for(i=0; i<bin; i++)
	{
		(*pphist)[i] /= total;
	}
}

/**********calculate the LBP code of each pixel**************/
/* pppArray:   data of the input image(sequence)
/* type:       0 for LBP
/*             1 for LBP-TOP
/*             2 for RIVLBP
/* x,y,t:      the coordinates of a pixel
/*             (x--width/column;y--height/row;t--time)
/* return:     the decimal value of the pattern
/************************************************************/
struct Code LBP::LBPCode(uchar*** pppArray, int algorithm, int x, int y, int t)
{
	struct Code res_code={0,0,0};
	
	if(!pppArray)
	{
		cout<<"No input data!"<<endl;
		return res_code;
	}
	
	int i;
	int number;
	Point3D32f* pSP = NULL;
	uint* pData = NULL;
	uint** ppData = NULL;

	
	Point3D p;
	Point3D ltp;
	Point3D lbp;
	Point3D rtp;
	Point3D rbp;
	Point3D32f fp;
	Point3D32f new_fp;
	
	//x-y plane
	if(algorithm==0 || algorithm==1)
	{
		number = SN.xy;
		
		pSP = new Point3D32f[number];
		pData = new uint[number];
	
		//calculate the coordinates of the sample points
		for(i=0; i<number; i++)
		{
			pSP[i].x = x+R.xR*cos((2*PI/number)*i);
			pSP[i].y = y-R.yR*sin((2*PI/number)*i);
		}
		
		//calculate the values of the sample points 	
		//without interpolation
		if(interp==0)	
		{
			for(i=0; i<number; i++)
			{
				p.x = (int)(pSP[i].x+0.5);
				p.y = (int)(pSP[i].y+0.5);
				
				BOUND(p.x, 0, width-1);
				BOUND(p.y, 0, height-1);
				
				pData[i] = (uint)(pppArray[t][p.y][p.x]);
			}	
		}
		else	//with bilinear interpolation
		{
			for(i=0; i<number; i++)
			{
				fp.x = pSP[i].x;
				fp.y = pSP[i].y;
			
				//calculate coordinates of four points which are used in bilinear interpolation
				ltp.x = floor(fp.x); ltp.y = floor(fp.y); 
				lbp.x = floor(fp.x); lbp.y = ceil(fp.y);  
				rtp.x = ceil(fp.x);  rtp.y = floor(fp.y); 
				rbp.x = ceil(fp.x);  rbp.y = ceil(fp.y);  
			
				new_fp.x = fp.x-ltp.x;
				new_fp.y = fp.y-ltp.y;
			
				if(new_fp.x<1e-6 && new_fp.y<1e-6)  //interpolation is not needed
				{
					pData[i] = (uint)pppArray[t][(int)fp.y][(int)fp.x];
				}
				else  //bilinear interpolation
				{
					pData[i] = (uint)(pppArray[t][ltp.y][ltp.x]*(1-new_fp.x)*(1-new_fp.y)+
									  pppArray[t][rtp.y][rtp.x]*new_fp.x*(1-new_fp.y)+
								      pppArray[t][lbp.y][lbp.x]*(1-new_fp.x)*new_fp.y+
									  pppArray[t][rbp.y][rbp.x]*new_fp.x*new_fp.y);
				}
			}
		}	
	
		//calculate the LBP code
		res_code.xy = 0;
		for(i=number-1; i>=0; i--)
		{
			res_code.xy = (res_code.xy<<1) + (pData[i] >= (uint)pppArray[t][y][x]);
		}		
	}	

	//calculate LBP code of previous, current and post frames
	int num_fr = 3;
	if(pSP)
	{
		delete []pSP;
		pSP = NULL;
	}
	if(pData)
	{
		delete []pData;
		pData = NULL;
	}
	if(ppData)
	{
		for(i=0; i<num_fr; i++)
		{
			if(ppData[i])
			{
				delete []ppData[i];
				ppData[i] = NULL;
			}
		}
		delete []ppData;
		ppData = NULL;
	}
	
	return res_code;	
}

/*******************release memories*************************/
/************************************************************/
void LBP::Release()
{
	if(basic_hist.pHist_xy)
	{	
		delete []basic_hist.pHist_xy;
		basic_hist.pHist_xy = NULL;
	}
	if(basic_hist.pHist_xt)
	{	
		delete []basic_hist.pHist_xt;
		basic_hist.pHist_xt = NULL;
	}
	if(basic_hist.pHist_yt)
	{	
		delete []basic_hist.pHist_yt;
		basic_hist.pHist_yt = NULL;
	}	
	
	if(uni==1)
	{
		if(uni_hist.pHist_xy)
		{	
			delete []uni_hist.pHist_xy;
			uni_hist.pHist_xy = NULL;
		}
		if(uni_hist.pHist_xt)
		{	
			delete []uni_hist.pHist_xt;
			uni_hist.pHist_xt = NULL;
		}
		if(uni_hist.pHist_yt)
		{	
			delete []uni_hist.pHist_yt;
			uni_hist.pHist_yt = NULL;
		}	
	}	
	
	if(pHist_ri)
	{
		delete []pHist_ri;
		pHist_ri = NULL;
	}			
}
