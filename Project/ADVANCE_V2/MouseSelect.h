#include<opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

void mouseHandler(int event, int x, int y, int flags, void* param);
Rect SelectRoi(Mat img,int select_flag);