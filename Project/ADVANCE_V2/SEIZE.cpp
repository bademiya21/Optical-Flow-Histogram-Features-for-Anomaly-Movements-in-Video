// SEIZE.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include <opencv2/highgui/highgui.hpp>
#include "ActivityDetection.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <windows.h>
#include "MouseSelect.h"

using namespace std;
using namespace boost::filesystem;
using namespace cv;

ActivityDetection    ActivityDet;

//string TestPath		= "F:/FAMS_Capture/00/2010/11/09/20101109_18to23_P20/";
//string OutputPath		= "F:/FAMS_Capture/00/2010/11/09/20101109_18to23_P20-output/";
int initflag = 1;
Rect Sel_rect;
Mat PrevFrame;


void ProcessFiles(const path& basepath, const string& outpath,int nOpt) 
{
	// Recursive function that process files in sub directories and runs activity detection on it
	for (directory_iterator iter = directory_iterator(basepath); iter
			!= directory_iterator(); iter++)
	{
		directory_entry entry = *iter;
		if (is_directory(entry.path())) 
		{
			cout << "Processing directory " << entry.path().string() << endl;
			ProcessFiles(entry.path(),outpath, nOpt);
		} else 
		{
			path entryPath = entry.path();
			if (entryPath.extension() == ".jpg" || entryPath.extension() == ".jpeg") 
			{
				// process the images:
				cout << " Processing image: " << entryPath.string() << endl;
				Mat CurrFrame = imread(entryPath.string());
				if (initflag == 1)
				{
					// Read first image and then select ROI					
					Sel_rect = SelectRoi(CurrFrame,0);
					ActivityDet.InitThresMode(nOpt);
					initflag = 0;
				}
				ActivityDet.startActDetection(CurrFrame,Sel_rect,outpath,entryPath.string());					
			}
		}
	}
}

int main(int argc, char ** argv) {
	// This is an 'off-line' version of SEIZE's algorithm, does not require GUI
	// Allows for the tests to be done on 64-bit machines.
	
	string TestPath, OutputPath;
	cout << "Test Path: \n";
	getline (cin, TestPath);
	cout << "\nOutput Path: \n";
	getline (cin, OutputPath);

	int nThresOpt;
	cout << "\n1 - Adaptive, 2 - Fixed: ";
	cin >> nThresOpt;
	
	if (!exists(OutputPath)){
		mkdir(OutputPath.c_str());		
	} else {
		remove(OutputPath.c_str());
		mkdir(OutputPath.c_str());
	}

	ProcessFiles(path(TestPath),OutputPath, nThresOpt);
}