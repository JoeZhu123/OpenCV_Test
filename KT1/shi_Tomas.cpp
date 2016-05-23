#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include <vector>

using namespace std;
using namespace cv;

Mat src, src_gray;

int maxCorners = 1000;
int maxTrackbar = 100;

RNG rng(12345);

char* source_window = "Image";

void goodFeaturesToTrack_demo(int, void*);

int main()
{
	VideoCapture capture;
	capture.open(0);

	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	createTrackbar("角点个数：", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_demo);
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}
	while (true)
	{
		capture >> src;
		cvtColor(src, src_gray, CV_BGR2GRAY);
		goodFeaturesToTrack_demo(0, 0);
		if (!src.empty())
		{
			imshow(source_window, src);
		}
		
		if (cvWaitKey(12) == 'q')
			break;

	}	
	
	
	return 0;
}

void goodFeaturesToTrack_demo(int, void*)
{
	if (maxCorners<1)
	{
		maxCorners = 1;
	}

	//Shi-Tomasi 角点算法参数定义
	vector<Point2f> corners;
	double qualityLevel = 0.01;//最大最小特征值乘法因子
	double minDistance = 10;//角点之间最小距离
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;


	Mat copy;
	copy = src.clone();

	goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

	cout << "检测到角点数:" << corners.size() << endl;
	int r = 4;
	for (int i = 0; i<corners.size(); i++)
	{
		circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, 8, 0);
	}
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	imshow(source_window, copy);
}