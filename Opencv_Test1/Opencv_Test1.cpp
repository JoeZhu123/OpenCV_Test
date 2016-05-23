// Opencv_Test1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

Mat src, dst, tmp;
char* window_name = "Pyramids Demo";

int _tmain(int argc, _TCHAR* argv[])
{
	printf("\n Zoom In-Out demo \n");
	//	printf("\n Blur Or-not demo \n");
	printf("-------------------- \n");
	//	printf("*[b]-> Blur the photo \n");
	//	printf("*[a]-> Anti bluring the photo \n");
	printf("*[u]-> Zoom in \n");
	printf("*[d]-> Zoom out \n");
	printf("*[ESC]-> Close program \n\n");

	src = imread("D:\\Me.jpg");
	if (!src.data)
	{
		printf("No data!--Exiting the program \n");
		return -1;
	}

	tmp = src;
	dst = tmp;
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, dst);

	
	while (true)
	{
		int c;
		c = waitKey(10);
		if ((char)c == 27)
		{
			break;
		}
		/*
		if ((char)c == 'b')
		{
			GaussianBlur(tmp,dst,Size(5,5),0,0);
			printf("** Blur the Image\n");
		}
		else if ((char)c == 'a')
		{
			GaussianBlur(tmp, dst, Size(1, 1), 0, 0);
			printf("**Anti bluring the Image\n");
		}
		*/
		if ((char)c == 'u')
		{
			pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
			printf("** Zoom In:Image x 2\n");
		}
		else if ((char)c == 'd')
		{
			pyrDown(tmp, dst, Size(tmp.cols / 2, tmp.rows / 2));
			printf("**Zoom Out:Image / 2\n");
		}
		
		imshow(window_name, dst);
		tmp = dst;
		
	}

	return 0;
}