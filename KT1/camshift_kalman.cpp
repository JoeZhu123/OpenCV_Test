#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string> 
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <cv.h>
#include <cxcore.h>
#include <cvaux.h>
#include <highgui.h>
#include <stdlib.h>
#include <time.h>
//*************************************
//调整矩形框B，使其在A的范围
CvRect my_ChangeRect(CvRect A, CvRect B)
{
	if (B.x<A.x)
		B.x = A.x;
	if (B.x + B.width>A.width)
		B.width = A.width - B.x;
	if (B.y<A.y)
		B.y = A.y;
	if (B.y + B.height>A.height)
		B.height = A.height - B.y;
	return(B);
}
//**************************************
//kalman初始化
CvKalman* InitializeKalman(CvKalman* kalman)
{

	const float A[] = {
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1 };
	kalman = cvCreateKalman(4, 2, 0);
	memcpy(kalman->transition_matrix->data.fl, A, sizeof(A));//A
	cvSetIdentity(kalman->measurement_matrix, cvScalarAll(1));//H
	cvSetIdentity(kalman->process_noise_cov, cvScalarAll(1e-1));//Q w ；
	cvSetIdentity(kalman->measurement_noise_cov, cvScalarAll(1e-5));//R v
	cvSetIdentity(kalman->error_cov_post, cvScalarAll(1));//P
	return kalman;
}
//***************************************
//point1上一帧中心，point2当前帧中心，把x,y,vx,vy存入kalman->state_post
void GetCurentState(CvKalman* kalman, CvPoint point1, CvPoint point2)
{
	float input[4] = { point2.x, point2.y, point2.x - point1.x, point2.y - point1.y };//currentstate
	memcpy(kalman->state_post->data.fl, input, sizeof(input));
}
//****************************************
//point1上一帧中心，point2当前帧中心，把x,y,vx,vy存入矩阵
CvMat* GetMeasurement(CvMat* mat, CvPoint point1, CvPoint point2)
{
	float input[4] = { point2.x, point2.y, point2.x - point1.x, point2.y - point1.y };
	memcpy(mat->data.fl, input, sizeof(input));
	return mat;
}
/*****************************************/
#define region 50
#define calc_point(kalman) cvPoint( cvRound(kalman[0]),cvRound(kalman[1]))

IplImage *image = 0, *hsv = 0, *hue = 0, *mask = 0, *histimg = 0;
CvHistogram *hist;
IplImage *backproject;

CvPoint lastpoint; //上一帧的中心坐标
CvPoint predictpoint, measurepoint;//预测坐标以及当前帧的坐标

int select_object = 0;
int track_object;

CvPoint origin;
CvPoint point_text;
CvRect selection;
CvRect origin_box;

CvRect track_window;
CvBox2D track_box; // tracking 返回的区域 box，带角度
CvConnectedComp track_comp;
int hdims = 256; // 划分HIST的个数，越高越精确
float hranges_arr[] = { 0, 180 };
float* hranges = hranges_arr;
int vmin = 10, vmax = 256, smin = 30;
int start_track = 0;
CvFont font;//显示的文本字体
char nchar[10];//显示数字字符串用
CvMat* measurement;
CvMat* realposition;
const CvMat* prediction;
CvRect search_window;
int filename1_n = 0;
int frame_count = 0;

void on_mouse(int event, int x, int y, int flags, void *p)
{
	if (!image)
		return;

	if (image->origin)
		y = image->height - y;

	if (select_object)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = selection.x + CV_IABS(x - origin.x);
		selection.height = selection.y + CV_IABS(y - origin.y);

		selection.x = MAX(selection.x, 0);
		selection.y = MAX(selection.y, 0);
		selection.width = MIN(selection.width, image->width);
		selection.height = MIN(selection.height, image->height);
		selection.width -= selection.x;
		selection.height -= selection.y;

	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = cvPoint(x, y);
		selection = cvRect(x, y, 0, 0);
		select_object = 1;
		break;
	case CV_EVENT_LBUTTONUP:
		select_object = 0;
		if (selection.width > 0 && selection.height > 0)
			track_object = -1;
		origin_box = selection;

#ifdef _DEBUG
		printf("鼠标的选择区域：");
		printf("X = %d, Y = %d, Width = %d, Height = %d\n",
			selection.x, selection.y, selection.width, selection.height);
		printf("");
#endif
		break;
	}
}
/*************************************main***************************/
int main(int argc, char** argv)
{
	CvCapture* capture = 0;
	IplImage* frame = 0;
	if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
		//打开摄像头
		capture = cvCaptureFromCAM(argc == 2 ? argv[1][0] - '0' : 0);
	cvNamedWindow("CamShiftDemo", 1);
	cvSetMouseCallback("CamShiftDemo", on_mouse, NULL); // on_mouse 自定义事件
	cvCreateTrackbar("Vmin", "CamShiftDemo", &vmin, 256, 0);
	cvCreateTrackbar("Vmax", "CamShiftDemo", &vmax, 256, 0);
	cvCreateTrackbar("Smin", "CamShiftDemo", &smin, 256, 0);
	//初始化kalman
	CvKalman* kalman = 0;
	kalman = InitializeKalman(kalman);
	measurement = cvCreateMat(2, 1, CV_32FC1);//Z(k)
	realposition = cvCreateMat(4, 1, CV_32FC1);//real X(k)
	for (;;)
	{
		int i = 0, c;
		frame = cvQueryFrame(capture);
		if (!frame)
			break;
		if (!image)
		{
			/* allocate all the buffers */
			image = cvCreateImage(cvGetSize(frame), 8, 3);
			image->origin = frame->origin;
			hsv = cvCreateImage(cvGetSize(frame), 8, 3);
			hue = cvCreateImage(cvGetSize(frame), 8, 1);
			mask = cvCreateImage(cvGetSize(frame), 8, 1);

			backproject = cvCreateImage(cvGetSize(frame), 8, 1);
			backproject->origin = frame->origin;
			hist = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1); // 计算直方图
		}
		cvCopy(frame, image, 0);
		cvCvtColor(image, hsv, CV_BGR2HSV); // 彩色空间转换 BGR to HSV 
		frame_count++;
		//*******************************************************************************************
		if (track_object)
		{
			int _vmin = vmin, _vmax = vmax;
			cvInRangeS(hsv, cvScalar(0, smin, MIN(_vmin, _vmax), 0),
				cvScalar(180, 256, MAX(_vmin, _vmax), 0), mask); // 得到二值的MASK
			cvSplit(hsv, hue, 0, 0, 0); // 只提取 HUE 分量 
			if (track_object < 0)
			{
				float max_val = 0.f;
				cvSetImageROI(hue, origin_box); // 得到选择区域 for ROI
				cvSetImageROI(mask, origin_box); // 得到选择区域 for mask
				cvCalcHist(&hue, hist, 0, mask); // 计算直方图
				cvGetMinMaxHistValue(hist, 0, &max_val, 0, 0); // 只找最大值
				cvConvertScale(hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0); // 缩放 bin 到区间 [0,255] 
				cvResetImageROI(hue); // remove ROI
				cvResetImageROI(mask);
				track_window = origin_box;
				track_object = 1;
				lastpoint = predictpoint = cvPoint(track_window.x + track_window.width / 2,
					track_window.y + track_window.height / 2);
				GetCurentState(kalman, lastpoint, predictpoint);//input curent state
			}
			//预测目标框的位置(x,y,vx,vy), 
			prediction = cvKalmanPredict(kalman, 0);//predicton=kalman->state_post 
			predictpoint = calc_point(prediction->data.fl);
			//预测出的矩形框
			track_window = cvRect(predictpoint.x - track_window.width / 2, predictpoint.y - track_window.height / 2,
				track_window.width, track_window.height);
			track_window = my_ChangeRect(cvRect(0, 0, frame->width, frame->height), track_window);
			//只对目标周围计算投影
			search_window = cvRect(track_window.x - region, track_window.y - region,
				track_window.width + 2 * region, track_window.height + 2 * region);
			//修正search_window，使其范围在图像内
			search_window = my_ChangeRect(cvRect(0, 0, frame->width, frame->height), search_window);
			cvSetImageROI(hue, search_window);
			cvSetImageROI(mask, search_window);
			cvSetImageROI(backproject, search_window);
			cvCalcBackProject(&hue, backproject, hist); // 使用 back project 方法
			cvAnd(backproject, mask, backproject, 0);
			//因为设置了_1ROI，所以更新track_window
			track_window = cvRect(region, region, track_window.width, track_window.height);
			// calling CAMSHIFT 算法模块 
			cvCamShift(backproject, track_window,
				cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1),
				&track_comp, &track_box);
			/*cvMeanShift(backproject, track_window,
			cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1),
			&track_comp);*/
			//******************************************************************************** 
			cvResetImageROI(hue);
			cvResetImageROI(mask);
			cvResetImageROI(backproject);
			track_window = track_comp.rect;
			track_window = cvRect(track_window.x + search_window.x, track_window.y + search_window.y,
				track_window.width, track_window.height);
			//实际的质心坐标 
			measurepoint = cvPoint(track_window.x + track_window.width / 2,
				track_window.y + track_window.height / 2);
			realposition->data.fl[0] = measurepoint.x;
			realposition->data.fl[1] = measurepoint.y;
			realposition->data.fl[2] = measurepoint.x - lastpoint.x;
			realposition->data.fl[3] = measurepoint.y - lastpoint.y;
			lastpoint = measurepoint;
			//keep the current real position
			//实际算出的measurement只是当前的x，y
			cvMatMulAdd(kalman->measurement_matrix/*2x4*/, realposition/*4x1*/,/*measurementstate*/ 0, measurement);
			cvKalmanCorrect(kalman, measurement);
			cvRectangle(image, cvPoint(track_window.x, track_window.y),
				cvPoint(track_window.x + track_window.width, track_window.y + track_window.height),
				CV_RGB(255, 0, 0), 2, 8, 0);
			//把目标是第几个标出来
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0F, 1.0F, 0, 3, 8);
			sprintf(nchar, "tag:%d", i + 1);
			point_text = cvPoint(track_window.x, track_window.y + track_window.height);
			cvPutText(image, nchar, point_text, &font, CV_RGB(255, 255, 0));
		}
		//*********************************************************************************************** 
		if (select_object && selection.width > 0 && selection.height > 0)
		{
			cvSetImageROI(image, selection);
			cvXorS(image, cvScalarAll(255), image, 0);
			cvResetImageROI(image);
		}
		cvShowImage("CamShiftDemo", image);
		c = cvWaitKey(30);
		//根据键盘关闭跟踪目标
		if (c == 27)
			break; // exit from for-loop 
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("CamShiftDemo");
	return 0;
}