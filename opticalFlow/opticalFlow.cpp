#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

void tracking(Mat &frame, Mat &output);
bool addNewPoints();
bool acceptTrackedPoint(int i);

string window_name = "optical flow tracking";
Mat gray;	// ��ǰͼƬ
Mat gray_prev;	// Ԥ��ͼƬ
vector<Point2f> points[2];	// point0Ϊ�������ԭ��λ�ã�point1Ϊ���������λ��
vector<Point2f> initial;	// ��ʼ�����ٵ��λ��
vector<Point2f> features;	// ��������
int maxCount = 500;	// �������������
double qLevel = 0.01;	// �������ĵȼ�
double minDist = 10.0;	// ��������֮�����С����
vector<uchar> status;	// ����������״̬��������������Ϊ1������Ϊ0
vector<float> err;

int main()
{
	Mat frame;
	Mat result;

	VideoCapture capture;
	capture.open(0);

	//��������ͷ��ȡ��ͼ���СΪ340x240   
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}

		while(true)
		{
			capture >> frame;

			if(!frame.empty())
			{ 
				tracking(frame, result);
			}
			else
			{ 
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			int c = waitKey(100);
			if( (char)c == 27 )
			{
				break; 
			} 
		}
	
	return 0;
}

//////////////////////////////////////////////////////////////////////////
// function: tracking
// brief: ����
// parameter: frame	�������Ƶ֡
//			  output �и��ٽ������Ƶ֡
// return: void
//////////////////////////////////////////////////////////////////////////
void tracking(Mat &frame, Mat &output)
{
	cvtColor(frame, gray, CV_BGR2GRAY);
	frame.copyTo(output);
	// ����������
	if (addNewPoints())
	{
		goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
		points[0].insert(points[0].end(), features.begin(), features.end());
		initial.insert(initial.end(), features.begin(), features.end());
	}

	if (gray_prev.empty())
	{
		gray.copyTo(gray_prev);
	}
	// l-k�������˶�����
	calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);
	// ȥ��һЩ���õ�������
	int k = 0;
	for (size_t i=0; i<points[1].size(); i++)
	{
		if (acceptTrackedPoint(i))
		{
			initial[k] = initial[i];
			points[1][k++] = points[1][i];
		}
	}
	points[1].resize(k);
	initial.resize(k);
	// ��ʾ��������˶��켣
	for (size_t i=0; i<points[1].size(); i++)
	{
		line(output, initial[i], points[1][i], Scalar(0, 0, 255));
		circle(output, points[1][i], 3, Scalar(255, 0, 0), -1);
	}

	// �ѵ�ǰ���ٽ����Ϊ��һ�˲ο�
	swap(points[1], points[0]);
	swap(gray_prev, gray);

	imshow(window_name, output);
}

//////////////////////////////////////////////////////////////////////////
// function: addNewPoints
// brief: ����µ��Ƿ�Ӧ�ñ�����
// parameter:
// return: �Ƿ����ӱ�־
//////////////////////////////////////////////////////////////////////////
bool addNewPoints()
{
	return points[0].size() <= 10;
}

//////////////////////////////////////////////////////////////////////////
// function: acceptTrackedPoint
// brief: ������Щ���ٵ㱻����
// parameter:
// return:
//////////////////////////////////////////////////////////////////////////
bool acceptTrackedPoint(int i)
{
	return status[i] && ((abs(points[0][i].x - points[1][i].x) + abs(points[0][i].y - points[1][i].y)) > 2);
}