
#include <opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat *img_01 = new Mat(400, 400, CV_8UC3);
	Mat *img_02 = new Mat(400, 400, CV_8UC3);
	*img_01 = Scalar::all(0);
	*img_02 = Scalar::all(0);
	// ��������ɵ�����
	vector<Point> points_01, points_02;


	// �������鸳ֵ
	points_01.push_back(Point(10, 10)); points_01.push_back(Point(10, 390));
	points_01.push_back(Point(390, 390)); points_01.push_back(Point(150, 250));
	points_02.push_back(Point(10, 10)); points_02.push_back(Point(10, 390));
	points_02.push_back(Point(390, 390)); points_02.push_back(Point(250, 150));

	vector<int> hull_01, hull_02;
	// ����͹��
	convexHull(points_01, hull_01, true);
	convexHull(points_02, hull_02, true);

	// ��������
	for (int i = 0; i < 4; ++i)
	{
		circle(*img_01, points_01[i], 3, Scalar(0, 255, 255), CV_FILLED, CV_AA);
		circle(*img_02, points_02[i], 3, Scalar(0, 255, 255), CV_FILLED, CV_AA);
	}
	// ����͹������
	CvPoint poi_01 = points_01[hull_01[hull_01.size() - 1]];
	for (int i = 0; i < hull_01.size(); ++i)
	{
		line(*img_01, poi_01, points_01[i], Scalar(255, 255, 0), 1, CV_AA);
		poi_01 = points_01[i];
	}
	CvPoint poi_02 = points_02[hull_02[hull_02.size() - 1]];
	for (int i = 0; i < hull_02.size(); ++i)
	{
		line(*img_02, poi_02, points_02[i], Scalar(255, 255, 0), 1, CV_AA);
		poi_02 = points_02[i];
	}

	cout << "img_01 �Ƿ���͹����" << isContourConvex(points_01) << endl;
	cout << "img_02 �Ƿ���͹����" << isContourConvex(points_02) << endl;

	imshow("img_01 ��������͹����", *img_01);
	imshow("img_02 ��������͹����", *img_02);
	cvWaitKey();
	return 0;
}