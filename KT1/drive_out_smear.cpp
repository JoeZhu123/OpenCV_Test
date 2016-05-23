/*
*	2015.1.5
*
*	程序作者：朱幸尔
*
*/
#include <drive_out_smear.h>
#include<iostream>
#include<stdio.h>

typedef unsigned char uchar;


using namespace std;
using namespace cv;


//处理竖直方向的拖影
void smear_off(cv::Mat& img_yes,cv::Mat& img_no)
{
	
	float t_exp = 1;//图像曝光时间：依据相机特性定 单位：ms
	float t_sh = 0.5;//图像帧转移时间：依据相机特性定 单位：ms
	int offset = 1;//图像偏置灰度
	float k = 1;//偏置系数
	int Rows = img_yes.rows;
	int Cols = img_yes.cols;

	vector<uchar> temp;//过程临时灰度值
	//Mat image_t;

	printf("\n Rows = %d \t",Rows);
	printf("\t Cols = %d \n", Cols);
	
	temp[1] = offset*k;//给temp值初始化
	//img_yes = image_t;//保存原始图片


	for (int j = 1; j <= Rows;j++)
	{
		k = t_exp / (t_sh *1);//计算偏置系数
		for (int i = 1; i <= Cols; i++)
		{
			if (j<=1)
				img_no.at<uchar>(j, i) = img_yes.at<uchar>(j, i) - temp[j];
			else
				img_no.at<uchar>(j, i) = img_yes.at<uchar>(j,i) - temp[j]/k;   //取得像素或者赋值
			temp[j + 1] = temp[j] + img_no.at<uchar>(j, i);
		}
	}	
	//img_yes = image_t;//还原原始图片
	

}

//处理任意方向的拖影

//对图像进行锐化处理
void ncu::sharpen(cv::Mat& img_in, Mat& img_out)
{
	img_out.create(img_in.size(), img_in.type());

	//处理边界内部的像素点，图像最外围的像素点应该额外处理
	for (int row = 1; row < img_in.rows - 1; row++)
	{
		//前一行像素点
		const uchar* previous = img_in.ptr<const uchar>(row - 1);
		//待处理的当前行
		const uchar* current = img_in.ptr<const uchar>(row);
		//下一行
		const uchar* next = img_in.ptr<const uchar>(row + 1);

		uchar *output = img_out.ptr<uchar>(row + 1);
		int ch = img_in.channels();
		int starts = ch;
		int ends = (img_in.cols - 1)*ch;
		for (int col = starts; col < ends; col++)
		{
			//输出图像的遍历指针与当前行的指针同步递增，以每行的每一个像素点的每一个通道值为一个递增量，
			//因为要考虑到图像的通道数
			*output++ = saturate_cast<uchar>(5 * current[col] - current[col - ch] - current[col + ch] - previous[col] - next[col]);
		}
	}//结束循环
	//处理边界，外围像素点设为 0 
	img_out.row(0).setTo(Scalar::all(0));
	img_out.row(img_out.rows - 1).setTo(Scalar::all(0));
	img_out.col(0).setTo(Scalar::all(0));
	img_out.col(img_out.cols - 1).setTo(Scalar::all(0));

}

 
















