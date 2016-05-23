/*
*	2015.1.5
*
*	�������ߣ����Ҷ�
*
*/
#include <drive_out_smear.h>
#include<iostream>
#include<stdio.h>

typedef unsigned char uchar;


using namespace std;
using namespace cv;


//������ֱ�������Ӱ
void smear_off(cv::Mat& img_yes,cv::Mat& img_no)
{
	
	float t_exp = 1;//ͼ���ع�ʱ�䣺����������Զ� ��λ��ms
	float t_sh = 0.5;//ͼ��֡ת��ʱ�䣺����������Զ� ��λ��ms
	int offset = 1;//ͼ��ƫ�ûҶ�
	float k = 1;//ƫ��ϵ��
	int Rows = img_yes.rows;
	int Cols = img_yes.cols;

	vector<uchar> temp;//������ʱ�Ҷ�ֵ
	//Mat image_t;

	printf("\n Rows = %d \t",Rows);
	printf("\t Cols = %d \n", Cols);
	
	temp[1] = offset*k;//��tempֵ��ʼ��
	//img_yes = image_t;//����ԭʼͼƬ


	for (int j = 1; j <= Rows;j++)
	{
		k = t_exp / (t_sh *1);//����ƫ��ϵ��
		for (int i = 1; i <= Cols; i++)
		{
			if (j<=1)
				img_no.at<uchar>(j, i) = img_yes.at<uchar>(j, i) - temp[j];
			else
				img_no.at<uchar>(j, i) = img_yes.at<uchar>(j,i) - temp[j]/k;   //ȡ�����ػ��߸�ֵ
			temp[j + 1] = temp[j] + img_no.at<uchar>(j, i);
		}
	}	
	//img_yes = image_t;//��ԭԭʼͼƬ
	

}

//�������ⷽ�����Ӱ

//��ͼ������񻯴���
void ncu::sharpen(cv::Mat& img_in, Mat& img_out)
{
	img_out.create(img_in.size(), img_in.type());

	//����߽��ڲ������ص㣬ͼ������Χ�����ص�Ӧ�ö��⴦��
	for (int row = 1; row < img_in.rows - 1; row++)
	{
		//ǰһ�����ص�
		const uchar* previous = img_in.ptr<const uchar>(row - 1);
		//������ĵ�ǰ��
		const uchar* current = img_in.ptr<const uchar>(row);
		//��һ��
		const uchar* next = img_in.ptr<const uchar>(row + 1);

		uchar *output = img_out.ptr<uchar>(row + 1);
		int ch = img_in.channels();
		int starts = ch;
		int ends = (img_in.cols - 1)*ch;
		for (int col = starts; col < ends; col++)
		{
			//���ͼ��ı���ָ���뵱ǰ�е�ָ��ͬ����������ÿ�е�ÿһ�����ص��ÿһ��ͨ��ֵΪһ����������
			//��ΪҪ���ǵ�ͼ���ͨ����
			*output++ = saturate_cast<uchar>(5 * current[col] - current[col - ch] - current[col + ch] - previous[col] - next[col]);
		}
	}//����ѭ��
	//����߽磬��Χ���ص���Ϊ 0 
	img_out.row(0).setTo(Scalar::all(0));
	img_out.row(img_out.rows - 1).setTo(Scalar::all(0));
	img_out.col(0).setTo(Scalar::all(0));
	img_out.col(img_out.cols - 1).setTo(Scalar::all(0));

}

 
















