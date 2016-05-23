/*
*	2014.12.24
*
*	�����޸��ߣ����Ҷ�
*	
*/
#include <opencv2/opencv.hpp>
#include <string> 
#include <tld_utils.h>
#include <iostream>
#include <sstream>
#include <TLD.h>
#include <stdio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <vector>

using namespace cv;
using namespace std;
RNG rng(12345);
//#define BACKGROUND

//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;
//ȫ��ģ������  
vector<vector<Point>> g_TemplateContours;
//ģ�����  
int g_badmintonTNum = 6;

/*
��ȡ��¼bounding box���ļ������bounding box���ĸ����������Ͻ�����x��y�Ϳ��
����\datasets\06_car\init.txt�У���¼�˳�ʼĿ���bounding box����������
142,125,232,164
*/
void readBB(char* file)
{
  ifstream bb_file (file);//�����뷽ʽ���ļ�
  string line;
  //istream& getline ( istream& , string& );  
  //��������is�ж������ַ�����str�У��ս��Ĭ��Ϊ '\n'�����з���
  getline(bb_file,line);
  istringstream linestream(line);//istringstream������԰�һ���ַ�����Ȼ���Կո�Ϊ�ָ����Ѹ��зָ������� 
  string x1,y1,x2,y2;
  //istream& getline ( istream &is , string &str , char delim );   
  //��������is�ж������ַ�����str�У�ֱ�������ս��delim�Ž�����
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');

  //atoi �� �ܣ� ���ַ���ת����������
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback 
//������Ӧ���ǵõ�Ŀ������ķ�Χ�������ѡ��bounding box��
void mouseHandler(int event, int x, int y, int flags, void *param)
{
  switch( event )
  {
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box)
	{
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 )
	{
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 )
	{
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;//�Ѿ����bounding box  
    break;
  }
}

void print_help(char** argv)
{
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}
//�������г���ʱ�������в���
void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs)
{
  for (int i=0;i<argc;i++)
  {
      if (strcmp(argv[i],"-b")==0)
	  {
          if (argc>i)
		  {
              readBB(argv[i+1]);//�Ƿ�ָ����ʼ��bounding box  
              gotBB = true;
          }
          else
            print_help(argv);
      }
	  //Similar in format to XML, Yahoo! Markup Language (YML) provides functionality to Open   
	  //Applications in a safe and standardized fashion. You include YML tags in the HTML code  
	  //of an Open Application.
      if (strcmp(argv[i],"-p")==0)//��ȡ�����ļ�parameters.yml  
	  {
		  if (argc>i) //FileStorage��Ķ�ȡ��ʽ�����ǣ�FileStorage fs(".\\parameters.yml", FileStorage::READ);
		  {
              fs.open(argv[i+1], FileStorage::READ);
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-no_tl")==0)//To train only in the first frame (no tracking, no learning) 
	  {
          tl = false;
      }
      if (strcmp(argv[i],"-r")==0)
	  {
		  //Repeat the video, first time learns, second time detects  
          rep = true;
      }
  }
}

/*
���г���ʱ��
%To run from camera
./run_tld -p ../parameters.yml
%To run from file
./run_tld -p ../parameters.yml
%To init bounding box from file
./run_tld -p ../parameters.yml -b ../datasets/06_car/init.txt
%To train only in the first frame (no tracking, no learning)
./run_tld -p ../parameters.yml -b ../datasets/06_car/init.txt -no_tl
%To test the final detector (Repeat the video, first time learns, second time detects)
./run_tld -p ../parameters.yml -b ../datasets/06_car/init.txt -r
*/
//�о����Ƕ���ʼ֡���г�ʼ��������Ȼ����֡����ͼƬ���У������㷨����
/*
%YAML:1.0
	Parameters:
		min_win: 15
		patch_size: 15
		ncc_thesame: 0.95
		valid: 0.5
		num_trees: 10
		num_features: 13
		thr_fern: 0.6
		thr_nn: 0.65
		thr_nn_valid: 0.7
		num_closest_init: 10
		num_warps_init: 20
		noise_init: 5
		angle_init: 20
		shift_init: 0.02
		scale_init: 0.02
		num_closest_update: 10
		num_warps_update: 10
		noise_update: 5
		angle_update: 10
		shift_update: 0.02
		scale_update: 0.02
		overlap: 0.2
		num_patches: 100
		bb_x: 288
		bb_y: 36
		bb_w: 25
		bb_h: 42
*/

#ifndef BACKGROUND
/*opencv������ģ���˶�������   ��Ƶ���б�����ģ��Ȼ���ֵó��˶�����*/
int main(int argc, char** argv)
{
	//VideoCapture video("Test1_6.MPG");//��Ƶ����
	
	VideoCapture capture;//����ͷ��׽
	capture.open(0);//������ͷ

//	BackgroundSubtractorMOG bgSubtractor(20, 10, 0.5 , false);
	BackgroundSubtractorMOG2 bgSubtractor2(20,16,false);
	
	int nFrmNum = 0;
	bool stop = false;
	Mat frame;
	Mat fore_frame;
	Mat back_frame;	
	Mat closerect = getStructuringElement(MORPH_RECT, Size(3, 3)); //���нṹ��������
	
//	Mat back_ground;

	cvNamedWindow("NCU-video", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("NCU-backframe", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("NCU-foreframe", CV_WINDOW_AUTOSIZE);
//	cvNamedWindow("NCU-background", CV_WINDOW_AUTOSIZE);

	//���д���
//	cvMoveWindow("NCU-video",30,0);
//	cvMoveWindow("NCU-backframe",300,0);
//	cvMoveWindow("NCU-foreframe",600,0);
//	cvMoveWindow("NCU-foreframe", 900, 0);

	//��������ͷ��ȡ��ͼ���СΪ340x240   
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}
	
	//��֡��ȡ��Ƶ
	while (!stop)
	{
			capture >> frame;
			//Output file
			
			
			++nFrmNum;
//			cout << nFrmNum << endl;

			//�˶�ǰ����⣬�����±���
//			bgSubtractor(frame,back_frame,0.01);
			bgSubtractor2(frame,back_frame,0.01);

/******************opencv1.x�汾�ı�����ַ�******************************/
//�ȸ�˹�˲�����ƽ��ͼ��
//GaussianBlur(fore_frame,frame,frame.size(),0);			
//��ǰ֡������ͼ���
//absdiff(frame,back_frame,fore_frame);
//��ֵ��ǰ��ͼ
//threshold(fore_frame,fore_frame, 60, 255.0, CV_THRESH_BINARY);
//���±���
//cvRunningAvg(frame,back_frame, 0.003, 0);
/************************************************************************/
			//������̬ѧ�˲���ȥ������
			erode(back_frame, fore_frame, Mat());
			dilate(fore_frame, fore_frame, Mat());
			morphologyEx(fore_frame, fore_frame,MORPH_OPEN,closerect);//������̬ѧ������
			/**********************************����ƥ��*******************************************************/
			/*
			vector<vector<Point>> w, w1;
			vector<Vec4i> hierarchy, hierarchy1;
			findContours(fore_frame, w, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//��ȡ����Ԫ��
			//findContours(f1, w1, hierarchy1, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
			FileStorage fs("f.dat", FileStorage::WRITE);
			fs << "f" << w1[0];
			int idx = 0;
			double ffff = matchShapes(w[0], w1[0], CV_CONTOURS_MATCH_I3, 1.0);//��������ƥ��
			std::cout << ffff << std::endl;
			*/
			/***********************************************************************************************/
//			bgSubtractor2.getBackgroundImage(back_ground);				
			Mat drawing = Mat::zeros(fore_frame.size(), CV_8UC3);
			int largestComp = 0;
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			//findContours(fore_frame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			findContours(fore_frame, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
			vector<RotatedRect> minRect(contours.size());
			/// �����
			vector<Moments> mu(contours.size());
			///  �������ľ�:
			vector<Point2f> mc(contours.size());
			//iterate through all the top-level contours
			//draw each connected component with its own random color		
			for (int idx = 0; idx < contours.size(); idx++)
			{
				//��ȥ̫������̫�̵�����(ע��ֻ��һ���ǳ��򵥵��˳�����)  
				int cmin = 0;
				int cmax = 100;
				double whRatio_max = 10.0;//3//2.0//3.0
				double whRatio_min = 0.5;//1.5

				double dAreaMax = 280.0;//280
				double dAreaMin = 5.0;//50	

				double dConArea = fabs(contourArea(contours[idx]));
				double cLength = arcLength(contours[idx], true);
				Rect aRect = boundingRect(contours[idx]);

				if ((cLength<cmax) && (cLength>cmin))
				{
					if (((aRect.width / aRect.height) < whRatio_max) && ((aRect.width / aRect.height) > whRatio_min) && (dConArea < dAreaMax) && (dConArea > dAreaMin))
					{
						Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

						minRect[idx] = minAreaRect(Mat(contours[idx]));
						float angle = minRect[idx].angle;
						printf("angle: %f \n", angle);
						if (true)
						{
							mu[idx] = moments(contours[idx], false);// �����
							mc[idx] = Point2f(mu[idx].m10 / mu[idx].m00, mu[idx].m01 / mu[idx].m00);//  �������ľ�

							drawContours(drawing, contours, idx, color, 2, 8, vector<Vec4i>(), 0, Point());
							circle(drawing, mc[idx], 10, color, 1, 8, 0);

							Point2f rect_points[4];
							minRect[idx].points(rect_points);
							for (int j = 0; j < 4; j++)
							{
								line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
							}

							largestComp++;
						}

						
					}
				}	
			}
			//printf("contous.size: %d \t contous_first.size: %d\n", contours.size(), largestComp);

			//show Image
			if (!frame.empty())
			{
				imshow("NCU-video", frame);
				imshow("NCU-backframe",back_frame);
				imshow("NCU-foreframe", fore_frame);
				imshow("NCU-Drawing", drawing);
			}

//			if (!back_ground.empty())
//				imshow("NCU-background", back_ground);
			//fclose(bb_file);
			if (cvWaitKey(12) == 'q')
				break;
	
	}
	//���ٴ���
	cvDestroyWindow("NCU-video");
//	cvDestroyWindow("NCU-background");
	cvDestroyWindow("NCU-foreframe");
	cvDestroyWindow("NCU-backframe");
	capture.release();
	/*
	Mat image;
	image = imread("D:\VSProjects\Data_T\Data_T\1.png");
	namedWindow("t_test");
	imshow("dt_test", image);
	cvtColor(image,image,CV_BGR2GRAY);
	threshold(image, image, 60, 255.0, CV_THRESH_BINARY);
	vector<vector<Point>> contours;
	findContours(image,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
	Mat result(image.size(),CV_8U,Scalar(0));
	drawContours(result, contours, -1, Scalar(255), 2);
	namedWindow("contours_test");
	imshow("contours_test",result);
	waitKey();
	*/
	
	return 0;
}
#else

int main(int argc, char * argv[])
{
  VideoCapture capture;//����ͷ��׽
  capture.open(0);//������ͷ 
  //OpenCV��C++�ӿ��У����ڱ���ͼ���imwriteֻ�ܱ����������ݣ�������Ϊͼ���ʽ������Ҫ���渡  
  //�����ݻ�XML/YML�ļ�ʱ��OpenCV��C���Խӿ��ṩ��cvSave����������һ������C++�ӿ����Ѿ���ɾ����  
  //ȡ����֮����FileStorage�ࡣ
  FileStorage fs;
  //Read options
  read_options(argc, argv, capture, fs); //���������в��� 
  //�����޸ĺ󣬲���Read options��ֱ�Ӿ��ǵ�������ͷ�����������в������룺KT1 -p D:\VSProjects\KT1\parameters.yml
   //argv[2]����parameters.yml���ļ�λ���ַ�����D:\VSProjects\KT1\parameters.yml
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  cvNamedWindow("�ϲ���ѧ�����˶�",CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback( "�ϲ���ѧ�����˶�", mouseHandler, NULL );//�����ѡ�г�ʼĿ���bounding box  
  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  
  //��������ͷ��ȡ��ͼ���СΪ340x240   
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  

  ///Initialization
GETBOUNDINGBOX://��־λ����ȡbounding box  
  while(!gotBB)
  {
	  capture >> frame;
    cvtColor(frame, last_gray, CV_RGB2GRAY);
	drawBox(frame, box,Scalar(0, 0, 255), 2);//��bounding box ������  
    imshow("�ϲ���ѧ�����˶�", frame);
    if (cvWaitKey(12) == 'q')
	    return 0;
  }
  //����ͼ��Ƭ��min_win Ϊ15x15���أ�����bounding box�в����õ��ģ�����box�����min_winҪ��
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"])
  {
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  //Remove callback
  cvSetMouseCallback( "�ϲ���ѧ�����˶�", NULL, NULL );//����Ѿ���õ�һ֡�û��򶨵�box�ˣ���ȡ�������Ӧ
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");
  //TLD initialization
  tld.init(last_gray,box,bb_file);

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1;
  vector<Point2f> pts2;
  bool status=true;//��¼���ٳɹ�����״̬ whether lastbox  has been found  
  int frames = 1;//��¼�ѹ�ȥ֡�� 
  int detections = 1;//��¼�ɹ���⵽��Ŀ��box��Ŀ
  CvPoint point0;  
REPEAT:
  while (capture.read(frame))
  {

	point0.x = frame.rows *7/ 12;
	point0.y = frame.cols / 3;
    //get frame
    cvtColor(frame, current_gray, CV_RGB2GRAY);

    //Process Frame
    tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
	circle(frame, point0, 15, Scalar(0, 0, 255), 2);//��ͼ������Բ
    //Draw Points
    if (status)//������ٳɹ� 
	{
      drawPoints(frame,pts1);
      drawPoints(frame,pts2,Scalar(0,255,0));//��ǰ������������ɫ���ʾ
	  drawBox(frame, pbox, Scalar(0, 0, 255),2);//��ǰ�ı߽�ͼ���ú�ɫ	  
      detections++;
    }
    //Display
    imshow("�ϲ���ѧ�����˶�", frame);
	printf("ˮƽƫ��: %d-%d = %d\n", pbox.tl().x, point0.x, pbox.tl().x - point0.x);//pbox.tl().x + pbox.tl().width/2 Ӧ�������
	printf("��ֱƫ��: %d-%d = %d\n", pbox.tl().y, point0.x, pbox.tl().y - point0.y);
    //swap points and images
    swap(last_gray,current_gray);//STL����swap()���������������ֵ���䷺�ͻ��汾������<algorithm>;  
    pts1.clear();
    pts2.clear();
    frames++;
    printf("�����: %d/%d\n",detections,frames);	
	printf("###########################################################\n");
    if (cvWaitKey(12) == 'q')
      break;
  }
  if (rep)//rep����
  {
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  fclose(bb_file);
  return 0;
}
#endif