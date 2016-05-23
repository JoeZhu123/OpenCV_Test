/*
*	2014.12.24
*
*	程序修改者：朱幸尔
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
//全局模板轮廓  
vector<vector<Point>> g_TemplateContours;
//模板个数  
int g_badmintonTNum = 6;

/*
读取记录bounding box的文件，获得bounding box的四个参数：左上角坐标x，y和宽高
如在\datasets\06_car\init.txt中：记录了初始目标的bounding box，内容如下
142,125,232,164
*/
void readBB(char* file)
{
  ifstream bb_file (file);//以输入方式打开文件
  string line;
  //istream& getline ( istream& , string& );  
  //将输入流is中读到的字符存入str中，终结符默认为 '\n'（换行符）
  getline(bb_file,line);
  istringstream linestream(line);//istringstream对象可以绑定一行字符串，然后以空格为分隔符把该行分隔开来。 
  string x1,y1,x2,y2;
  //istream& getline ( istream &is , string &str , char delim );   
  //将输入流is中读到的字符存入str中，直到遇到终结符delim才结束。
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');

  //atoi 功 能： 把字符串转换成整型数
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback 
//鼠标的响应就是得到目标区域的范围，用鼠标选中bounding box。
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
    gotBB = true;//已经获得bounding box  
    break;
  }
}

void print_help(char** argv)
{
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}
//分析运行程序时的命令行参数
void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs)
{
  for (int i=0;i<argc;i++)
  {
      if (strcmp(argv[i],"-b")==0)
	  {
          if (argc>i)
		  {
              readBB(argv[i+1]);//是否指定初始的bounding box  
              gotBB = true;
          }
          else
            print_help(argv);
      }
	  //Similar in format to XML, Yahoo! Markup Language (YML) provides functionality to Open   
	  //Applications in a safe and standardized fashion. You include YML tags in the HTML code  
	  //of an Open Application.
      if (strcmp(argv[i],"-p")==0)//读取参数文件parameters.yml  
	  {
		  if (argc>i) //FileStorage类的读取方式可以是：FileStorage fs(".\\parameters.yml", FileStorage::READ);
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
运行程序时：
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
//感觉就是对起始帧进行初始化工作，然后逐帧读入图片序列，进行算法处理。
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
/*opencv背景建模，运动物体检测   视频进行背景建模，然后差分得出运动物体*/
int main(int argc, char** argv)
{
	//VideoCapture video("Test1_6.MPG");//视频测试
	
	VideoCapture capture;//摄像头捕捉
	capture.open(0);//打开摄像头

//	BackgroundSubtractorMOG bgSubtractor(20, 10, 0.5 , false);
	BackgroundSubtractorMOG2 bgSubtractor2(20,16,false);
	
	int nFrmNum = 0;
	bool stop = false;
	Mat frame;
	Mat fore_frame;
	Mat back_frame;	
	Mat closerect = getStructuringElement(MORPH_RECT, Size(3, 3)); //进行结构算子生成
	
//	Mat back_ground;

	cvNamedWindow("NCU-video", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("NCU-backframe", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("NCU-foreframe", CV_WINDOW_AUTOSIZE);
//	cvNamedWindow("NCU-background", CV_WINDOW_AUTOSIZE);

	//排列窗口
//	cvMoveWindow("NCU-video",30,0);
//	cvMoveWindow("NCU-backframe",300,0);
//	cvMoveWindow("NCU-foreframe",600,0);
//	cvMoveWindow("NCU-foreframe", 900, 0);

	//设置摄像头获取的图像大小为340x240   
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}
	
	//逐帧读取视频
	while (!stop)
	{
			capture >> frame;
			//Output file
			
			
			++nFrmNum;
//			cout << nFrmNum << endl;

			//运动前景检测，并更新背景
//			bgSubtractor(frame,back_frame,0.01);
			bgSubtractor2(frame,back_frame,0.01);

/******************opencv1.x版本的背景差分法******************************/
//先高斯滤波，以平滑图像，
//GaussianBlur(fore_frame,frame,frame.size(),0);			
//当前帧跟背景图相减
//absdiff(frame,back_frame,fore_frame);
//二值化前景图
//threshold(fore_frame,fore_frame, 60, 255.0, CV_THRESH_BINARY);
//更新背景
//cvRunningAvg(frame,back_frame, 0.003, 0);
/************************************************************************/
			//进行形态学滤波，去掉噪音
			erode(back_frame, fore_frame, Mat());
			dilate(fore_frame, fore_frame, Mat());
			morphologyEx(fore_frame, fore_frame,MORPH_OPEN,closerect);//进行形态学开运算
			/**********************************轮廓匹配*******************************************************/
			/*
			vector<vector<Point>> w, w1;
			vector<Vec4i> hierarchy, hierarchy1;
			findContours(fore_frame, w, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);//提取轮廓元素
			//findContours(f1, w1, hierarchy1, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
			FileStorage fs("f.dat", FileStorage::WRITE);
			fs << "f" << w1[0];
			int idx = 0;
			double ffff = matchShapes(w[0], w1[0], CV_CONTOURS_MATCH_I3, 1.0);//进行轮廓匹配
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
			/// 计算矩
			vector<Moments> mu(contours.size());
			///  计算中心矩:
			vector<Point2f> mc(contours.size());
			//iterate through all the top-level contours
			//draw each connected component with its own random color		
			for (int idx = 0; idx < contours.size(); idx++)
			{
				//除去太长或者太短的轮廓(注：只是一个非常简单的滤除干扰)  
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
							mu[idx] = moments(contours[idx], false);// 计算矩
							mc[idx] = Point2f(mu[idx].m10 / mu[idx].m00, mu[idx].m01 / mu[idx].m00);//  计算中心矩

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
	//销毁窗口
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
  VideoCapture capture;//摄像头捕捉
  capture.open(0);//打开摄像头 
  //OpenCV的C++接口中，用于保存图像的imwrite只能保存整数数据，且需作为图像格式。当需要保存浮  
  //点数据或XML/YML文件时，OpenCV的C语言接口提供了cvSave函数，但这一函数在C++接口中已经被删除。  
  //取而代之的是FileStorage类。
  FileStorage fs;
  //Read options
  read_options(argc, argv, capture, fs); //分析命令行参数 
  //程序修改后，不用Read options，直接就是调用摄像头，所以命令行参数输入：KT1 -p D:\VSProjects\KT1\parameters.yml
   //argv[2]代表parameters.yml的文件位置字符串：D:\VSProjects\KT1\parameters.yml
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
  cvNamedWindow("南昌大学机器人队",CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback( "南昌大学机器人队", mouseHandler, NULL );//用鼠标选中初始目标的bounding box  
  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  
  //设置摄像头获取的图像大小为340x240   
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  

  ///Initialization
GETBOUNDINGBOX://标志位：获取bounding box  
  while(!gotBB)
  {
	  capture >> frame;
    cvtColor(frame, last_gray, CV_RGB2GRAY);
	drawBox(frame, box,Scalar(0, 0, 255), 2);//把bounding box 画出来  
    imshow("南昌大学机器人队", frame);
    if (cvWaitKey(12) == 'q')
	    return 0;
  }
  //由于图像片（min_win 为15x15像素）是在bounding box中采样得到的，所以box必须比min_win要大
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"])
  {
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  //Remove callback
  cvSetMouseCallback( "南昌大学机器人队", NULL, NULL );//如果已经获得第一帧用户框定的box了，就取消鼠标响应
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
  bool status=true;//记录跟踪成功与否的状态 whether lastbox  has been found  
  int frames = 1;//记录已过去帧数 
  int detections = 1;//记录成功检测到的目标box数目
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
	circle(frame, point0, 15, Scalar(0, 0, 255), 2);//画图像中心圆
    //Draw Points
    if (status)//如果跟踪成功 
	{
      drawPoints(frame,pts1);
      drawPoints(frame,pts2,Scalar(0,255,0));//当前的特征点用蓝色点表示
	  drawBox(frame, pbox, Scalar(0, 0, 255),2);//当前的边界图框用红色	  
      detections++;
    }
    //Display
    imshow("南昌大学机器人队", frame);
	printf("水平偏差: %d-%d = %d\n", pbox.tl().x, point0.x, pbox.tl().x - point0.x);//pbox.tl().x + pbox.tl().width/2 应该是这个
	printf("竖直偏差: %d-%d = %d\n", pbox.tl().y, point0.x, pbox.tl().y - point0.y);
    //swap points and images
    swap(last_gray,current_gray);//STL函数swap()用来交换两对象的值。其泛型化版本定义于<algorithm>;  
    pts1.clear();
    pts2.clear();
    frames++;
    printf("检测率: %d/%d\n",detections,frames);	
	printf("###########################################################\n");
    if (cvWaitKey(12) == 'q')
      break;
  }
  if (rep)//rep参数
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