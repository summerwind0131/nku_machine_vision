#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

Mat myConv(Mat img)
{
	Mat Conv_img;

	float kernel[3][3]={
		{1/9.0f, 1/9.0f,1/9.0f},
		{1/9.0f, 1/9.0f,1/9.0f},
		{1/9.0f, 1/9.0f,1/9.0f}
	};

	for(int i=1;i<img.rows-1;i++){
		for(int j=1;j<img.cols-1;j++){
			float sum=0.0f;

			for(int m=-1;m<=1;m++){
				for(int n=-1;n<=1;n++){
					int pixel_val=img.at<char>(i+m,j+n);
					sum+=pixel_val*kernel[m+1][n+1];
				}
			}
			Conv_img.at<uchar>(i,j)=saturate_cast<uchar>(sum);
		}
	}

	//返回卷积处理后的图像
	return Conv_img;
}

void main()
{
	Mat input = imread("../../data/testimg.jpg");

	Mat gray;
	//彩色图转为灰度图
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//图像卷积处理，需编程实现

	Mat Conv_img = myConv(gray);

	imshow("Conv_img", Conv_img);
	waitKey(0);
}