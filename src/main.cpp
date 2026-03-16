#include <iostream>
#include <opencv2/opencv.hpp>
//test测试
int main() {
    // 打印 OpenCV 版本
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    // 创建一个 300x300 的黑色图像 (8位无符号整型，3通道)
    cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC3);

    // 在图像上写点字
    cv::putText(img, "C++ OpenCV OK!", cv::Point(50, 150), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    // 显示图像
    cv::imshow("C++ Verification", img);
    
    // 等待任意按键后关闭窗口
    cv::waitKey(0); 
    cv::destroyAllWindows();

    return 0;
}