#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dirent.h>//遍历系统指定目录下文件要包含的头文件
#include <string>

using namespace std;
using namespace cv;

void orb_match(string path1,string path2,string outPath);
vector<vector<Point2f>> orb_match2(string path1,string path2,string outPath);
vector<vector<Point2f>> getMatchPoints(string path1,string path2);
vector<vector<Point2f>> getMatchPointsIMG(Mat img1,Mat img2);
void getMatchPoints(string path1,string path2,vector<Point2f> &pts1,vector<Point2f> &pts2);