#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dirent.h>//遍历系统指定目录下文件要包含的头文件
#include <string>
#include "ORBMatching.h"

using namespace std;
using namespace cv;

double DIST_THRESHOLD = 20;
int NUM_MATCH_POINTS = 500;

//ORB特征提取
//1.读取图像
//2.初始化KeyPoint、Descriptor、ORB对象
//3.检测FAST角点
//4.由角点位置计算BRIEF描述子
//5.新建匹配对象以及vector用于存放点对，对两幅图中的BRIEF描述子进行匹配，使用Hamming距离
//6.筛选匹配点对（距离小于最小值的两倍）
//7.绘制匹配结果
vector<vector<Point2f>> orb_match2(string path1,string path2,string outPath)
{
  //Step1
  Mat img1 = imread(path1);
  Mat img2 = imread(path2);

  //Step2
  vector<KeyPoint> keyPoint1,keyPoint2;
  Mat descriptor1,descriptor2;
  //!!!新建一个ORB对象，注意create的参数!!!
  Ptr<ORB> orb = ORB::create(NUM_MATCH_POINTS,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  //Step3
  orb->detect(img1,keyPoint1);
  orb->detect(img2,keyPoint2);
  
  //Step4
  orb->compute(img1,keyPoint1,descriptor1);
  orb->compute(img2,keyPoint2,descriptor2);
  
  //Step5
  //!!!注意表示匹配点对用DMatch类型，以及匹配对象的新建方法!!!
  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(descriptor1,descriptor2,matches);
  
  //Step6
  double min_dist = 100;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
  }
  
  vector<Point2f> points1;
  vector<Point2f> points2;
  vector<DMatch> good_matches;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<max(2*min_dist,DIST_THRESHOLD))
    {
      good_matches.push_back(matches[i]);
      //注意这两个Idx是不一样的
      points1.push_back(keyPoint1[matches[i].queryIdx].pt);
      points2.push_back(keyPoint2[matches[i].trainIdx].pt);
    }
  }
  vector<vector<Point2f>> result;
  result.push_back(points1);
  result.push_back(points2);
  
  //Step7
  Mat outImg;
  drawMatches(img1,keyPoint1,img2,keyPoint2,good_matches,outImg);
  imwrite(outPath,outImg);
  cout<<"保留的匹配点对："<<good_matches.size()<<endl;
  
  return result;
}


void orb_match(string path1,string path2,string outPath)
{
  //Step1
  Mat img1 = imread(path1);
  Mat img2 = imread(path2);

  //Step2
  vector<KeyPoint> keyPoint1,keyPoint2;
  Mat descriptor1,descriptor2;
  //!!!新建一个ORB对象，注意create的参数!!!
  Ptr<ORB> orb = ORB::create(NUM_MATCH_POINTS,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  //Step3
  orb->detect(img1,keyPoint1);
  orb->detect(img2,keyPoint2);
  
  //Step4
  orb->compute(img1,keyPoint1,descriptor1);
  orb->compute(img2,keyPoint2,descriptor2);
  
  //Step5
  //!!!注意表示匹配点对用DMatch类型，以及匹配对象的新建方法!!!
  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(descriptor1,descriptor2,matches);
  
  //Step6
  double min_dist = 100;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
  }
  
  vector<DMatch> good_matches;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<max(2*min_dist,DIST_THRESHOLD))
    {
      good_matches.push_back(matches[i]);
    }
  }
  
  //Step7
  Mat outImg;
  drawMatches(img1,keyPoint1,img2,keyPoint2,good_matches,outImg);
  imwrite(outPath,outImg);
//   cout<<"保留的匹配点对："<<good_matches.size()<<endl;
}

vector<vector<Point2f>> getMatchPoints(string path1,string path2)
{
  //Step1
  Mat img1 = imread(path1);
  Mat img2 = imread(path2);

  //Step2
  vector<KeyPoint> keyPoint1,keyPoint2;
  Mat descriptor1,descriptor2;
  //!!!新建一个ORB对象，注意create的参数!!!
  Ptr<ORB> orb = ORB::create(NUM_MATCH_POINTS,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  //Step3
  orb->detect(img1,keyPoint1);
  orb->detect(img2,keyPoint2);
  
  //Step4
  orb->compute(img1,keyPoint1,descriptor1);
  orb->compute(img2,keyPoint2,descriptor2);
  
  //Step5
  //!!!注意表示匹配点对用DMatch类型，以及匹配对象的新建方法!!!
  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(descriptor1,descriptor2,matches);
  
  //Step6
  double min_dist = 100;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
  }
  
  vector<Point2f> points1;
  vector<Point2f> points2;
  vector<DMatch> good_matches;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<max(2*min_dist,DIST_THRESHOLD))
    {
      good_matches.push_back(matches[i]);
      //注意这两个Idx是不一样的
      points1.push_back(keyPoint1[matches[i].queryIdx].pt);
      points2.push_back(keyPoint2[matches[i].trainIdx].pt);
    }
  }
  
//   cout<<"Num of good matches:"<<good_matches.size()<<endl;
  
  vector<vector<Point2f>> result;
  result.push_back(points1);
  result.push_back(points2);
  
  return result;
}

void getMatchPoints(string path1,string path2,vector<Point2f> &pts1,vector<Point2f> &pts2)
{
  cout<<"--------matched points--------"<<endl;
  //Step1
  Mat img1 = imread(path1);
  Mat img2 = imread(path2);

  //Step2
  vector<KeyPoint> keyPoint1,keyPoint2;
  Mat descriptor1,descriptor2;
  //!!!新建一个ORB对象，注意create的参数!!!
  Ptr<ORB> orb = ORB::create(NUM_MATCH_POINTS,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  //Step3
  orb->detect(img1,keyPoint1);
  orb->detect(img2,keyPoint2);
  
  //Step4
  orb->compute(img1,keyPoint1,descriptor1);
  orb->compute(img2,keyPoint2,descriptor2);
  
  //Step5
  //!!!注意表示匹配点对用DMatch类型，以及匹配对象的新建方法!!!
  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(descriptor1,descriptor2,matches);
  
  //Step6
  double min_dist = 100;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
  }
  
  vector<DMatch> good_matches;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<max(2*min_dist,DIST_THRESHOLD))
    {
      good_matches.push_back(matches[i]);
      //注意这两个Idx是不一样的
      pts1.push_back(keyPoint1[matches[i].queryIdx].pt);
      pts2.push_back(keyPoint2[matches[i].trainIdx].pt);
    }
  }

  for (int j = 0; j < pts1.size(); ++j) {
    cout<<"point pair "<<j+1<<":"<<pts1[j]<<" "<<pts2[j]<<endl;
  }
  cout<<"--------matched points--------"<<endl;
  return;
}

vector<vector<Point2f>> getMatchPointsIMG(Mat img1,Mat img2)
{
  //Step1

  //Step2
  vector<KeyPoint> keyPoint1,keyPoint2;
  Mat descriptor1,descriptor2;
  //!!!新建一个ORB对象，注意create的参数!!!
  Ptr<ORB> orb = ORB::create(NUM_MATCH_POINTS,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
  
  //Step3
  orb->detect(img1,keyPoint1);
  orb->detect(img2,keyPoint2);
  
  //Step4
  orb->compute(img1,keyPoint1,descriptor1);
  orb->compute(img2,keyPoint2,descriptor2);
  
  //Step5
  //!!!注意表示匹配点对用DMatch类型，以及匹配对象的新建方法!!!
  vector<DMatch> matches;
  BFMatcher matcher = BFMatcher(NORM_HAMMING);
  matcher.match(descriptor1,descriptor2,matches);
  
  //Step6
  double min_dist = 100;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<min_dist)
    {
      min_dist = matches[i].distance;
    }
  }
  
  vector<Point2f> points1;
  vector<Point2f> points2;
  vector<DMatch> good_matches;
  for(int i=0;i<matches.size();i++)
  {
    if(matches[i].distance<max(2*min_dist,DIST_THRESHOLD))
    {
      good_matches.push_back(matches[i]);
      //注意这两个Idx是不一样的
      points1.push_back(keyPoint1[matches[i].queryIdx].pt);
      points2.push_back(keyPoint2[matches[i].trainIdx].pt);
    }
  }
  
  vector<vector<Point2f>> result;
  result.push_back(points1);
  result.push_back(points2);
  
  return result;
}