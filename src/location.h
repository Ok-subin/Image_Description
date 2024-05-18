#pragma once
#include <iostream>

#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

float IOU(int roi[], float compare[]);
void extractLocation(string originalFile, vector<vector<int>> result_coordinate, vector<string>& result_location);
