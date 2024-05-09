#pragma once

#include <iostream>
#include<fstream>
#include <io.h>
#include <stdio.h>

#include <string>

#include <opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;


struct detectionResult;
vector<String> getOutputsNames(const Net& net);
void NMS(std::vector<detectionResult>& vResultRect);
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat frame, vector<string> classes, Mat &saveImg);
void remove_box(Mat frame, const vector<Mat>& outs, float conf_threshold, vector<string> classes, float nms, vector<string>& result_label, vector<float>& result_conf);
void detect_main(string img_name, vector<string>& result_label, vector<float>& result_conf, vector<vector<int>> & location);