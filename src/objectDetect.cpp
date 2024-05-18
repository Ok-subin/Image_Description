
#include "objectDetect.h"

struct detectionResult
{
	cv::Rect plateRect;
	double confidence;
	int type;
};

vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		vector<int> outLayers = net.getUnconnectedOutLayers();

		vector<String> layersNames = net.getLayerNames();

		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void NMS(std::vector<detectionResult>& vResultRect)
{
	for (int i = 0; i < vResultRect.size() - 1; i++)
	{
		for (int j = i + 1; j < vResultRect.size(); j++)
		{
			double IOURate = (double)(vResultRect[i].plateRect & vResultRect[j].plateRect).area() / (vResultRect[i].plateRect | vResultRect[j].plateRect).area();
			if (IOURate >= 0.5)
			{
				if (vResultRect[i].confidence > vResultRect[j].confidence) {
					vResultRect.erase(vResultRect.begin() + j);
					j--;
				}
				else {
					vResultRect.erase(vResultRect.begin() + i);
					i--;
					break;
				}
			}
		}
	}
}

void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat frame, vector<string> classes, Mat& saveImg)
{
	rectangle(saveImg, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		label = classes[classId] + ":" + label;
	}

	//cout << label << endl;

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);

	// rectangle(img, Rect, Scalar, thickness, lineType, shift)
	// Rect : 사각형 범위 = (Point(x1, y1), Point(x2, y2))
	// 또는 Rect(x, y, w, h)
	rectangle(saveImg,
		Point(left, top - round(1.5 * labelSize.height)),
		Point(left + round(1.5 * labelSize.width), top + baseLine),
		Scalar(255, 255, 255), FILLED);
	putText(saveImg, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

	Mat result = frame(Rect(Point(left, top), Point(right, bottom)));

	int count = 0;
	string save_name = "";
	while (true)
	{
		save_name = "result/" + classes[classId] + ".jpg";
		const char* save_name_char = save_name.c_str();

		if ((_access(save_name_char, 0) == -1))
		{
			break;
		}
		else
		{
			count++;
			save_name = ("result/" + classes[classId] + to_string(count) + ".jpg").c_str();
			save_name_char = save_name.c_str();
		}
	}
	imwrite(save_name, result);

}

void remove_box(Mat frame, const vector<Mat>& outs, float conf_threshold, vector<string> classes, float nms,
	vector<string>& result_label, vector<float>& result_conf, vector<vector<int>>& location)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	Mat saveImg = frame.clone();

	for (size_t i = 0; i < outs.size(); ++i)
	{
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;

			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > conf_threshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	vector<int> indices;

	//cout << "-----BEFORE NMS------" << endl;
	NMSBoxes(boxes, confidences, conf_threshold, nms, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		//cout << "draw" << endl;
		int idx = indices[i];

		Rect box = boxes[idx];
		draw_box(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes, saveImg);

		result_label.push_back(classes[classIds[idx]]);
		result_conf.push_back(confidences[idx]);

		//cout << classes[classIds[idx]] << " , " << confidences[idx] << endl;

		vector<int> temp = { box.x, box.y, box.x + box.width, box.y + box.height };

		//cout << "\nfor 내부" << endl;
		//for (int j = 0; j < 4; j++)
		//{
		//	cout << temp[j] << ", ";
		//}

		location.push_back(temp);
	}

	imwrite("result/box_img.jpg", saveImg);
}

void detect_main(string img_name, vector<string>& result_label, vector<float>& result_conf, vector<vector<int>>& location)
{
	vector<string> classes;

	float conf_threshold = 0.5;
	float nms = 0.4;
	int width = 416;
	int height = 416;

	// load label
	string classesFile = "data/coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// load model
	String modelConfiguration = "data/yolov3.cfg";
	String modelWeights = "data/yolov3.weights";

	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// load image
	//Mat img = cv::imread("test/dog.jpg");
	Mat img = cv::imread(img_name);

	if (img.empty())
	{
		cerr << "Image Loaded Fail" << endl;
		return;
	}

	Mat inputBlob = blobFromImage(img, 1 / 255., Size(width, height), Scalar(), true, false);
	net.setInput(inputBlob);

	// generate output
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// 결과 저장할 vector
	//cout << "-----BEFORE remove_box-------" << endl;
	remove_box(img, outs, conf_threshold, classes, nms, result_label, result_conf, location);

	//vector<double> layersTimes;
	//double freq = getTickFrequency() / 1000;
	//double t = net.getPerfProfile(layersTimes) / freq;
	//string label = format("Inference time for a frame : %.2f ms", t);
	//putText(img, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

	//Mat detectedFrame;
	//img.convertTo(detectedFrame, CV_8U);
	//static const string kWinName = "Deep learning object detection in OpenCV";

	//imshow(kWinName, img);
	//waitKey();
}
