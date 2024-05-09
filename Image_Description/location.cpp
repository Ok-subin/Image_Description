#include "location.h"

float IOU(int roi[], float compare[])
{
	float x1, y1, x2, y2;
	float width, height, area_overlap, area_a, area_b, area_combined;
	float interArea, boxAArea, boxBArea;
	float intersection_x_length, intersection_y_length;
	float iou, epsilon = 1e-5;

	x1 = max(float(roi[0]), compare[0]);
	y1 = max(float(roi[1]), compare[1]);
	x2 = min(float(roi[2]), compare[2]);
	y2 = min(float(roi[3]), compare[3]);

	interArea = max(float(0), x2 - x1 + 1) * max(float(0), y2 - y1 + 1);

	boxAArea = (roi[2] - roi[0] + 1) * (roi[3] - roi[1] + 1);
	boxBArea = (compare[2] - compare[0] + 1) * (compare[3] - compare[1] + 1);

	iou = interArea / float(boxAArea + boxBArea - interArea);

	width = (x2 - x1);
	height = (y2 - y1);

	if (width < 0 || height < 0)
		return 0.0;

	return iou;
}

void extractLocation(string originalFile, vector<vector<int>> result_coordinate, vector<string> &result_location)
{
	Mat originalImg = imread(originalFile);
	float width = originalImg.cols, height = originalImg.rows;

	float area1[4] = { 0, 0, width / 2, height / 2 };						// left top
	float area2[4] = { width / 2, 0, width, height / 2 };					// right top
	float area3[4] = { 0, height / 2, width / 2, height };					// left bottom
	float area4[4] = { width / 2, height / 2, width, height };				// right bottom
	float center[4] = { width / 4, height / 4, (width / 4) * 3, (height / 4) * 3 };	// center
	float iou[5];

	for (int i = 0; i < result_coordinate.size(); i++)
	{
		int index[4];
		for (int j = 0; j < 4; j++)
		{
			index[j] = result_coordinate[i][j];
			//cout << result_coordinate[i][j] << endl;
		}		

		iou[0] = IOU(index, area1);
		iou[1] = IOU(index, area2);
		iou[2] = IOU(index, area3);
		iou[3] = IOU(index, area4);
		iou[4] = IOU(index, center);
		float maxValue = 0.0;
		int maxIndex = 0;
		for (int x = 0; x < 5; x++)
		{
			if (iou[x] > maxValue)
			{
				maxValue = iou[x];
				maxIndex = x;
			}
		}

		//for (int j = 0; j < 5; j++)
		//{
		//	cout << iou[j] << endl;
		//}

		string resultLocation;
		switch (maxIndex)
		{
		case 0:
			resultLocation = "left top";
			break;
		case 1:
			resultLocation = "right top";
			break;
		case 2:
			resultLocation = "left bottom";
			break;
		case 3:
			resultLocation = "right bottom";
			break;
		case 4:
			resultLocation = "center";
		default:
			break;
		}

		//cout << resultLocation << endl;
		result_location.push_back(resultLocation);
	}

}