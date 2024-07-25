#include <iostream>
#include "objectDetect.h"
#include "location.h"
#include "color.h"

using namespace std;

void main()
{
	string img_name = "{test image path}"     # ex) "test/dog.jpg";

	vector<string> result_label;
	vector<float> result_conf;
	vector<vector<int>> result_coordinate;		// x1, y1, x2, y2
	vector<string> result_location;

	// step 1. object detection
	cout << "step 1. object detection" << endl;
	detect_main(img_name, result_label, result_conf, result_coordinate);

	// step 2. extract location information
	cout << "step 2. extract location information" << endl;
	extractLocation(img_name, result_coordinate, result_location);

	// step 3. extract color information
	cout << "step 3. extract color information" << endl;
	MeanShift MSProc(8, 22);
	vector<string> result_color;

	for (int i = 0; i < result_label.size(); i++)
	{
		string bbFile = "result/" + result_label[i] + ".jpg";
		string saveSegmentImage = "result/segment_" + result_label[i] + ".jpg";
		string saveColorImage = "result/color_" + result_label[i] + ".jpg";

		MSProc.mainExe(bbFile, saveSegmentImage, saveColorImage, result_color);
	}

	// step 4. make sentence
	cout << "step 4. ma0ke sentence" << endl;
	string result;
	string object, color, location;
	if (result_label.size() == 1)
	{
		result = "There is a ";

		object = result_label[0];
		color = result_color[0];
		location = result_location[0];

		result += (color + " " + object + " in the " + location + " in the image.");
	}

	else
	{
		result = "There are ";
		for (int i = 0; i < result_label.size(); i++)
		{
			if (i == result_label.size() - 1)
			{
				object = result_label[i];
				color = result_color[i];
				location = result_location[i];

				result += ("a " + color + " " + object + " in the " + location + " in the image.");
			}

			else
			{
				object = result_label[i];
				color = result_color[i];
				location = result_location[i];

				result += ("a " + color + " " + object + " in the " + location + " and ");
			}
		}
	}
	cout << result << endl;
}
