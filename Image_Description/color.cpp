#include "color.h"

#define MS_MAX_NUM_CONVERGENCE_STEPS	5									
#define MS_MEAN_SHIFT_TOL_COLOR			0.3										
#define MS_MEAN_SHIFT_TOL_SPATIAL		0.3										
const int dxdy[][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };

Point5D::Point5D() {
	x = -1;
	y = -1;
}

Point5D::~Point5D() {
}

void Point5D::PointLab() {
	l = l * 100 / 255;
	a = a - 128;
	b = b - 128;
}

void Point5D::PointRGB() {
	l = l * 255 / 100;
	a = a + 128;
	b = b + 128;
}

void Point5D::MSPoint5DAccum(Point5D Pt) {
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	a += Pt.a;
	b += Pt.b;
}

void Point5D::MSPoint5DCopy(Point5D Pt) {
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	a = Pt.a;
	b = Pt.b;
}

float Point5D::MSPoint5DColorDistance(Point5D Pt) {
	return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}

float Point5D::MSPoint5DSpatialDistance(Point5D Pt) {
	return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

void Point5D::MSPoint5DScale(float scale) {
	x *= scale;
	y *= scale;
	l *= scale;
	a *= scale;
	b *= scale;
}

void Point5D::MSPOint5DSet(float px, float py, float pl, float pa, float pb) {
	x = px;
	y = py;
	l = pl;
	a = pa;
	b = pb;
}

void Point5D::Print() {
	cout << x << " " << y << " " << l << " " << a << " " << b << endl;
}


void RGBtoHSV(int& fR, int& fG, int fB, float& fH, float& fS, float& fV) {
	float fCMax = max(max(fR, fG), fB);
	float fCMin = min(min(fR, fG), fB);
	float fDelta = fCMax - fCMin;

	if (fDelta > 0) {
		if (fCMax == fR) {
			fH = 60 * (fmod(((fG - fB) / fDelta), 6));
		}
		else if (fCMax == fG) {
			fH = 60 * (((fB - fR) / fDelta) + 2);
		}
		else if (fCMax == fB) {
			fH = 60 * (((fR - fG) / fDelta) + 4);
		}

		if (fCMax > 0) {
			fS = fDelta / fCMax;
		}
		else {
			fS = 0;
		}

		fV = fCMax;
	}
	else {
		fH = 0;
		fS = 0;
		fV = fCMax;
	}

	if (fH < 0) {
		fH = 360 + fH;
	}
}

MeanShift::MeanShift(float s, float r) {
	hs = s;
	hr = r;
}

void MeanShift::MSFiltering(Mat& Img) {
	int ROWS = Img.rows;
	int COLS = Img.cols;
	split(Img, IMGChannels);

	Point5D PtCur;
	Point5D PtPrev;
	Point5D PtSum;
	Point5D Pt;
	int Left;
	int Right;
	int Top;
	int Bottom;
	int NumPts;
	int step;

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			Left = (j - hs) > 0 ? (j - hs) : 0;
			Right = (j + hs) < COLS ? (j + hs) : COLS;
			Top = (i - hs) > 0 ? (i - hs) : 0;
			Bottom = (i + hs) < ROWS ? (i + hs) : ROWS;

			PtCur.MSPOint5DSet(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
			PtCur.PointLab();
			step = 0;

			do {
				PtPrev.MSPoint5DCopy(PtCur);
				PtSum.MSPOint5DSet(0, 0, 0, 0, 0);
				NumPts = 0;

				for (int hx = Top; hx < Bottom; hx++) {
					for (int hy = Left; hy < Right; hy++) {
						Pt.MSPOint5DSet(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
						Pt.PointLab();

						if (Pt.MSPoint5DColorDistance(PtCur) < hr) {
							PtSum.MSPoint5DAccum(Pt);
							NumPts++;
						}
					}
				}
				PtSum.MSPoint5DScale(1.0 / NumPts);
				PtCur.MSPoint5DCopy(PtSum);
				step++;

			} while ((PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR) && (PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL) && (step < MS_MAX_NUM_CONVERGENCE_STEPS));

			PtCur.PointRGB();
			Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b);
		}
	}
}

void MeanShift::MSSegmentation(Mat& Img, float& indexX, float& indexY, int& totalLabelSum) {

	int ROWS = Img.rows;
	int COLS = Img.cols;
	split(Img, IMGChannels);

	Point5D PtCur;
	Point5D PtPrev;
	Point5D PtSum;
	Point5D Pt;
	int Left;
	int Right;
	int Top;
	int Bottom;
	int NumPts;
	int step;

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			Left = (j - hs) > 0 ? (j - hs) : 0;
			Right = (j + hs) < COLS ? (j + hs) : COLS;
			Top = (i - hs) > 0 ? (i - hs) : 0;
			Bottom = (i + hs) < ROWS ? (i + hs) : ROWS;
			PtCur.MSPOint5DSet(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
			PtCur.PointLab();
			step = 0;

			do {
				PtPrev.MSPoint5DCopy(PtCur);
				PtSum.MSPOint5DSet(0, 0, 0, 0, 0);
				NumPts = 0;
				for (int hx = Top; hx < Bottom; hx++) {
					for (int hy = Left; hy < Right; hy++) {

						Pt.MSPOint5DSet(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
						Pt.PointLab();

						if (Pt.MSPoint5DColorDistance(PtCur) < hr) {
							PtSum.MSPoint5DAccum(Pt);
							NumPts++;
						}
					}
				}
				PtSum.MSPoint5DScale(1.0 / NumPts);
				PtCur.MSPoint5DCopy(PtSum);
				step++;

			} while ((PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR) && (PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL) && (step < MS_MAX_NUM_CONVERGENCE_STEPS));

			PtCur.PointRGB();
			Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b);
		}
	}

	int RegionNumber = 0;
	int label = -1;
	float* Mode = new float[ROWS * COLS * 3];
	int* MemberModeCount = new int[ROWS * COLS];
	memset(MemberModeCount, 0, ROWS * COLS * sizeof(int));
	split(Img, IMGChannels);
	int** Labels = new int* [ROWS];
	for (int i = 0; i < ROWS; i++)
		Labels[i] = new int[COLS];

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			Labels[i][j] = -1;
		}
	}


	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++) {
			if (Labels[i][j] < 0)
			{
				Labels[i][j] = ++label;
				PtCur.MSPOint5DSet(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
				PtCur.PointLab();

				Mode[label * 3 + 0] = PtCur.l;
				Mode[label * 3 + 1] = PtCur.a;
				Mode[label * 3 + 2] = PtCur.b;

				vector<Point5D> NeighbourPoints;
				NeighbourPoints.push_back(PtCur);
				while (!NeighbourPoints.empty())
				{
					Pt = NeighbourPoints.back();
					NeighbourPoints.pop_back();

					for (int k = 0; k < 8; k++) {
						int hx = Pt.x + dxdy[k][0];
						int hy = Pt.y + dxdy[k][1];
						if ((hx >= 0) && (hy >= 0) && (hx < ROWS) && (hy < COLS) && (Labels[hx][hy] < 0))
						{
							Point5D P;
							P.MSPOint5DSet(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
							P.PointLab();

							if (PtCur.MSPoint5DColorDistance(P) < hr) {
								Labels[hx][hy] = label;
								NeighbourPoints.push_back(P);
								MemberModeCount[label]++;
								Mode[label * 3 + 0] += P.l;
								Mode[label * 3 + 1] += P.a;
								Mode[label * 3 + 2] += P.b;
							}
						}
					}
				}

				MemberModeCount[label]++;
				Mode[label * 3 + 0] /= MemberModeCount[label];
				Mode[label * 3 + 1] /= MemberModeCount[label];
				Mode[label * 3 + 2] /= MemberModeCount[label];
			}
		}
	}
	RegionNumber = label + 1;


	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			label = Labels[i][j];
			float l = Mode[label * 3 + 0];
			float a = Mode[label * 3 + 1];
			float b = Mode[label * 3 + 2];
			Point5D Pixel;
			Pixel.MSPOint5DSet(i, j, l, a, b);
			Pixel.PointRGB();
			Img.at<Vec3b>(i, j) = Vec3b(Pixel.l, Pixel.a, Pixel.b);
		}
	}

	int max = 0, labelSize = 0;
	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			if (max < Labels[i][j])
			{
				max = Labels[i][j];
			}
			labelSize++;
		}
	}

	int* labelCount = new int[max + 1];

	for (int i = 0; i <= max; i++)
	{
		labelCount[i] = 0;
	}


	int labelValue;
	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			labelValue = Labels[i][j];
			labelCount[labelValue]++;

		}
	}

	int labelMax = 0, maximumLabelIndex = 0;
	for (int i = 0; i <= max; i++)
	{
		if (labelMax < labelCount[i])
		{
			labelMax = labelCount[i];
			maximumLabelIndex = i;
		}
	}

	int anyIndexX, anyIndexY;
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++)
		{
			if (Labels[i][j] == maximumLabelIndex)
			{
				anyIndexX = i;
				anyIndexY = j;

				totalLabelSum++;
			}
		}
	}

	indexX = anyIndexX;
	indexY = anyIndexY;

	delete[] Mode;
	delete[] MemberModeCount;

	for (int i = 0; i < ROWS; i++)
		delete[] Labels[i];
	delete[] Labels;
}

float Distance(int point1[], int point2[])
{
	float distance, d1, d2, d3;

	if (point1[0] < point2[0])
	{
		d1 = (point2[0] - point1[0]) ^ 2;
	}

	else
	{
		d1 = (point1[0] - point2[0]) ^ 2;
	}

	if (point1[1] < point2[1])
	{
		d2 = (point2[1] - point1[1]) ^ 2;
	}

	else
	{
		d2 = (point1[1] - point2[1]) ^ 2;
	}

	if (point1[2] < point2[2])
	{
		d3 = (point2[2] - point1[2]) ^ 2;
	}

	else
	{
		d3 = (point1[2] - point2[2]) ^ 2;
	}

	distance = sqrt(d1 + d2 + d3);

	return distance;
}

void MeanShift::colorRecognition(int R, int G, int B, vector<string>& result_color)
{
	float H, S, V;
	RGBtoHSV(R, G, B, H, S, V);
	string color = "";
	int rgb[3] = { R, G, B };

	if (S < 0.27)
	{
		float meanColor = (R + G + B) / 3.0;
		if (248 < meanColor <= 255)
		{
			color = "White";
		}
		else if (140 < meanColor <= 248)
		{
			color = "Gray";
		}
		else if (68 < meanColor <= 140)
		{
			color = "Dark Gray";
		}
		else if (0 <= meanColor <= 68)
		{
			color = "Black";
		}
	}

	else {
		int colorList[6][3] = { {0,0,255}, {0,255,0}, {255,0,0}, {0,255,255}, {255,0,255}, {255,255,0} };
		string colorName[6] = { "Blue", "Green", "Red", "Cyan", "Magenta", "Yellow" };
		float minDistance = 999;
		int minIndex;

		for (int i = 0; i < 8; i++)
		{
			if (minDistance > Distance(rgb, colorList[i]))
			{
				minDistance = Distance(rgb, colorList[i]);
				minIndex = i;
			}

		}
		color = colorName[minIndex];
	}

	//cout << color << endl;
	result_color.push_back(color);
}

void MeanShift::mainExe(string bbImageName, string saveSegmentImage, string saveColorImage, vector<string> &result_color)
{
	Mat Img = imread(bbImageName);
	resize(Img, Img, Size(256, 256), 0, 0, 1);
	cvtColor(Img, Img, COLOR_RGB2Lab);

	int totalLabelSum = 0;
	float rValue, gValue, bValue;
	float indexX, indexY;
	MSSegmentation(Img, indexX, indexY, totalLabelSum);
	cvtColor(Img, Img, COLOR_Lab2RGB);
	bValue = Img.at<Vec3b>(indexY, indexX)[0];
	gValue = Img.at<Vec3b>(indexY, indexX)[1];
	rValue = Img.at<Vec3b>(indexY, indexX)[2];


	Mat resultImg(Size(300, 300), CV_8UC3, Scalar(bValue, gValue, rValue));

	float labelratio;
	labelratio = (float(totalLabelSum) / (256 * 256)) * 100.0;
	imwrite(saveColorImage, resultImg);
	imwrite(saveSegmentImage, Img);

	colorRecognition(rValue, gValue, bValue, result_color);
}