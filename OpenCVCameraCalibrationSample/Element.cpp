#include "stdafx.h"
#include "Element.h"
#define PI 3.1415

using namespace std;
using namespace cv;

Mat getElement(int elHeight, calibrationResult calibData)
{
	const int elX = 0;
	const int elY = 0;

	const int bufX = 1000;
	const int bufY = 1000;

	Mat el = Mat::zeros(bufX, bufY, CV_8UC1);
	
	vector<Point3d> objectPoints;
	vector<Point2d> imagePoints;

	int lastImgIndex = calibData.distCoeffs.rows-1;

	objectPoints.push_back(Point3d(elX,elY, 0));
	objectPoints.push_back(Point3d(elX,elY, -elHeight));
	

	cv::projectPoints(Mat(objectPoints), calibData.rvecs[lastImgIndex], calibData.tvecs[lastImgIndex], calibData.cameraMatrix,
		calibData.distCoeffs, imagePoints);

	pair<double, double> elVector;
	double norm;
	double cosinus;
	double sinus;
	int angle;

	elVector = make_pair(imagePoints[1].x - imagePoints[0].x, imagePoints[1].y - imagePoints[0].y);
	norm = sqrt(elVector.first*elVector.first + elVector.second*elVector.second);
	cosinus = elVector.first / norm;
	sinus = elVector.second / norm;
	angle = (int)(acos(cosinus) * 180 / PI * (sinus > 0 ? -1 : 1)*(sinus == 0 ? 0 : 1) + 360) % 360;
	ellipse(el, Point2d(bufX / 2, bufY/2), Size(norm *0.5, norm*0.5*0.5), angle, 0, 360, Scalar(255, 255, 255), -1, 8, 0);
	

	Mat Points;
	findNonZero(el, Points);
	Rect Min_Rect = boundingRect(Points);
	rectangle(el, Min_Rect.tl(), Min_Rect.br(), Scalar(0, 255, 0), 2);
	cv::Mat croppedImage = el(Min_Rect);
	
	cv::namedWindow("Element projected");
	//cvResizeWindow("Element projected", 640, 480);
	imshow("Element projected", croppedImage);

	return croppedImage;
}