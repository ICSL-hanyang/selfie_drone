#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "simple_tracking/centerMsg.h"
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	ros::init(argc,argv, "simple_tracking_publisher");
	ros::NodeHandle nh;

	VideoCapture capture(0);
	Mat frame;
	Mat gray_image;
	Mat roiImg;
	vector < vector<Point > > contours;
	vector<Vec4i> hierarchy;

	if (!capture.isOpened()) {
		std::cerr << "Could not open camera" << std::endl;
		return 0;
	}
	ros::Publisher simple_tracking_pub = nh.advertise<simple_tracking::centerMsg>("simple_tracking_msg",100);
	ros::Rate loop_rate(10);

	while (true) {
		simple_tracking::centerMsg msg;
		capture >> frame;
		cv::imshow("cam", frame); // 640 * 480;
		cvtColor(frame, gray_image, CV_BGR2GRAY);
		//imshow("gray", gray_image);

		Rect roi(0, 190, 640, 100);
		gray_image(roi).copyTo(roiImg);

		threshold(roiImg, roiImg, 100, 255, 0);
		bitwise_not(roiImg, roiImg); // negative image
		Mat erodeElmt = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat dilateElmt = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(roiImg, roiImg, erodeElmt);
		dilate(roiImg, roiImg, dilateElmt);

		findContours(roiImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		for (size_t i = 0; i < contours.size(); i++) {
			float area = contourArea(contours[i]);
			if (area > 2000) {
				Moments mu;
				mu = moments(contours[i], false);
				msg.data = mu.m10 / mu.m00;
				Point2f center(mu.m10 / mu.m00, 240); // point in center (x only)
				circle(frame, center, 5, Scalar(0, 255, 0), -1, 8, 0);
				cout << center << endl;
			}
		}


		imshow("cam", frame);
		imshow("roi", roiImg);
		if (cv::waitKey(30) >= 0) break;
		//msg.data = center;
		simple_tracking_pub.publish(msg);
	}

	// VideoCapture automatically deallocate camera object
	return 0;
}