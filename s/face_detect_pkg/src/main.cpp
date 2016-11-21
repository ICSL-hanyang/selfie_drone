#include "SW_detection.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>
#include "face_detect_pkg/centerMsg.h"

int main(int argc, char **argv)
{
	ros::init(argc,argv, "face_detect_publisher");
	ros::NodeHandle nh;

	cv::VideoCapture camera(0);
	if (!camera.isOpened()) {
		fprintf(stderr, "Error getting camera...\n");
		exit(1);
	}

	//SW_detection detec(camera);
	SW_detection detec(CASCADE_FILE, camera);
	cv::Mat frame;
	double fps = 0, time_per_frame;
	clock_t begin, end;

	ros::Publisher face_detect_pub = nh.advertise<face_detect_pkg::centerMsg>("face_detect_msg",100);
	ros::Rate loop_rate(10);
	
	face_detect_pkg::centerMsg msg;
	while (1)
	{
		auto start = cv::getCPUTickCount();
		//detec.detect_shape(detec.imgproc_shape_detection(detec >> frame));'
		//detec.getFrameAndDetect(detec >> frame);
		detec >> frame;
		detec.detect_face();
		cv::Point temp_pos = detec.facePosition();
		msg.data = detec.distance_from_middle(temp_pos, 150);
		msg.x_pos = temp_pos.x;
		msg.y_pos = temp_pos.y;
	
		cout << "distance from middle : " << msg.data << endl;
		auto end = cv::getCPUTickCount();

		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);

		//circle(frame, detec.shape_position(), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		cv::rectangle(frame, detec.face(), cv::Scalar(255, 0, 0));
		cv::circle(frame, detec.facePosition(), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		msg.wid = detec.face().width;
		msg.hei = detec.face().height;	
		cv::imshow("frame", frame);

		if (cv::waitKey(25) >= 0) break;
	}
	return 0;
}
