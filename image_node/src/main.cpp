#include "SW_detection.h"
#include <stdio.h>
#include <raspicam/raspicam_cv.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/ros.h>
#include "image_node/centerMsg.h"

int main(int argc, char **argv)
{
	ros::init(argc,argv, "image_node_publisher");
	ros::NodeHandle nh;

    raspicam::RaspiCam_Cv Camera;
    cv::Mat image;

    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3);
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 480);


    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}


	SW_detection detec(CASCADE_FILE);
	//cv::Mat frame;
	double fps = 0, time_per_frame;
	clock_t begin, end;	//cv::Point detect_shape();

	ros::Publisher image_node_pub = nh.advertise<image_node::centerMsg>("image_node_msg",100);
	ros::Rate loop_rate(10);

	
	while (1)
	{
		image_node::centerMsg msg;
		auto start = cv::getCPUTickCount();
		//detec.detect_shape(detec.imgproc_shape_detection(detec >> frame));'
		//detec.getFrameAndDetect(detec >> frame);
		  Camera.grab();
        Camera.retrieve (image);

		detec.frame_input(image);
		detec.detect_face();
		cv::Point temp_pos = detec.facePosition();
		msg.data = detec.distance_from_middle(temp_pos, 150);
		msg.x_pos = temp_pos.x;
		msg.y_pos = temp_pos.y;

		cout << "distance from middle : " << msg.data << " cm " <<  endl;
		auto end = cv::getCPUTickCount();

		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);

		//circle(frame, detec.shape_position(), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		cv::rectangle(image, detec.face(), cv::Scalar(255, 0, 0));
		cv::circle(image, detec.facePosition(), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		msg.wid = detec.face().width;
		msg.hei = detec.face().height;
		cv::imshow("frame", image);

		if (cv::waitKey(25) >= 0) break;
	}
	return 0;
}
