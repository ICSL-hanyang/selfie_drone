#include "SW_detection.h"
#include "kalman_cv.h"

int main()
{
	cv::VideoCapture camera(0);
	if (!camera.isOpened()) {
		fprintf(stderr, "Error getting camera...\n");
		exit(1);
	}

	//SW_detection detec(camera);
	SW_kalman kalman_face;
	SW_detection detec(CASCADE_FILE, camera);
	cv::Mat frame;
	double fps = 0, time_per_frame, ticks;
	clock_t end;
	ticks = 0;
	
	while (1)
	{
		//auto start = cv::getCPUTickCount();
		double precTick = ticks;
		ticks = (double)cv::getTickCount();
		double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
		//detec.detect_shape(detec.imgproc_shape_detection(detec >> frame));'
		//detec.getFrameAndDetect(detec >> frame);
		detec >> frame;
		if (detec.get_face_found())
		{
			kalman_face.predict_loop(dT, frame);
		}
		detec.detect_face();
		cout << "distance from middle : " << detec.distance_from_middle(detec.facePosition(), 150) << endl;
		auto end = cv::getCPUTickCount();

		time_per_frame = (end - ticks) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);

		//circle(frame, detec.shape_position(), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		cv::rectangle(frame, detec.face(), cv::Scalar(255, 0, 0));
		cv::circle(frame, detec.facePosition(), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		kalman_face.correct_loop( detec.face(), detec.get_face_found());
		cv::imshow("frame", frame);

		if (cv::waitKey(25) >= 0) break;
	}
	return 0;
}

