#include <opencv2/core/core.hpp> // Module "highgui"
#include <opencv2/highgui/highgui.hpp> // Module "imgproc"
#include <opencv2/imgproc/imgproc.hpp> // Module "video"
#include <opencv2/video/video.hpp> // Output
#include <iostream> // Vector
#include <vector>

using namespace std;
// >>>>> Color to be tracked
#define MIN_H_BLUE 200
#define MAX_H_BLUE 300
#define FRAME_WIDTH  
#define FRAME_HEIGHT 

//#define stateSize  6
//#define measSize   4
//#define contrSize  0
//#define type  CV_32F
// <<<<< Color to be tracked

class SW_kalman
{
private:
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;
	unsigned int type = CV_32F;

	cv::KalmanFilter kf;
	cv::Mat state;  // [x,y,v_x,v_y,w,h]
	cv::Mat meas;    // [z_x,z_y,z_w,z_h]
	//cv::Mat procNoise(stateSize, 1, type)
	// [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

	// Camera Index
	int idx = 0;
	// Camera Capture
	cv::VideoCapture* capture = NULL;
	//char ch = 0;

	//double ticks;
	//double precTick, dT;
	//bool found;
	int notFoundCount;

	cv::Mat frame;
	cv::Mat res;
public:
	SW_kalman();
	cv::Mat&	operator>>(cv::Mat &frame); // get frame;
	void		predict_loop(double dT,cv::Mat &frame);
	void		correct_loop(cv::Rect ballsBox, bool found); //thread separation needed - later
	bool		get_found();
	~SW_kalman();
};