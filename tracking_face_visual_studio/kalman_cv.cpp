#include "kalman_cv.h"

SW_kalman::SW_kalman()
{
	kf.init(stateSize, measSize, contrSize, type);
	cv::Mat state_cy(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
	cv::Mat meas_cy(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
	//cv::Mat procNoise(stateSize, 1, type)
	// [E_x,E_y,E_v_x,E_v_y,E_w,E_h]
	state = state_cy.clone();
	meas = meas_cy.clone();

	// Transition State Matrix A
	// Note: set dT at each processing step!
	// [ 1 0 dT 0  0 0 ]
	// [ 0 1 0  dT 0 0 ]
	// [ 0 0 1  0  0 0 ]
	// [ 0 0 0  1  0 0 ]
	// [ 0 0 0  0  1 0 ]
	// [ 0 0 0  0  0 1 ]
	cv::setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0  ]
	// [ 0    Ey  0     0     0    0  ]
	// [ 0    0   Ev_x  0     0    0  ]
	// [ 0    0   0     Ev_y  0    0  ]
	// [ 0    0   0     0     Ew   0  ]
	// [ 0    0   0     0     0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
	// <<<< Kalman Filter
}
cv::Mat& SW_kalman::operator >>(cv::Mat &frame) // get frame;
{
	*capture >> frame;
	return frame;
}
void SW_kalman::predict_loop(double dT,cv::Mat &res)
{
	// >>>> Matrix A
	kf.transitionMatrix.at<float>(2) = dT;
	kf.transitionMatrix.at<float>(9) = dT;
	// <<<< Matrix A

	cout << "dT:" << endl << dT << endl;

	state = kf.predict();
	cout << "State post:" << endl << state << endl;

	cv::Rect predRect;
	predRect.width = state.at<float>(4);
	predRect.height = state.at<float>(5);
	predRect.x = state.at<float>(0) - predRect.width / 2;
	predRect.y = state.at<float>(1) - predRect.height / 2;

	cv::Point center;
	center.x = state.at<float>(0);
	center.y = state.at<float>(1);
	cv::circle(res, center, 2, CV_RGB(255, 0, 0), -1);
	//cv::rectangle(res, predRect, CV_RGB(255, 0, 0), 2);
}
void SW_kalman::correct_loop(cv::Rect ballsBox, bool found)
{
	cout << "face found" << found << endl;

	meas.at<float>(0) = ballsBox.x + ballsBox.width / 2;
	meas.at<float>(1) = ballsBox.y + ballsBox.height / 2;
	meas.at<float>(2) = (float)ballsBox.width;
	meas.at<float>(3) = (float)ballsBox.height;

	if (!found) // First detection!
	{
		// >>>> Initialization
		kf.errorCovPre.at<float>(0) = 1; // px
		kf.errorCovPre.at<float>(7) = 1; // px
		kf.errorCovPre.at<float>(14) = 1;
		kf.errorCovPre.at<float>(21) = 1;
		kf.errorCovPre.at<float>(28) = 1; // px
		kf.errorCovPre.at<float>(35) = 1; // px

		state.at<float>(0) = meas.at<float>(0);
		state.at<float>(1) = meas.at<float>(1);
		state.at<float>(2) = 0;
		state.at<float>(3) = 0;
		state.at<float>(4) = meas.at<float>(2);
		state.at<float>(5) = meas.at<float>(3);
		// <<<< Initialization

		kf.statePost = state;

		found = true;
	}
	else
		kf.correct(meas); // Kalman Correction

	cout << "Measure matrix:" << endl << meas << endl;
}
SW_kalman::~SW_kalman()
{

}