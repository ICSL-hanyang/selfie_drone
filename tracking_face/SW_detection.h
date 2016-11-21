#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;

const cv::String    CASCADE_FILE("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
//const cv::String    CASCADE_FILE("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_upperbody.xml");

const float focal_length = 700; 
//float actual_distance = 70; // actual distance from camera (user should measuer it by himself) (cm)
//float actual_width = 25.2; // actual width of object (user should measure too)

class SW_detection
{
private:
	static const double TICK_FREQUENCY;
	bool   m_templateMatchingRunning = false;
	bool   face_found = false;
	bool   face_kfound = false;
	int    m_resizedWidth = 320; // 
	int64  m_templateMatchingStartTime = 0;
	int64  m_templateMatchingCurrentTime = 0;
	double m_scale;
	double m_templateMatchingMaxDuration = 3;
	//float  actual_distance = 150; // actual distance from camera (user should measuer it by himself) (cm)
	float  actual_width; // actual width of object (user should measure too)

	cv::Mat 				frame, gray_image, bw; //gotten frame from webcam / grayscale image / processed image by canny edge detector
	cv::Mat					face_frame; // frame which for face detection 
	cv::Mat 				roiImg, dst; // from contouring and shape detecting could get roi(this could be rectangle or circle) / image which copy of original image and finally presented to user
	cv::Mat 				resizedFrame;
	cv::Mat                 m_faceTemplate;
	cv::Mat                 m_matchingResult;

	cv::VideoCapture* 		capture = NULL; 
	cv::CascadeClassifier* 	faceCascade = NULL; // for facetracking (.xml file)	

	vector<vector<cv::Point>> contours; // 
	vector< cv::Rect > 		m_allFaces; // 
	vector< cv::Vec4i > 	hierarchy; // hierarchy between contours 
	vector<cv::Point> 		approx; // approx.size() means the vertecies of detected shape

	cv::Point 				mid; // center of image (image.width / 2 , image.height / 2)
	cv::Point 				center; // center of detected shape
	cv::Point               m_facePosition; // face position in downgraded scale
	cv::Point 				facePos; // center of trakced face 
	cv::Rect                m_trackedFace;
	cv::Rect                m_faceRoi;

	cv::Rect    doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const; 
	cv::Rect    biggestFace(std::vector<cv::Rect> &faces) const;
	cv::Point   centerOfRect(const cv::Rect &rect) const;
	cv::Mat     getFaceTemplate(const cv::Mat &frame, cv::Rect face);
	void        detectFaceAllSizes(const cv::Mat &frame);
	void        detectFaceAroundRoi(const cv::Mat &frame);
	void        detectFacesTemplateMatching(const cv::Mat &frame);
public:
	SW_detection(const std::string cascadeFilePath, cv::VideoCapture &videoCapture);
	SW_detection(cv::VideoCapture &videoCapture);
	cv::Mat&	operator>>(cv::Mat &frame); // get frame;
	cv::Mat imgproc_shape_detection();
	cv::Mat imgproc_face_detection();
	cv::Point detect_shape();
	cv::Point   detect_face();
	cv::Point shape_position() const;

	void 		calc_focal_length(int P, float W, float D);
	double 		angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);
	float 		distance_from_middle(cv::Point ob, float actual_distance);

	void 					setResizedWidth(const int width);
	int                     resizedWidth() const;
	cv::Rect                face() const;
	cv::Point               facePosition() const;
	cv::Point				mid_point() const;
	void                    setTemplateMatchingMaxDuration(const double s);
	double                  templateMatchingMaxDuration() const;
	bool					get_face_found();
	~SW_detection();
};