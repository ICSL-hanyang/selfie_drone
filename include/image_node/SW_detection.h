#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;

const cv::String    CASCADE_FILE("/home/raspberry/haarcascade_frontalface_default.xml");

const float focal_length = 530;
//float actual_distance = 70; // actual distance from camera (user should measuer it by himself) (cm)
//float actual_width = 25.2; // actual width of object (user should measure too)

class SW_detection
{
private:
	static const double TICK_FREQUENCY;
	bool   m_templateMatchingRunning = false;
	bool   face_found = false;
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


	cv::CascadeClassifier* 	faceCascade = NULL; // for facetracking (.xml file)


	vector< cv::Rect > 		m_allFaces; //

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
	SW_detection(const std::string cascadeFilePath);

	cv::Mat imgproc_face_detection();

	cv::Point   detect_face();
	cv::Point shape_position() const;

	void 		frame_input(cv::Mat &image);

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

	~SW_detection();
};
