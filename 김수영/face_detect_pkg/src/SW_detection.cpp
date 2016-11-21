#include "SW_detection.h"

const double SW_detection::TICK_FREQUENCY = cv::getTickFrequency();

SW_detection::SW_detection(const std::string cascadeFilePath, cv::VideoCapture &videoCapture)
{
	capture = &videoCapture;
	if (faceCascade == NULL) {
		faceCascade = new cv::CascadeClassifier(cascadeFilePath);
	}
	else {
		faceCascade->load(cascadeFilePath);
	}

	if (faceCascade->empty()) {
		std::cerr << "Error creating cascade classifier. Make sure the file" << std::endl
			<< cascadeFilePath << " exists." << std::endl;
	}
}
SW_detection::SW_detection(cv::VideoCapture &videoCapture)
{
	capture = &videoCapture;
	cout << "type actual width of object : ";
	cin >> actual_width;
}
cv::Mat& SW_detection::operator >>(cv::Mat &frame) // get frame;
{
	*capture >> frame;
	mid.x = frame.size().width / 2;
	mid.y = frame.size().height / 2;
	//frame.copyTo(face_frame);
	face_frame = frame.clone();
	return frame;
}
cv::Point SW_detection::detect_face()
{
	m_scale = (double)std::min(320, face_frame.cols) / face_frame.cols;
	cv::Size resizedFrameSize = cv::Size((int)(m_scale*face_frame.cols), (int)(m_scale*face_frame.rows));

	cv::Mat resizedFrame;
	cv::resize(face_frame, resizedFrame, resizedFrameSize);

	if (!face_found){
		detectFaceAllSizes(resizedFrame); // Detect using cascades over whole image
	}
	else {
		detectFaceAroundRoi(resizedFrame); // Detect using cascades only in ROI face_found == true
		if (m_templateMatchingRunning) {
			detectFacesTemplateMatching(resizedFrame); // Detect using template matching
		}
	}
	return m_facePosition;
}
cv::Mat SW_detection::imgproc_shape_detection()
{
	cvtColor(face_frame, gray_image, CV_BGR2GRAY);
	Canny(gray_image, bw, 50, 240, 3); // meaning of third and fourth value :  / 
	cv::imshow("gray", bw);
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	face_frame.copyTo(dst);
	return bw;
}
cv::Mat SW_detection::imgproc_face_detection()
{
	// Downscale frame to m_resizedWidth width - keep aspect ratio
	m_scale = (double)min(m_resizedWidth, face_frame.cols) / face_frame.cols;
	cv::Size resizedFrameSize = cv::Size((int)(m_scale*face_frame.cols), (int)(m_scale*face_frame.rows));
	cv::resize(face_frame, resizedFrame, resizedFrameSize);
	return resizedFrame;
}
void SW_detection::detectFaceAllSizes(const cv::Mat &frame) // frme : resized frame
{
	// Minimum face size is 1/5th of screen height
	// Maximum face size is 2/3rds of screen height
	faceCascade->detectMultiScale(frame, m_allFaces, 1.1, 3, 0,
		cv::Size(frame.rows / 5, frame.rows / 5),
		cv::Size(frame.rows * 2 / 3, frame.rows * 2 / 3)); // in m_allFaces, there are rectangles

	if (m_allFaces.empty()) return; // std::vector<cv::Rect>   m_allFaces;

	face_found = true;

	// Locate biggest face
	m_trackedFace = biggestFace(m_allFaces);

	// Copy face template
	m_faceTemplate = getFaceTemplate(frame, m_trackedFace); // getFaceTemplate : 

	// Calculate roi
	m_faceRoi = doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

	// Update face position
	m_facePosition = centerOfRect(m_trackedFace);
	cout << "first face detect success" << endl;
}
void SW_detection::detectFaceAroundRoi(const cv::Mat &frame)
{
	// Detect faces sized +/-20% off biggest face in previous search
	faceCascade->detectMultiScale(frame(m_faceRoi), m_allFaces, 1.1, 3, 0,
		cv::Size(m_trackedFace.width * 8 / 10, m_trackedFace.height * 8 / 10),
		cv::Size(m_trackedFace.width * 12 / 10, m_trackedFace.width * 12 / 10));

	if (m_allFaces.empty())
	{
		// Activate template matching if not already started and start timer
		m_templateMatchingRunning = true;
		if (m_templateMatchingStartTime == 0)
			m_templateMatchingStartTime = cv::getTickCount();
		return;
	}

	// Turn off template matching if running and reset timer
	m_templateMatchingRunning = false;
	m_templateMatchingCurrentTime = m_templateMatchingStartTime = 0;

	// Get detected face
	m_trackedFace = biggestFace(m_allFaces);

	// Add roi offset to face
	m_trackedFace.x += m_faceRoi.x;
	m_trackedFace.y += m_faceRoi.y;

	// Get face template
	m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

	// Calculate roi
	m_faceRoi = doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

	// Update face position
	m_facePosition = centerOfRect(m_trackedFace);
}
cv::Mat     SW_detection::getFaceTemplate(const cv::Mat &frame, cv::Rect face)
{
	face.x += face.width / 4;
	face.y += face.height / 4;
	face.width /= 2;
	face.height /= 2;

	cv::Mat faceTemplate = frame(face).clone();
	return faceTemplate;
}
void SW_detection::detectFacesTemplateMatching(const cv::Mat &frame)
{
	// Calculate duration of template matching
	m_templateMatchingCurrentTime = cv::getTickCount();
	double duration = (double)(m_templateMatchingCurrentTime - m_templateMatchingStartTime) / TICK_FREQUENCY;

	// If template matching lasts for more than 2 seconds face is possibly lost
	// so disable it and redetect using cascades
	if (duration > m_templateMatchingMaxDuration) {
		face_found = false;
		m_templateMatchingRunning = false;
		m_templateMatchingStartTime = m_templateMatchingCurrentTime = 0;
	}

	// Template matching with last known face 
	//cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_CCOEFF);
	cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_SQDIFF_NORMED);
	cv::normalize(m_matchingResult, m_matchingResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
	double min, max;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc);

	// Add roi offset to face position
	minLoc.x += m_faceRoi.x;
	minLoc.y += m_faceRoi.y;

	// Get detected face
	//m_trackedFace = cv::Rect(maxLoc.x, maxLoc.y, m_trackedFace.width, m_trackedFace.height);
	m_trackedFace = cv::Rect(minLoc.x, minLoc.y, m_faceTemplate.cols, m_faceTemplate.rows);
	m_trackedFace = doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

	// Get new face template
	m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

	// Calculate face roi
	m_faceRoi = doubleRectSize(m_trackedFace, cv::Rect(0, 0, frame.cols, frame.rows));

	// Update face position
	m_facePosition = centerOfRect(m_trackedFace);
}
cv::Rect SW_detection::biggestFace(std::vector<cv::Rect> &faces) const
{
	assert(!faces.empty());

	cv::Rect *biggest = &faces[0];
	for (auto &face : faces) {
		if (face.area() < biggest->area())
			biggest = &face;
	}
	return *biggest;
}
cv::Point SW_detection::centerOfRect(const cv::Rect &rect) const
{
	return cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);

}
cv::Rect SW_detection::doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const
{
	cv::Rect outputRect;
	// Double rect size
	outputRect.width = inputRect.width * 2;
	outputRect.height = inputRect.height * 2;

	// Center rect around original center
	outputRect.x = inputRect.x - inputRect.width / 2;
	outputRect.y = inputRect.y - inputRect.height / 2;

	// Handle edge cases
	if (outputRect.x < frameSize.x) {
		outputRect.width += outputRect.x;
		outputRect.x = frameSize.x;
	}
	if (outputRect.y < frameSize.y) {
		outputRect.height += outputRect.y;
		outputRect.y = frameSize.y;
	}

	if (outputRect.x + outputRect.width > frameSize.width) {
		outputRect.width = frameSize.width - outputRect.x;
	}
	if (outputRect.y + outputRect.height > frameSize.height) {
		outputRect.height = frameSize.height - outputRect.y;
	}

	return outputRect;
}
void SW_detection::setResizedWidth(const int width)
{
	m_resizedWidth = std::max(width, 1);
}
int SW_detection::resizedWidth() const
{
	return m_resizedWidth;
}
cv::Rect SW_detection::face() const
{
	cv::Rect faceRect = m_trackedFace;
	faceRect.x = (int)(faceRect.x / m_scale);
	faceRect.y = (int)(faceRect.y / m_scale);
	faceRect.width = (int)(faceRect.width / m_scale);
	faceRect.height = (int)(faceRect.height / m_scale);
	return faceRect;
}
cv::Point SW_detection::facePosition() const
{
	cv::Point facePos;
	facePos.x = (int)(m_facePosition.x / m_scale);
	facePos.y = (int)(m_facePosition.y / m_scale);
	return facePos;
}
cv::Point SW_detection::mid_point() const
{
	return mid;
}
void SW_detection::setTemplateMatchingMaxDuration(const double s)
{
	m_templateMatchingMaxDuration = s;
}
double SW_detection::templateMatchingMaxDuration() const
{
	return m_templateMatchingMaxDuration;
}
///////////////////////from this shape detection /////////////////////////////////////
cv::Point SW_detection::detect_shape()
{
	for (int i = 0; i < contours.size(); i++){
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);
		if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
			continue;
		if (approx.size() >= 4 && approx.size() <= 6){ // else if
			int vtc = approx.size();
			if (vtc == 4) // rectangular
			{
				cv::Rect r = cv::boundingRect(contours[i]);
				cv::Mat roi = dst(r);
				cv::imshow("roi image ", roi);
				center.x = r.x + (r.width) / 2;
				center.y = r.y + (r.height) / 2;
				//circle(dst, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
				cout << "RECTANGLE width / length / distance : " << r.width << " , " << r.height << " ] " << endl;
				return center;
			}
		}
		else // circle
		{
			double area = cv::contourArea(contours[i]);
			cv::Rect r = cv::boundingRect(contours[i]);
			int radius = r.width / 2;
			if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
				std::abs(1 - (area / (CV_PI * (radius*radius)))) <= 0.2){
				cv::Rect r = cv::boundingRect(contours[i]);
				cv::Mat roi = dst(r);
				cv::imshow("roi image ", roi);
				center.x = r.x + (r.width) / 2;
				center.y = r.y + (r.height) / 2;
				//circle(dst, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
				cout << "CIRCLE width and lenght : " << r.width << " , " << r.height << " ] " << endl;
				return center;
			}
		}
	}
}
cv::Point SW_detection::shape_position() const
{
	return center;
}
void SW_detection::calc_focal_length(int P, float W, float D)
{
	//focal_length = (P*D) / W;
	//std::cout << "Focal Length is " << focal_length << endl;
}
double SW_detection::angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
float SW_detection::distance_from_middle(cv::Point ob,float actual_distance)
{
	int x = mid.x - ob.x;
	int y = mid.y - ob.y;
	float norm = sqrt(pow(x, 2) + pow(y, 2));
	float W = (norm * actual_distance) / focal_length;
	return W;
}
SW_detection::~SW_detection()
{
	capture->release();
	if (faceCascade != NULL) {
		delete faceCascade;
	}
}