#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <raspicam/raspicam_cv.h>

using namespace std;
using namespace cv;

/** @function main */
int main()
{
    Mat src, src_gray;

    float focal_length;
    float w_distance = 6.7;
    float distance = 50;
    int radius;
    float result;

    raspicam::RaspiCam_Cv Camera;

    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3);
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 480);


    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}


/*
    VideoCapture cap(0); // open the default camera

    if(!cap.isOpened())  // check if we succeeded
        return -1;
*/

    while(1)
    {
        //cap >> src;
        Camera.grab();
        Camera.retrieve (src);

        cvtColor( src, src_gray, CV_BGR2GRAY );

        /// Reduce the noise so we avoid false circle detection
        GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

        vector<Vec3f> circles;

        /// Apply the Hough Transform to find the circles
        HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

        /// Draw the circles detected
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            radius = cvRound(circles[i][2]);
            // circle center
            circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
            cout << " radius : " << radius << endl;
        }
        imshow( "Hough Circle Transform Demo", src );

        focal_length = ( ( radius * distance ) / w_distance );
        cout << "focal_length : " << focal_length << endl;
        //focal_length += 20;
        result = ( (radius * distance) / focal_length ) ;
        cout << "result : " << result << endl;

        if(waitKey(30) >= 0) break;
    }
    cout << src.rows << " , " << src.cols << endl;
    return 0;
}
