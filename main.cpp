#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


/** Function Headers */
void detectByCascade( Mat frame );

void detectByCascadeAndKalmanFilter( Mat frame, map<int,KalmanFilter>& KFm);

void bgSub(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame);

void bgSubAndKalmanFilter(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame, map<int,KalmanFilter>& KFm);

/** Global variables */
String car_cascade_name = "../classifier/cars3.xml";
String input_file, output_file;
CascadeClassifier car_cascade;
string window_name = "Capture - Car detection";
RNG rng(12345);


int getTrackId(int x, int y, map<int,KalmanFilter>& KFm){
    for (map<int,KalmanFilter>::iterator it = KFm.begin(); it!=KFm.end(); ++it){
        if (pow(it->second.statePre.at<float>(0) - x,2) + pow(it->second.statePre.at<float>(1)-y,2) < 20)
            return it->first;
    }
    KalmanFilter tempKF(4,2,0);
    tempKF.statePre.at<float>(0,0) = x; //State x
    tempKF.statePre.at<float>(1,0) = y; //State y
    tempKF.statePost.at<float>(0,0) = x; //State Vx
    tempKF.statePost.at<float>(1,0) = y; //State Vy

    tempKF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1) ;
    setIdentity(tempKF.measurementMatrix, Scalar::all(1));
    setIdentity(tempKF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(tempKF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(tempKF.errorCovPost, Scalar::all(.1));
    KFm[KFm.size()] = tempKF;
    cout << KFm.size() <<endl;
    return KFm.size()-1;
}

//detect and draw bounding box by backgroun subtraction
void bgSub(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame){
    // Mat back;
    Mat fore;
    Mat frame_gray;



    vector<vector<Point> > contours;
    vector<KeyPoint> keypoints;

    namedWindow("Frame");
    // namedWindow("Background");
    cvtColor(frame,frame_gray,CV_BGR2GRAY);
    // imshow("gray", frame_gray);
    bg.operator ()(frame_gray,fore);
    // bg.getBackgroundImage(back);
    
    GaussianBlur( fore, fore, Size( 3, 3), 1,1);
    // threshold(fore,fore,100,255,THRESH_TRUNC);
    // Canny(fore,fore,30,90,3);

    
    // detector.detect(fore,keypoints);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    
    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // drawContours(frame_gray,contours,-1,Scalar(0,0,255),2);
    for (int i = 0; i < contours.size(); i++){
        vector<Point> contours_poly;
        approxPolyDP( contours[i], contours_poly, 3 , true);
        Rect boundRect = boundingRect(contours_poly);
        if (norm(boundRect.tl() - boundRect.br()) < 30)
            continue;
        rectangle(frame, boundRect.tl(), boundRect.br(), Scalar(255,0,255));
    }
    // drawKeypoints( frame, keypoints, frame, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    // for (int i = 0; i < keypoints.size(); i++){
    //     circle(frame,keypoints[i].pt,keypoints[i].size,Scalar(255,0,255));
    // }
    // imshow("Frame",frame);
    return;
}

void bgSubAndKalmanFilter(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame, map<int,KalmanFilter>& KFm){
    // Mat back;
    Mat fore;
    Mat frame_gray;



    vector<vector<Point> > contours;
    vector<KeyPoint> keypoints;

    namedWindow("Frame");
    // namedWindow("Background");
    cvtColor(frame,frame_gray,CV_BGR2GRAY);
    // imshow("gray", frame_gray);
    bg.operator ()(frame_gray,fore);
    // bg.getBackgroundImage(back);
    
    GaussianBlur( fore, fore, Size( 3, 3), 1,1);
    // threshold(fore,fore,100,255,THRESH_TRUNC);
    // Canny(fore,fore,30,90,3);

    
    // detector.detect(fore,keypoints);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    
    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // drawContours(frame_gray,contours,-1,Scalar(0,0,255),2);
    for (int i = 0; i < contours.size(); i++){
        vector<Point> contours_poly;
        approxPolyDP( contours[i], contours_poly, 3 , true);
        Rect boundRect = boundingRect(contours_poly);
        if (norm(boundRect.tl() - boundRect.br()) < 30)
            continue;
        rectangle(frame, boundRect.tl(), boundRect.br(), Scalar(255,0,255));
    }
    // drawKeypoints( frame, keypoints, frame, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    // for (int i = 0; i < keypoints.size(); i++){
    //     circle(frame,keypoints[i].pt,keypoints[i].size,Scalar(255,0,255));
    // }
    // imshow("Frame",frame);
    return;
}


//detect and display bounding box by haar cascade classifier
void detectByCascade( Mat frame )
{
    std::vector<Rect> cars;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    
    //-- Detect cars
    car_cascade.detectMultiScale( frame_gray, cars, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(24,24), Size(64,64));
    
    for( size_t i = 0; i < cars.size(); i++ )
    {
     rectangle(frame, cvPoint(cars[i].x, cars[i].y), cvPoint(cars[i].x+cars[i].width, cars[i].y+cars[i].height), Scalar(255,0,255));
 }
}

void detectByCascadeAndKalmanFilter( Mat frame , map<int,KalmanFilter>& KFm)
{
    std::vector<Rect> cars;
    Mat frame_gray, hsv, hue;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    
    //-- Detect cars
    car_cascade.detectMultiScale( frame_gray, cars, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(24,24), Size(64,64));
    

    for( size_t i = 0; i < cars.size(); i++ )
    {
        //int id = getTrackId(cars[i].x+cars[i].width/2, cars[i].y+cars[i].height/2, KFm);
        // Mat_<float> measurement(1,2);
        // measurement.at<float>(0) = cars[i].x+cars[i].width/2;
        // measurement.at<float>(1) = cars[i].y+cars[i].height/2;
        // measurement.at<float>(2) = cars[i].x+cars[i].width/2;
        // measurement.at<float>(3) = cars[i].y+cars[i].height/2;
        //Mat_<float> measurement = Mat_<float>::zeros(2,1);
        //measurement.at<float>(0, 0) = cars[i].x+cars[i].width/2;
        //measurement.at<float>(1, 0) = cars[i].y+cars[i].height/2;
        //cout << "current " << KFm[id].statePre.at<float>(0) << " " << KFm[id].statePre.at<float>(1) << endl;
        //Mat prediction = KFm[id].predict();
        //cout << "predict "<< prediction.size().width << " " << prediction.size().height <<endl;
        //cout << "measurement "<< measurement.size().width << " " << measurement.size().height <<endl;
        //cout << "measurementMatrix " << KFm[i].measurementMatrix.size().width << " " << KFm[i].measurementMatrix.size().height <<endl;
        //cout << "errorCovPre" << KFm[i].errorCovPre.size().width << " " << KFm[i].errorCovPre.size().height <<endl;
        //cout << "measurement noise " << KFm[i].measurementNoiseCov.size().width << " " << KFm[i].measurementNoiseCov.size().height <<endl;
        //cout << "statePre " << KFm[i].statePre.size().width << " " << KFm[i].statePre.size().height <<endl;
        //cout << "predict " << prediction.at<float>(0) << " " << prediction.at<float>(1)<<endl;
        //circle(frame, cvPoint(prediction.at<float>(0), prediction.at<float>(1)), 10,Scalar(255, 255, 0));
        //KFm[id].correct(measurement);
        rectangle(frame, cvPoint(cars[i].x, cars[i].y), cvPoint(cars[i].x+cars[i].width, cars[i].y+cars[i].height), Scalar(255,0,255));
        //circle(frame, cvPoint(KFm[id].statePre.at<float>(0), KFm[id].statePre.at<float>(1)), 10,Scalar(255, 0, 0));
    }


}


/** @function main */
int main( int argc, const char** argv )
{
    BackgroundSubtractorMOG2 bg(30,16);
    bg.set("nmixtures",3);
    SimpleBlobDetector::Params params;
    params.filterByColor = true;
    params.blobColor = 0;
    // params.filterByArea = true;
    // params.minArea = 20;
    // params.maxArea = 200;
    // params.minDistBetweenBlobs = 
    SimpleBlobDetector detector(params);
    // SimpleBlobDetector detector;

    map<int,KalmanFilter> KFm;
    KFm.clear();
    


    Mat frame;
    if (argc > 1)
    {
        input_file = argv[1];
        output_file = argv[2];
    }
    if( !car_cascade.load( car_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    VideoCapture capture(input_file);
    VideoWriter output(output_file, CV_FOURCC('F','M','P','4'), 14.999, Size(320,240));
    if( capture.isOpened() && output.isOpened())
    {
        while( true ){
            capture >> frame;
            if (!frame.empty()) {
                // bgSub(bg, detector,frame);
                // detectByCascade(frame);
                detectByCascadeAndKalmanFilter(frame, KFm);
                imshow(window_name,frame);
                output << frame;
                int c = waitKey(1);
                if( (char)c == 'c' ) { return 0; }
            }
            else {
                cout << "empty frame\n";
                exit(0);
            }
        }
    }
    else {
        cout << "not opened\n";
    }
    return 0;
}
