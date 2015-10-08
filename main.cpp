#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <alpr.h>
#include <unistd.h>


#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


/** Function Headers */
std::vector<Rect> detectByCascade( Mat frame );

void detectByCascadeAndKalmanFilter( Mat frame, map<int,KalmanFilter>& KFm);

void bgSub(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame);

void bgSubAndKalmanFilter(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame, map<int,KalmanFilter>& KFm);

/** Global variables */
String car_cascade_name = "../classifier/lbpcars1.xml";
String input_file, output_file;
CascadeClassifier car_cascade;
string window_name = "Capture - Car detection";
RNG rng(12345);

int trackWithCamshift(Rect ROI_rec, Mat &Image) {
    Mat backproj, ROI_hsv, hist;
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    Mat ROI(Image, ROI_rec);

    cvtColor(ROI, ROI_hsv, CV_BGR2HSV);
    int channels[] = {0, 1};
    calcHist(&ROI_hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    calcBackProject(&Image, 1, channels, hist, backproj, ranges);
    RotatedRect trackBox = CamShift(backproj, ROI_rec, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    imshow("backproj", backproj);
    return 1;
}

int openalprDemo() {
    // Initialize the library using United States style license plates.  
    // You can use other countries/regions as well (for example: "eu", "au", or "kr")
    alpr::Alpr openalpr("sg", "~/openalpr/config/openalpr.conf.in", "~/openalpr/runtime_data");

    // Optionally specify the top N possible plates to return (with confidences).  Default is 10
    openalpr.setTopN(20);

    // Optionally, provide the library with a region for pattern matching.  This improves accuracy by 
    // comparing the plate text with the regional pattern.
    openalpr.setDefaultRegion("sg");

    // Make sure the library loaded before continuing.  
    // For example, it could fail if the config/runtime_data is not found
    if (openalpr.isLoaded() == false)
    {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return 1;
    }

    // Recognize an image file.  You could alternatively provide the image bytes in-memory.
    alpr::AlprResults results = openalpr.recognize("/path/to/image.jpg");

    // Iterate through the results.  There may be multiple plates in an image, 
    // and each plate return sthe top N candidates.
    for (int i = 0; i < results.plates.size(); i++)
    {
      alpr::AlprPlateResult plate = results.plates[i];
      std::cout << "plate" << i << ": " << plate.topNPlates.size() << " results" << std::endl;

        for (int k = 0; k < plate.topNPlates.size(); k++)
        {
          alpr::AlprPlate candidate = plate.topNPlates[k];
          std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
          std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
        }
    }
}

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
std::vector<Rect> detectByCascade( Mat frame )
{
    std::vector<Rect> cars;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    
    //-- Detect cars
    car_cascade.detectMultiScale( frame_gray, cars, 1.1, 6,0, Size(200,200), Size(300,300));
    
    // for( size_t i = 0; i < cars.size(); i++ )
    // {
    //     // rectangle(frame, cars[i], Scalar(255,0,255));
    //     circle (frame, (cars[i].tl() + cars[i].br())*0.5, 5, Scalar(255,0,255));
    // }
    return cars;
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
        int id = getTrackId(cars[i].x+cars[i].width/2, cars[i].y+cars[i].height/2, KFm);
        Mat_<float> measurement(1,2);
        measurement.at<float>(0) = cars[i].x+cars[i].width/2;
        measurement.at<float>(1) = cars[i].y+cars[i].height/2;
        measurement.at<float>(2) = cars[i].x+cars[i].width/2;
        measurement.at<float>(3) = cars[i].y+cars[i].height/2;
        // measurement = Mat_<float>::zeros(2,1);
        // measurement.at<float>(0, 0) = cars[i].x+cars[i].width/2;
        // measurement.at<float>(1, 0) = cars[i].y+cars[i].height/2;
        cout << "current " << KFm[id].statePre.at<float>(0) << " " << KFm[id].statePre.at<float>(1) << endl;
        Mat prediction = KFm[id].predict();
        cout << "predict "<< prediction.size().width << " " << prediction.size().height <<endl;
        cout << "measurement "<< measurement.size().width << " " << measurement.size().height <<endl;
        cout << "measurementMatrix " << KFm[i].measurementMatrix.size().width << " " << KFm[i].measurementMatrix.size().height <<endl;
        cout << "errorCovPre" << KFm[i].errorCovPre.size().width << " " << KFm[i].errorCovPre.size().height <<endl;
        cout << "measurement noise " << KFm[i].measurementNoiseCov.size().width << " " << KFm[i].measurementNoiseCov.size().height <<endl;
        cout << "statePre " << KFm[i].statePre.size().width << " " << KFm[i].statePre.size().height <<endl;
        cout << "predict " << prediction.at<float>(0) << " " << prediction.at<float>(1)<<endl;
        circle(frame, cvPoint(prediction.at<float>(0), prediction.at<float>(1)), 10,Scalar(255, 255, 0));
        KFm[id].correct(measurement);
        rectangle(frame, cvPoint(cars[i].x, cars[i].y), cvPoint(cars[i].x+cars[i].width, cars[i].y+cars[i].height), Scalar(255,0,255));
        circle(frame, cvPoint(KFm[id].statePre.at<float>(0), KFm[id].statePre.at<float>(1)), 10,Scalar(255, 0, 0));
    }


}


/** @function main */
int main( int argc, const char** argv )
{
    // Set up background subtraction
    // BackgroundSubtractorMOG2 bg(30,16);
    // bg.set("nmixtures",3);
    // SimpleBlobDetector::Params params;
    // params.filterByColor = true;
    // params.blobColor = 0;
    // params.filterByArea = true;
    // params.minArea = 20;
    // params.maxArea = 200;
    // params.minDistBetweenBlobs = 
    // SimpleBlobDetector detector(params);
    // SimpleBlobDetector detector;

    // Set up Kalman filter
    // map<int,KalmanFilter> KFm;
    // KFm.clear();
    std::vector<Rect> cars;
    
    Mat backproj, ROI_hsv, hist;
    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    int channels[] = {0, 1};

    Mat frame;
    if (argc > 1)
    {
        input_file = argv[1];
        output_file = argv[2];
    }
    if( !car_cascade.load( car_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    VideoCapture capture(input_file);
    VideoWriter output(output_file, CV_FOURCC('F','M','P','4'), 14.999, Size(320,240));
    double scaleUp = 1;
    double scaleDown = 1;
    if( capture.isOpened() && output.isOpened())
    {
        while( true ){
            capture >> frame;
            // float scale_w = frame.cols/160;
            // float scale_h = frame.rows/120;
            if (!frame.empty()) {
                // if (!cars.empty()) {
                //     for (int i = 0; i < cars.size(); i++){
                //         calcBackProject(&frame, 1, channels, hist, backproj, ranges);
                //         RotatedRect trackBox = CamShift(backproj, cars[i], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 0.1 ));
                //         rectangle(frame, trackBox.boundingRect(),  Scalar(255,0,255));
                //         imshow("backproj", backproj);
                //     }
                // }
                // else {
                    // bgSub(bg, detector,frame);
                    Mat canvas;                    
                    resize(frame, canvas, Size(), scaleDown, scaleDown);
                    cars = detectByCascade(canvas);
                    // detectByCascadeAndKalmanFilter(frame, KFm);
                    for (int i = 0; i < cars.size(); i++){
                        rectangle(frame, cars[i].tl()*scaleUp, cars[i].br()*scaleUp, Scalar(255,0,255), 3);
                            // rectangle(canvas, cars[i], Scalar(255,0,255));
                        // Mat ROI(frame, cars[i]);
                        // imshow("ROI", ROI);
                        // cvtColor(ROI, ROI_hsv, CV_BGR2HSV);
                        // calcHist(&ROI_hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
                        // calcBackProject(&frame, 1, channels, hist, backproj, ranges);
                        // RotatedRect trackBox = CamShift(backproj, cars[i], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 0.1 ));
                    }
                // }
                imshow(window_name,frame);
                output << canvas;
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

