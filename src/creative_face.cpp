// Denis Kotnik, november 2020
// https://www.kotnik.si
// Code not clean.

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "opencv2/objdetect.hpp"
#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>
#include <time.h>

using namespace std;
using namespace cv;

#define WINDOW_NAME "Creative Face"

// Config
int secondsOfShowingImage = 3;
// Face detection area multiplicator
double faceFactor = 0.4;
int canvasWidth = 768;
int canvasHeight = 768;
int nImagesInRow = 2;
int nImagesInColumn = 2;
// Has impact on performance
int imageCaptureWidth = 1024;
int imageCaptureHeight = 768;

CascadeClassifier cascade;
RNG rng(1);


// Function for effects
Mat warhol(Mat src, int K, int blurSize) {
	Mat img, bestLabels, centers, clustered;
	Vec3b *colors = new Vec3b[K];

	// Blur the image
	blur(src, img, Size(blurSize, blurSize));

	// Create samples
	Mat samples = img.reshape(1, img.rows * img.cols);
	samples.convertTo(samples, CV_32F);

	// K-Means clustering
	kmeans(samples, K, bestLabels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	// Select a random palette
	for(int i = 0; i < K; i ++) {
		Vec3b color = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		colors[i] = color;
	}

	// Draw the new image
	clustered = Mat(img.size(), img.type());
	for(int i = 0; i < img.cols * img.rows; i++) {
		clustered.at<Vec3b>(i / img.cols, i % img.cols) = colors[bestLabels.at<int>(0, i)];
	}

  return clustered;
}

// Make panel from multiple images (rows * cols)
Mat makePanel(Mat src, int rows, int cols) {
	Mat dst = Mat(src.rows * rows, src.cols * cols, src.type());
	Mat trgt;
	for(int i = 0; i < rows; i ++) {
		for (int j = 0; j < cols; j++) {
			warhol(src, 8, 5).copyTo(dst(Rect(j * src.cols, i * src.rows, src.cols, src.rows)));
		}
	}
	return dst;
}

int main(int argc, char** argv ) {
	CommandLineParser parser(argc, argv,
		"{width          |     | Width of the canvas (default: 768).   }"
		"{height         |     | Height of the canvas (default: 768).   }"
		"{cameraWidth    |     | Width of image captured by camera (default: 1024). }"
		"{cameraHeight   |     | Height of image captured by camera (default: 768). }"
		"{col            |     | Number of filtered images in colum (default: 2). }"
		"{row            |     | Number of filtered images in row (default: 2). }"
		"{time           |     | Time in seconds between the images (default: 3). }"
		"{faceFactor     |     | Face detection area multiplicator (default: 0.4). }"
	);

	parser.about("\nAuthor: Denis Kotnik\n");
	parser.printMessage();

	if (parser.has("width")) {
		canvasWidth = parser.get<int>("width");
	}
	if (parser.has("height")) {
		canvasHeight = parser.get<int>("height");
	}
	if (parser.has("cameraWidth")) {
		imageCaptureWidth = parser.get<int>("cameraWidth");
	}
	if (parser.has("cameraHeight")) {
		imageCaptureHeight = parser.get<int>("cameraHeight");
	}
	if (parser.has("col")) {
		nImagesInRow = parser.get<int>("col");
	}
	if (parser.has("row")) {
		nImagesInColumn = parser.get<int>("row");
	}
	if (parser.has("time")) {
		secondsOfShowingImage = parser.get<int>("time");
	}
	if (parser.has("faceFactor")) {
		faceFactor = parser.get<double>("faceFactor");
	}

	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////

	// Show blank image at the beginning
	Mat logo = imread("./logo.png");
	if (!logo.empty()) {
		resize(logo, logo, Size(canvasWidth, canvasHeight));
		imshow(WINDOW_NAME, logo);
	}

	// Load data with face features info
	if(!cascade.load("./face_cascade.xml")) {
		cerr << "ERROR: can not find face cascade file!" << endl;
		return -1;
	}

	VideoCapture camera(0); // Camera ID

	if (!camera.isOpened()) {
		cerr << "ERROR: no camera detected!" << endl;
		return -1;
	}

	// Set height and width of image capture from camera
	camera.set(CAP_PROP_FRAME_WIDTH, imageCaptureWidth);
	camera.set(CAP_PROP_FRAME_HEIGHT, imageCaptureHeight);

	Mat frame, gray, crop, imageOfFace;
	vector<Rect> faces;

	////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////
	
	while (true) {
		// Hack/workaround for Raspberry Pi / Linux OS, because there is always 5 images in the buffer of the camera.
		for (int i = 0; i < 5; i++) {
			camera.read(frame);
			frame.copyTo(crop);
		}

		// Convert to black and white
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		double t = 0;
		t = (double)getTickCount();

		// Clear vector with faces
		faces.clear();
		// Detect all faces.
		cascade.detectMultiScale(gray, faces, 1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE, Size(60, 60), Size(180, 180));

		t = (double)getTickCount() - t;
		cout << format("Face detection. Time for processing single image required: %.2f seconds.", t / (double)getTickFrequency()) << endl;

		Rect biggestFace(Point(0,0), Point(0,0));
		double biggestSize = 0;
		
		// For all detected faces
		for (unsigned int i = 0; i < faces.size(); i++) {
			Rect r = faces[i];

			// Draw rectangle around a face
			// rectangle(frame, Point(round(r.x), round(r.y)), Point(round((r.x + r.width-1)), round((r.y + r.height-1))), Scalar(255, 255, 255), 3, 8, 0);

			// Search for the biggest face in the image
			double size = r.width * r.height;
			if (size > biggestSize) {
				biggestSize = size;
				
				biggestFace.x = r.x;
				biggestFace.y = r.y;
				biggestFace.width = r.width;
				biggestFace.height = r.height;

				// Increase the size of the window around the face
				while (biggestFace.x > r.x - r.width * faceFactor && biggestFace.x > 0){
					biggestFace.x -= 1;
				}
				while (biggestFace.y > r.y - r.height * faceFactor && biggestFace.y > 0){
					biggestFace.y -= 1;
				}
				while(biggestFace.x + biggestFace.width < r.x + r.width + r.width * faceFactor && biggestFace.x + biggestFace.width < frame.cols) {
					biggestFace.width += 1;
				}
				while(biggestFace.y + biggestFace.height < r.y + r.height + r.height * faceFactor && biggestFace.y + biggestFace.height < frame.rows) {
					biggestFace.height += 1;
				}
			}
		}

		// If face has been found
		if (biggestSize != 0) {
			crop = crop(biggestFace);

			// canvas = nImagesInRow * nImagesInColumn
			resize(crop, crop, Size(canvasWidth / nImagesInRow, canvasHeight / nImagesInColumn));

			// Add effects
			crop = makePanel(crop, nImagesInColumn, nImagesInRow);

			imshow(WINDOW_NAME, crop);

			usleep(1000 * 1000 * secondsOfShowingImage);
		}

		// Hack for Raspberry Pi
		if (waitKey(10) > 0) {
			break;
		}
	}
	return 0;
}
