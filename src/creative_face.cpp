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


////////////////////////////////////////////////////////////////////
// Nastavitve
////////////////////////////////////////////////////////////////////
int secondsOfShowingImage = 3;
// Za koliko povecamo izrez okoli obraza, katerega zazna face detection algoritem.
double faceFactor = 0.4;
// Platno = mreza slik = nColumns * nRows.
int canvasWidth = 768;
int canvasHeight = 768;
int nColumns = 2; // Koliko slik bo v eni vrstici platna.
int nRows = 2; // Koliko slik bo v enem stolpcu platna.
// Odvisno od kamere - to vpliva na performance.
int imageCaptureWidth = 1024;
int imageCaptureHeight = 768;
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

CascadeClassifier cascade;
RNG rng(1);


// Funkcija za efekte.
Mat warhol(Mat src, int K, int blurSize) {
  Mat img, bestLabels, centers, clustered;
  Vec3b *colors = new Vec3b[K];

  //Blur the image
  blur(src, img, Size(blurSize, blurSize));

  //Create samples
  Mat samples = img.reshape(1, img.rows * img.cols);
  samples.convertTo(samples, CV_32F);

  //K-Means clustering
  kmeans(samples, K, bestLabels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

  //Select a random palette
  for(int i = 0; i < K; i ++) {
    Vec3b color = Vec3b( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    colors[i] = color;
  }

  //Draw the new image
  clustered = Mat(img.size(), img.type());
  for(int i = 0; i < img.cols*img.rows; i++) {
    clustered.at<Vec3b>(i/img.cols, i%img.cols) = colors[bestLabels.at<int>(0,i)];
  }

  return clustered;
}


// Funkcija, ki naredi platno iz vecih slik (rows * cols).
Mat makePanel(Mat src, int rows, int cols) {
  Mat dst = Mat(src.rows * rows, src.cols * cols, src.type());
  Mat trgt;
  for(int i = 0; i < rows; i ++) {
    for (int j = 0; j < cols; j ++) {
      warhol(src, 8, 5).copyTo(dst(Rect(j * src.cols, i * src.rows, src.cols, src.rows))); //8
      //trgt = warhol(src, 8, 5);
    }
  }
  return dst;
}

int main(int argc, char** argv )
{
  CommandLineParser parser(argc, argv,
    "{width          |     | Sirina platna (default: 768).   }"
    "{height         |     | Visina platna (default: 768).   }"
    "{cameraWidth    |     | Sirina zajemanja slik s kamere (default: 1024). }"
    "{cameraHeight   |     | Visina zajemanja slik s kamere (default: 768). }"
    "{col            |     | Stevilo slik bo v eni vrstici platna (default: 1). }"
    "{row            |     | Stevilo slik bo v enem stolpcu platna (default: 1). }"
    "{time           |     | Cas med iskani obrazov (prikazovanjem enega obraza) [sekunde] (default: 15). }"
    "{faceFactor     |     | Faktor povecanja slike okoli obraza (default: 0.4). }" );

  parser.about("\navtor projekta: Denis Kotnik\n\nReinplementacija 15 sekund slave na Raspberry Pi\n\n");
  parser.printMessage();

  if (parser.has("width"))
  {
    canvasWidth = parser.get<int>("width");
  }
  if (parser.has("height"))
  {
    canvasHeight = parser.get<int>("height");
  }
  if (parser.has("cameraWidth"))
  {
      imageCaptureWidth = parser.get<int>("cameraWidth");
  }
  if (parser.has("cameraHeight"))
  {
      imageCaptureHeight = parser.get<int>("cameraHeight");
  }
  if (parser.has("col"))
  {
      nColumns = parser.get<int>("col");
  }
  if (parser.has("row"))
  {
      nRows = parser.get<int>("row");
  }
  if (parser.has("time"))
  {
      secondsOfShowingImage = parser.get<int>("time");
  }
  if (parser.has("faceFactor"))
  {
      faceFactor = parser.get<double>("faceFactor");
  }

  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////
  // Na zacetku prikazemo sliko Marilyn Monroe.
  Mat logo = imread("./blank_face.png");
  if (!logo.empty()) {
    resize(logo, logo, Size(canvasWidth, canvasHeight));
    imshow(WINDOW_NAME, logo);
  }

  // Nalozimo datoteko z znacilkami obraza.
  if( !cascade.load("./face_cascade.xml") ) {
      cerr << "NAPAKA: ni mogoce naloziti datoteke 'classifier cascade'!" << endl;
      return -1;
  }

  VideoCapture camera(0); // ID kamere.
 
  if (!camera.isOpened()) { // Ce do kamere ni mogoce dostopati.
    cerr << "NAPAKA: do kamere ni mogoce dostopati!" << endl;
    return -1;
  }

  // Nastavimo sirino in visino zajemanja slik.
  camera.set(CAP_PROP_FRAME_WIDTH, imageCaptureWidth);
  camera.set(CAP_PROP_FRAME_HEIGHT, imageCaptureHeight);

  Mat frame, gray, crop, slikaObraza;
  vector<Rect> faces;
  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////
  while (true) {
    // Hack za Raspberry Pi oz. Linux OS, ker se v buffer gonilnika za kamero vedno nalozi 5 slik.
    for (int i = 0; i < 5; i++) {
      camera.read(frame);
      frame.copyTo(crop);
    }

    cvtColor(frame, gray, COLOR_BGR2GRAY); // Pretvorimo v crno belo sliko.

    double t = 0;
    t = (double)getTickCount();

    faces.clear(); // Izpraznimo vektor z obrazi.
    // Poiscemo vse obraze.
    cascade.detectMultiScale(gray, faces,
        1.1, 3, CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH | CASCADE_SCALE_IMAGE,
        Size(60, 60), Size(180, 180) );

    t = (double)getTickCount() - t;
    cout << format("Iskanje obrazov. Cas procesiranja za eno sliko: %.2f sekund.", 
      t / (double)getTickFrequency()) << endl;

    Rect najvecjiObraz(Point(0,0), Point(0,0));
    double najvecjaVelikost = 0;
    // Gremo cez vse obraze, ki so bili detektirani.
    for (unsigned int i = 0; i < faces.size(); i++) {
        Rect r = faces[i];

        // Okoli vsakega obraza izrisemo pravokotnik.
        //rectangle(frame, Point(round(r.x), round(r.y)),
                  //Point(round((r.x + r.width-1)), round((r.y + r.height-1))),
                  //Scalar(255, 255, 255), 3, 8, 0);

        // Poiscemo najvecji obraz na sliki.
        double velikost = r.width * r.height;
        if (velikost > najvecjaVelikost) {
          najvecjaVelikost = velikost;
          
          najvecjiObraz.x = r.x;
          najvecjiObraz.y = r.y;
          najvecjiObraz.width = r.width;
          najvecjiObraz.height = r.height;

          // Povecujem okno okoli obraza.
          while (najvecjiObraz.x > r.x - r.width * faceFactor && najvecjiObraz.x > 0){
            najvecjiObraz.x -= 1;
          }
          while (najvecjiObraz.y > r.y - r.height * faceFactor && najvecjiObraz.y > 0){
            najvecjiObraz.y -= 1;
          }
          while(najvecjiObraz.x + najvecjiObraz.width < r.x + r.width + r.width * faceFactor 
            && najvecjiObraz.x + najvecjiObraz.width < frame.cols) {
            najvecjiObraz.width += 1;
          }
          while(najvecjiObraz.y + najvecjiObraz.height < r.y + r.height + r.height * faceFactor 
            && najvecjiObraz.y + najvecjiObraz.height < frame.rows) {
            najvecjiObraz.height += 1;
          }
        }
    }
    // Ce je obraz najden
    if (najvecjaVelikost != 0) {
        crop = crop(najvecjiObraz); // Sliko obrezemo.

        // Spremenimo velikost slike (platno = mreza slik = nColumns * nRows)
        resize(crop, crop, Size(canvasWidth / nColumns, canvasHeight / nRows));

        // Sliko obogatimo z efekti.
        crop = makePanel(crop, nRows, nColumns);
	
        imshow(WINDOW_NAME, crop); // Prikazemo obrezano in obogateno sliko.

        usleep(1000 * 1000 * secondsOfShowingImage); // Pocakaj.
    }

    // To potrebujemo obvezno, drugace se na Raspberry Pi okna ne prikazujejo.
    if (waitKey(10) > 0) {
      break;
    }
  }
  return 0;

}
