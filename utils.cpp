#include "utils.h"
#include <vector>
#include <opencv2\opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace std;
using namespace cv;

#define MIN_HESSIAN 400
#define NUMBER_OF_FILES 500

vector< Mat > loadImages() {
	vector<Mat> vec;
	Mat img;

	for (int i = 1; i <= 1000; i++)
	{
		img = imread("../train/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		//flatten the image
		//img.resize(1);
		vec.push_back(img);
	}
	return vec;
}

vector<int> loadLabels() {

}

vector<KeyPoint> surf(Mat img) {
	SurfFeatureDetector detector(MIN_HESSIAN);
	vector<KeyPoint> kps;

	detector.detect(img, kps);

	return kps;
}