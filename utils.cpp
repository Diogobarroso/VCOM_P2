#include "utils.h"
#include <vector>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

vector< Mat > loadImages() {
	vector<Mat> vec;

	for (int i = 1; i <= 1000; i++)
	{
		vec.push_back(imread("../train/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE).resize(1));
	}
	return vec;
}