#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector< Mat > loadImages();

vector<int> loadLabels();

vector<KeyPoint> surf(Mat img);

