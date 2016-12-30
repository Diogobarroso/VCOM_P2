#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <fstream>

using namespace std;
using namespace cv;

#define MIN_HESSIAN 400
#define NUMBER_OF_FILES 500

Mat trainingData, labels;

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

void loadLabels() {
	ifstream file("../labels/trainLabels.csv");
	string line;

	for (int i = 0; i < NUMBER_OF_FILES; i++) {
		getline(file, line);
		string label = line.substr(line.find(",") + 1);
		if (label == "airplane")
			labels.push_back(0);
		else if (label == "automobile")
			labels.push_back(1);
		else if (label == "bird")
			labels.push_back(2);
		else if (label == "cat")
			labels.push_back(3);
		else if (label == "deer")
			labels.push_back(4);
		else if (label == "dog")
			labels.push_back(5);
		else if (label == "frog")
			labels.push_back(6);
		else if (label == "horse")
			labels.push_back(7);
		else if (label == "ship")
			labels.push_back(8);
		else if (label == "truck")
			labels.push_back(9);
	}

}

vector<KeyPoint> surf(Mat img) {
	SurfFeatureDetector detector(MIN_HESSIAN);
	vector<KeyPoint> kps;

	detector.detect(img, kps);

	return kps;
}

Mat sift(vector<Mat> images) {
	Mat trainingDescriptors;
	Mat descriptor;
	Ptr<FeatureDetector> detector = new cv::SiftFeatureDetector();
	Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

	for(int i = 0; i < images.size(); i++)
		if (images[i].data)
		{
			vector<KeyPoint> kps;
			Mat dsc;
			detector->detect(images[i], kps);
			extractor->compute(images[i], kps, dsc);
			trainingDescriptors.push_back(dsc);
		}
	return trainingDescriptors;
}

int main()
{
	vector<Mat> images = loadImages();

	trainingData = sift(images);

		
	bool use_knn = 0;	
	if (use_knn) {
		CvKNearest knn;
		knn.train(trainingData, labels, Mat(), false, 32, false);
	}

	return 0;
}