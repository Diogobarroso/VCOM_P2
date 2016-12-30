#include "utils.h"
#include <opencv2\nonfree\features2d.hpp>

using namespace std;
using namespace cv;

Mat trainingData, labels;

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

	Mat trainingDescriptors = sift(images);

		
	bool use_knn = 0;	
	if (use_knn) {
		CvKNearest knn;
		knn.train(trainingData, labels, Mat(), false, 32, false);
	}

	return 0;
}