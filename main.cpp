#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <fstream>

using namespace std;
using namespace cv;

#define MIN_HESSIAN 400
#define NUMBER_OF_FILES 500

Mat trainingData, labels, training;

TermCriteria termCrit(CV_TERMCRIT_ITER, 1000, 0.001);

vector< Mat > loadImages() {
	vector<Mat> vec;
	Mat img;

	for (int i = 1; i <= NUMBER_OF_FILES; i++)
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

	for (int i = 0; i < NUMBER_OF_FILES - 1; i++) {
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

Mat sift(vector<Mat> images) {
	Mat training;
	for(int i = 0; i < NUMBER_OF_FILES; i++)
		if (images[i].data)
		{
			Ptr<FeatureDetector> detector = new cv::SiftFeatureDetector();
			Ptr<DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();
			vector<KeyPoint> kps;
			Mat descriptor;
			Mat dsc;
			detector->detect(images[i], kps);
			extractor->compute(images[i], kps, dsc);
			training.push_back(dsc);
		}

	return training;
}

void bagOfWords(vector<Mat> images) {
	Mat dictionary;
	BOWKMeansTrainer bow(10, termCrit, 2, 0);
	dictionary = bow.cluster(training);
	Ptr<DescriptorMatcher> dsc_matcher(new FlannBasedMatcher);
	Ptr<DescriptorExtractor> dsc_train = new SiftDescriptorExtractor();
	Ptr<FeatureDetector> feat_train = new SiftFeatureDetector();
	BOWImgDescriptorExtractor bowDescExtractor(dsc_train, dsc_matcher);

	bowDescExtractor.setVocabulary(dictionary);

	for (int i = 0; i < NUMBER_OF_FILES; i++) {
		Ptr<FeatureDetector> feat_train = new SiftFeatureDetector();
		vector<KeyPoint> kps;
		feat_train->detect(images[i], kps);
		Mat dsc;
		bowDescExtractor.compute(images[i], kps, dsc);

		trainingData.push_back(dsc);
	}

}

int main()
{
	vector<Mat> images = loadImages();
	loadLabels();

	training = sift(images);
	bagOfWords(images);
		
	bool use_knn = 1;	
	if (use_knn) {
		CvKNearest *knn;
		knn = new KNearest(trainingData, labels, Mat(), false, 5);
		cout << "trainingData:" << trainingData.size();
		cout << "labels:" << labels.size();
	}

	cout << trainingData.size();
	cout << labels.size();
	while (true);
	return 0;
}