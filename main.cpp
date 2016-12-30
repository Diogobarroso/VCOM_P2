#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <fstream>

using namespace std;
using namespace cv;

#define MIN_HESSIAN 400
#define NUMBER_OF_FILES 40000

Mat trainingData, labels, training;

TermCriteria termCrit(CV_TERMCRIT_ITER, 1500, 0.0001);

vector<Mat> loadImages()
{
	vector<Mat> vec;
	Mat img;

	for (int i = 1; i <= NUMBER_OF_FILES; i++)
	{
		img = imread("../train/" + to_string(i) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		if (!img.data)
			while (true);
		//flatten the image
		//img.resize(1);
		vec.push_back(img);
		cout << i << "/" << NUMBER_OF_FILES << "\n";
	}
	cout << vec.size();
	return vec;
}

int parseLabel(string label)
{
	if (label == "airplane")
		return 0;
	else if (label == "automobile")
		return 1;
	else if (label == "bird")
		return 2;
	else if (label == "cat")
		return 3;
	else if (label == "deer")
		return 4;
	else if (label == "dog")
		return 5;
	else if (label == "frog")
		return 6;
	else if (label == "horse")
		return 7;
	else if (label == "ship")
		return 8;
	else if (label == "truck")
		return 9;
}

Mat sift(vector<Mat> images)
{
	Mat training;
	ifstream file("../labels/trainLabels.csv");
	string line;
	int label;
	for (int i = 0; i < NUMBER_OF_FILES; i++)
	{
		getline(file, line);
		label = parseLabel(line.substr(line.find(",") + 1));
		Ptr<FeatureDetector> detector = new SiftFeatureDetector();
		Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
		vector<KeyPoint> kps;
		Mat descriptor;
		Mat dsc;
		detector->detect(images[i], kps);
		extractor->compute(images[i], kps, dsc);
		if (!dsc.empty())
		{
			training.push_back(dsc);
			labels.push_back(label);
		}
	}
	cout << labels.size();

	return training;
}

BOWImgDescriptorExtractor bagOfWords(vector<Mat> images)
{
	Mat dictionary;
	BOWKMeansTrainer bow(10, termCrit, 3, 0);
	//cout << labels.size();
	dictionary = bow.cluster(training);
	Ptr<DescriptorMatcher> dsc_matcher(new FlannBasedMatcher);
	Ptr<DescriptorExtractor> dsc_train = new SiftDescriptorExtractor();
	Ptr<FeatureDetector> feat_train = new SiftFeatureDetector();
	BOWImgDescriptorExtractor bowDescExtractor(dsc_train, dsc_matcher);

	bowDescExtractor.setVocabulary(dictionary);

	for (int i = 0; i < NUMBER_OF_FILES; i++)
	{
		Ptr<FeatureDetector> feat_train = new SiftFeatureDetector();
		vector<KeyPoint> kps;
		feat_train->detect(images[i], kps);
		Mat dsc;
		bowDescExtractor.compute(images[i], kps, dsc);

		trainingData.push_back(dsc);
	}

	cout << trainingData.size();
	return bowDescExtractor;
}

int main()
{
	vector<Mat> images = loadImages();

	training = sift(images);
	BOWImgDescriptorExtractor bow = bagOfWords(images);

	bool use_knn = false;
	bool use_svm = true;
	if (use_knn)
	{
		CvKNearest* knn;
		cout << "Now training KNN...\n";
		knn = new KNearest(trainingData, labels, Mat(), false, 5);
		cout << "KNN trained. Now testing...\n";

		Mat test = imread("../train/" + to_string(NUMBER_OF_FILES + 1) + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		vector<KeyPoint> kps_test;
		Ptr<FeatureDetector> feat_test = new SiftFeatureDetector();
		Mat dsc_test;
		feat_test->detect(test, kps_test);
		bow.compute(test, kps_test, dsc_test);
		int result = 1;
		Mat results;
		knn->find_nearest(dsc_test, result, results, Mat(), Mat());
		cout << "Image:" << NUMBER_OF_FILES + 1 << "is a " << results;

	}

	// SVM

	if (use_svm)
	{
		Ptr<SVM> svm_linear = new SVM;
		Ptr<SVM> svm_rbf = new SVM;
		Ptr<SVM> svm_sig = new SVM;

		CvSVMParams params_lin;
		params_lin.svm_type = CvSVM::C_SVC;
		params_lin.kernel_type = CvSVM::LINEAR;
		params_lin.term_crit = termCrit;

		CvSVMParams params_rbf;
		params_rbf.svm_type = CvSVM::C_SVC;
		params_rbf.kernel_type = CvSVM::RBF;
		params_rbf.term_crit = termCrit;

		CvSVMParams params_sig;
		params_sig.svm_type = CvSVM::C_SVC;
		params_sig.kernel_type = CvSVM::SIGMOID;
		params_sig.term_crit = termCrit;

		cout << "Training the SVM Linear" << endl;
		svm_linear->train(trainingData, labels, Mat(), Mat(), params_lin);
		cout << "SVM Linear Trained" << endl;

		//cout << "Training SVM RBF" << endl;
		//svm_rbf->train(trainingData, labels, Mat(), Mat(), params_rbf);
		//cout << "SVM RBF Trained" << endl;

		//cout << "Training the SVM Sigmoid" << endl;
		//svm_sig->train(trainingData, labels, Mat(), Mat(), params_sig);
		//cout << "SVM Sigmoid Trained" << endl;

		cout << "trainingData:" << trainingData.size() << endl;
		cout << "labels:" << labels.size() << endl;

		float resultSVMLin;
		float resultSVMRBF;
		float resultSVMSig;

		ofstream svm_csv("./svm_results.csv");
		svm_csv << "id;label;\n";

		for (unsigned int i = 0; i < NUMBER_OF_FILES; i++)
		{
			Ptr<FeatureDetector> detector = new SiftFeatureDetector();
			Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
			vector<KeyPoint> keypoints;
			Mat bow_descriptor;

			detector->detect(images[i], keypoints);
			bow.compute(images[i], keypoints, bow_descriptor);

			try
			{
				resultSVMLin = svm_linear->predict(bow_descriptor);
				//resultSVMRBF = svm_rbf->predict(bow_descriptor);
				//resultSVMSig = svm_sig->predict(bow_descriptor);
			}
			catch (cv::Exception)
			{
				resultSVMLin = -1;
			}

			cout << "result SVMLin for image " << i << ": " << resultSVMLin << endl;
			svm_csv << i + 1 << ";" << resultSVMLin << ";\n";

			//cout << "result SVMRBF for image " << i << ": " << resultSVMLin << endl;
			//cout << "result SVMSig for image " << i << ": " << resultSVMLin << endl;
		}

		svm_linear->save("./Results.yaml");
	}

	cin.get();
	return 0;
}
