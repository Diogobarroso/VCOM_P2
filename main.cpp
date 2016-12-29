#include "utils.h"

int main()
{
	vector<Mat> images = loadImages();
	imshow("test", images[0]);
	return 0;
}