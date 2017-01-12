
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui//highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void threshold (int threshold, int w, int height, unsigned char* data);

typedef unsigned char uchar;

int main(int argc, char * argv[])
{
	//int endRet = 0;

	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		//return from 1 place!!
		//endRet = -1;
		return -1;
	}
	

	Mat image;
	//read the file from the command line
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	

	//check for an invalid input
	//Mat.data is a pointer to the image data -> if null, no image found/useable
	if (!image.data) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//we want: rows, cols, channels
	cout << "The image is " << image.rows << "x" << image.cols << " in dimension" << endl;

	cvtColor(image, image, cv::COLOR_RGB2GRAY);
	cout << "Image converted to Grayscale" << endl;

	threshold(128, image.cols, image.rows, image.data);

	//create a window for display
	namedWindow("Display Window", WINDOW_NORMAL);
	//show the image within the display window
	imshow("Display Window", image);

	//wait for the user to enter a keystroke
	waitKey(0);


    return 0;
}

void threshold(int threshold, int width, int height, unsigned char* data) {
	
	for (uchar* i = data; i < &data[width*height]; i++) {

		if (*i > threshold) {
			*i = 255;
		}
		else {
			*i = 0;
		}

	}

}