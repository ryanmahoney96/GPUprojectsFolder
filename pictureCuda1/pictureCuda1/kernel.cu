
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "highPerformanceTimer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui//highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

/*struct threshData {
	int* threshNum;
	Mat* image;
};*/

__global__ void threshKernel()
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = threadIdx.x;

}

double CPUthreshold (int threshold, int w, int height, unsigned char* data);
double GPUthreshold(int threshold, Mat* image);

//void on_trackbar (int thresholdNum, void* td);

typedef unsigned char uchar;

int main(int argc, char * argv[])
{
	//int endRet = 0;
	int thresholdNum = 128;
	//const int threshMAX = 255;

	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		//return from 1 place!!
		//endRet = -1;
		return -1;
	}
	

	Mat image;
	//read the file from the command line
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
	/*threshData td;
	td.threshNum = &thresholdNum;
	td.image = &image;*/

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

	double CPUTime = CPUthreshold(thresholdNum, image.cols, image.rows, image.data);
	double GPUTime = GPUthreshold(thresholdNum, &image);

	//create a window for display
	namedWindow("Display Window", WINDOW_NORMAL);

	//show the image within the display window
	imshow("Display Window", image);

	//the trackbar must be placed inside a window
	//whenever the user changes the trackbar, the on_trackbar function is called
	//trackbar name, window name, an int to change(?), a maximum value(?), and a function to call
	//createTrackbar("Threshold Slider", "Display Window", &thresholdNum, threshMAX, on_trackbar, &td);

	//on_trackbar(thresholdNum, &td);

	//wait for the user to enter a keystroke
	cout << "The CPU took " << CPUTime << " seconds to render the threshold on the image" << endl;
	cout << "The GPU took " << GPUTime << " seconds to render the threshold on the image" << endl;

	waitKey(0);


    return 0;
}

double CPUthreshold(int threshold, int width, int height, unsigned char* data) {
	
	//for CPU version, time this loop
	HighPrecisionTime render;
	double renderTime = 0.0;

	render.TimeSinceLastCall();

	for (uchar* i = data; i < &data[width*height]; i++) {

		if (*i > threshold) {
			*i = 255;
		}
		else {
			*i = 0;
		}

	}

	renderTime = render.TimeSinceLastCall();

	return renderTime;
}

double GPUthreshold(int threshold, Mat* image) {

	HighPrecisionTime render;
	double renderTime = 0.0;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int maxThreadsPerBlock = prop.maxThreadsPerBlock;
	int numberOfBlocks = (image->cols * image->rows) / maxThreadsPerBlock + 1;

	cout << maxThreadsPerBlock << endl << numberOfBlocks << endl;


	
	return renderTime;
}

/*void on_trackbar(int thresholdNum, void* td) {

	Mat* img = ((threshData*)td)->image;

	threshold(*td->thresholdNum, img->cols, img->rows, img->data);

	imshow("Display Window", *img);

}*/