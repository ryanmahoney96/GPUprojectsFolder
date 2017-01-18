
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "highPerformanceTimer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui//highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

typedef unsigned char uchar;

const char* cpuWindow = "CPU Window";
const char* gpuWindow = "GPU Window";

Mat image;
Mat GPUImage;

__global__ void threshKernel(uchar* imageData, size_t size_of_image, int threshold)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = threadIdx.x;

	//change the loop / make sure not to go over the "edge"
	//for (uchar* i = imageData; i < &imageData[size_of_image]; i++) {

		if (imageData[i] > threshold) {
			imageData[i] = 255;
		}
		else {
			imageData[i] = 0;
		}

	//}

}

double cpuBoxFilter(uchar* src, uchar*& dst, int width, int height, uchar* k, int kerWidth, int kerHeight, uchar* tmp);

double CPUthreshold (int threshold, int w, int height, unsigned char* data);
double GPUthreshold(int threshold, Mat* image, Mat& renderedImage);

void on_cpu_trackbar (int cpuThresholdNum, void*);
void on_gpu_trackbar(int gpuThresholdNum, void*);


int main(int argc, char * argv[])
{
	int cpuThresholdNum = 128;
	int gpuThresholdNum = 128;

	const int threshMAX = 255;

	try {
		if (argc != 2) {
			throw("Usage: display_image ImageToLoadAndDisplay");
		}

		//read the file from the command line
		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

		//check for an invalid input
		//Mat.data is a pointer to the image data -> if null, no image found/useable
		if (!image.data) {
			throw("Could not open or find the image");
		}

		//we want: rows, cols, channels
		cout << "The image is " << image.rows << "x" << image.cols << " in dimension" << endl;

		cvtColor(image, image, cv::COLOR_RGB2GRAY);
		cout << "Image converted to Grayscale" << endl;

		GPUImage = image.clone();

		Mat cputmp = image.clone();
		Mat gputmp = image.clone();

		double CPUTime = 0;
		double GPUTime = 0;

		uchar boxKernel[] = { 1, 1, 1, 
							  1, 3, 1, 
							  1, 1, 1 
							};

		//NOTE: change hardcoded 3's, change last parameter to useful data
		CPUTime = cpuBoxFilter(image.data, cputmp.data, image.cols, image.rows, boxKernel, 3, 3, 0);

		//thresholds
		//CPUTime = CPUthreshold(cpuThresholdNum, cputmp.cols, cputmp.rows, cputmp.data);
		//GPUTime = GPUthreshold(gpuThresholdNum, &cputmp, gputmp);

		//create a window for display
		namedWindow(cpuWindow, WINDOW_NORMAL);
		namedWindow(gpuWindow, WINDOW_NORMAL);

		//show the image within the display window
		imshow(cpuWindow, cputmp);
		imshow(gpuWindow, image);

		//trackbars
		//the trackbar must be placed inside a window
		//whenever the user changes the trackbar, the on_trackbar function is called
		//trackbar name, window name, an int to change(?), a maximum value(?), and a function to call
		/*createTrackbar("Threshold", cpuWindow, &cpuThresholdNum, threshMAX, on_cpu_trackbar);
		createTrackbar("Threshold", gpuWindow, &gpuThresholdNum, threshMAX, on_gpu_trackbar);

		on_cpu_trackbar(cpuThresholdNum, 0);
		on_gpu_trackbar(gpuThresholdNum, 0);*/

		//wait for the user to enter a keystroke
		cout << "The CPU took " << CPUTime << " seconds to render the threshold on the image" << endl;
		cout << "The GPU took " << GPUTime << " seconds to render the threshold on the image" << endl;

		//cout << "The GPU is " << CPUTime / GPUTime << " times faster than the CPU" << endl;

		waitKey(0);
	}
	catch (char* err) {
		cout << err << endl;
	}

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

double GPUthreshold(int threshold, Mat* img, Mat& renderedImage) {

	HighPrecisionTime render;
	HighPrecisionTime copying;
	double renderTime = 0.0;
	double copyTime = 0.0;

	//consider pass by reference
	//Mat GPUImage = (*image).clone();

	uchar* GPUImageData;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int maxThreadsPerBlock = prop.maxThreadsPerBlock;
	int numberOfBlocks = (img->cols * img->rows) / maxThreadsPerBlock + 1;

	size_t size_of_image = img->cols * img->rows * sizeof(uchar);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		throw("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&GPUImageData, size_of_image);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMalloc of image failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy((void*)GPUImageData, (void*)img->data, size_of_image, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy of image from CPU to GPU failed!");
		goto Error;
	}

	/*1024
	32401*/
	//cout << maxThreadsPerBlock << endl << numberOfBlocks << endl;

	render.TimeSinceLastCall();

	//The first argument in the execution configuration specifies the number of thread blocks in the grid, and the second specifies the number of threads in a thread block.
	threshKernel <<< numberOfBlocks, maxThreadsPerBlock >>> (GPUImageData, size_of_image, threshold);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	renderTime = render.TimeSinceLastCall();

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy((void*)renderedImage.data, GPUImageData, size_of_image, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy of image data from GPU to CPU failed!");
		goto Error;
	}



Error:
	cudaFree(GPUImageData);
	
	return renderTime;
}

void on_cpu_trackbar(int cpuThresholdNum, void*) {

	Mat cputmp = GPUImage.clone();

	double CPUTime = CPUthreshold(cpuThresholdNum, cputmp.cols, cputmp.rows, cputmp.data);

	cout << "CPU Threshold Number: " << cpuThresholdNum << endl;

	imshow(cpuWindow, cputmp);
}

void on_gpu_trackbar(int gpuThresholdNum, void*) {

	Mat cputmp = GPUImage.clone();
	Mat gputmp = GPUImage.clone();

	double GPUTime = GPUthreshold(gpuThresholdNum, &cputmp, gputmp);

	cout << "GPU Threshold Number: " << gpuThresholdNum << endl;

	imshow(gpuWindow, gputmp);

}

// CPUTime = cpuBoxFilter(image.data, cputmp.data, image.cols, image.rows, boxKernel, 3, 3, 0);

double cpuBoxFilter(uchar* src, uchar*& dst, int width, int height, uchar* k, int kerWidth, int kerHeight, uchar* tmp) {

	HighPrecisionTime filter;
	double filterTime = 0.0;
	
	//the sum of the prox pixels after filtering
	double currentSum = 0.0;

	double avgNum = 0.0;
	double numCells = 0;

	for (int it = 0; it < (numCells = kerWidth * kerHeight); it++) {
		avgNum += k[it];
	}

	//avgNum = avgNum / numCells;

	//start on pixel
	//find the pixels in proximity
	//multiply their b/w values by corresponding filter val
	//sum all prox pixels
	//divide by sum of vals in kernel array
	//save new value in dst.data[i]
	//move to next pixel, continue

	//the direction one needs to move in order to get to the desired border pixel relative to the current pixel
	//NOTE: generalize this using the dimensions of the kernel
	//
	int indices [] = {
		-(width + 1),  -width,     -(width - 1),
		-1,                0,           +1,
		width - 1,      width,      width + 1
	};

	filter.TimeSinceLastCall();

	//roll through every pixel in the image using this loop
	for (int i = width; i < (width*height - width); i++) {

		currentSum = 0.0;

		//if edge pixel
		if (i % width == 0 || (i % width) - 1 == 0 ) {
			continue;
		}

		//roll through each kernel corresponding pixel surrounding the current pixel (i)
		for (int j = 0; j < numCells; j++) {
			currentSum += src[i + indices[j]] * k[j];
		}

		dst[i] = currentSum / avgNum;

	}


	filterTime = filter.TimeSinceLastCall();

	return filterTime;
}