
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "highPerformanceTimer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui//highgui.hpp>

#include <algorithm>

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

__global__ void bfKernel(uchar* src, uchar* dst, int width, int height, int* k, int kerWidth, int kerHeight) {
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//the sum of the prox pixels after filtering
	float currentSum = 0.0f;

	float avgNum = 0.0f;

	for (int it = 0; it < (kerWidth * kerHeight); it++) {
		avgNum += k[it];
	}

	//don't divide by 0 down below
	if (avgNum == 0) {
		avgNum = 1;
	}

	int coeff = kerWidth / 2;

	//check to make sure its NOT an edge pixel
	//if (x - coeff < 0 || x + coeff > width || y - coeff < 0 || y + coeff > height) {
	//	//
	//}

	//else {

		//roll through each kernel corresponding pixel surrounding the current pixel
		int currentPixel = y * width + x;

		int j = 0;

		for (int m = -coeff; m <= coeff; m++) {
			for (int c = -coeff; c <= coeff; c++) {
				//i is the current pixel. We find the pixels around i by multiplying the number of widths away the desired modifying pixel is from origin by width, plus the number of pixels horizontally
				currentSum += src[currentPixel + (m * width + c)] * k[j];
				j++;
			}
		}

		dst[currentPixel] = (uchar)abs((currentSum / avgNum));

	//}
}

double cpuBoxFilter(uchar* src, uchar*& dst, int width, int height, int* k, int kerWidth, int kerHeight, uchar* tmp);

double GPUBoxFilter(Mat* src, Mat& dst, int width, int height, int* k, int kerWidth, int kerHeight);

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
		cout << "The image is " << image.cols << "x" << image.rows << " in dimension" << endl;

		cvtColor(image, image, cv::COLOR_RGB2GRAY);
		cout << "Image converted to Grayscale" << endl;

		GPUImage = image.clone();

		Mat cputmp = image.clone();
		Mat gputmp = image.clone();

		double CPUTime = 0;
		double GPUTime = 0;

		const int arraySize = 9;

		int boxKernel[arraySize* arraySize];
		/*{
			-1, 0, 1,
			-2, 0, 2,
			-1, 0, 1, 255
		};*/

		/*{ 0, 0, 0, 0, 0,
			0, 0, -1, 0, 0,
			0, -1, 2, -1, 0,
			0, 0, -1, 0, 0,
			0, 0, 0, 0, 0, 255
			};*/

		for (int i = 0; i < arraySize * arraySize; i++) {
			boxKernel[i] = 1;
		}

		//NOTE: change last parameter to useful data
		CPUTime = cpuBoxFilter(image.data, cputmp.data, image.cols, image.rows, boxKernel, arraySize, arraySize, 0);
		GPUTime = GPUBoxFilter(&image, gputmp, image.cols, image.rows, boxKernel, arraySize, arraySize);

		//thresholds
		//CPUTime = CPUthreshold(cpuThresholdNum, cputmp.cols, cputmp.rows, cputmp.data);
		//GPUTime = GPUthreshold(gpuThresholdNum, &cputmp, gputmp);

		//create a window for display
		namedWindow(cpuWindow, WINDOW_NORMAL);
		namedWindow(gpuWindow, WINDOW_NORMAL);

		//show the image within the display window
		imshow(cpuWindow, image);
		imshow(gpuWindow, image);

		waitKey(0);
		imshow(cpuWindow, cputmp);
		imshow(gpuWindow, gputmp);

		//trackbars
		//the trackbar must be placed inside a window
		//whenever the user changes the trackbar, the on_trackbar function is called
		//trackbar name, window name, an int to change(?), a maximum value(?), and a function to call
		/*createTrackbar("Threshold", cpuWindow, &cpuThresholdNum, threshMAX, on_cpu_trackbar);
		createTrackbar("Threshold", gpuWindow, &gpuThresholdNum, threshMAX, on_gpu_trackbar);

		on_cpu_trackbar(cpuThresholdNum, 0);
		on_gpu_trackbar(gpuThresholdNum, 0);*/

		//wait for the user to enter a keystroke
		cout << "The CPU took " << CPUTime << " seconds to render the effect on the image" << endl;
		cout << "The GPU took " << GPUTime << " seconds to render the effect on the image" << endl;

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

double GPUBoxFilter(Mat* src, Mat& dst, int width, int height, int* k, int kerWidth, int kerHeight) {

	HighPrecisionTime render;
	HighPrecisionTime copying;
	double renderTime = 0.0;
	double copyTime = 0.0;

	//consider pass by reference
	//Mat GPUImage = (*image).clone();

	uchar* GPUImageData;
	uchar* tmp;

	int* gpu_k;

	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int numPixels = width * height;

	int maxThreadsPerBlock = prop.maxThreadsPerBlock;
	int numberOfBlocks = numPixels / maxThreadsPerBlock + 1;

	size_t size_of_image = numPixels * sizeof(uchar);

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

	cudaStatus = cudaMalloc((void**)&tmp, size_of_image);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMalloc of image failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gpu_k, kerHeight * kerWidth);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMalloc of image failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy((void*)GPUImageData, (void*)src->data, size_of_image, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy of image from CPU to GPU failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy((void*)gpu_k, (void*)k, kerHeight * kerWidth, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy of image from CPU to GPU failed!");
		goto Error;
	}

	render.TimeSinceLastCall();

	//The first argument in the execution configuration specifies the number of thread blocks in the grid, and the second specifies the number of threads in a thread block.
	bfKernel <<< numberOfBlocks, maxThreadsPerBlock >>> (GPUImageData, tmp, width, height, gpu_k, kerWidth, kerHeight);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bfKernel!\n", cudaStatus);
		goto Error;
	}

	renderTime = render.TimeSinceLastCall();

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bfKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}



	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy((void*)dst.data, tmp, size_of_image, cudaMemcpyDeviceToHost);
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

double cpuBoxFilter(uchar* src, uchar*& dst, int width, int height, int* k, int kerWidth, int kerHeight, uchar* tmp) {

	HighPrecisionTime filter;
	double filterTime = 0.0;
	
	//the sum of the prox pixels after filtering
	float currentSum = 0.0f;

	float avgNum = 0.0f;

	for (int it = 0; it < (kerWidth * kerHeight); it++) {
		avgNum += k[it];
	}

	//don't divide by 0 down below
	if (avgNum == 0) {
		avgNum = 1;
	}

	//start on pixel
	//find the pixels in proximity
	//multiply their b/w values by corresponding filter val
	//sum all prox pixels
	//divide by sum of vals in kernel array
	//save new value in dst.data[i]
	//move to next pixel, continue

	//the direction one needs to move in order to get to the desired border pixel relative to the current pixel
	//NOTE: generalize this using the dimensions of the kernel
	//---> divide matrix width by 2, subtract 1. That number becomes the coefficient on width, as well as the constant 
	//----> -(coeff*width + coeff) ...coeff--
	//----->THIS ASSUMES 1) odd width 2) equal width & height

	int coeff = kerWidth / 2;

	filter.TimeSinceLastCall();

	//roll through every pixel in the image using this loop
	for (int i = (coeff * width) + coeff; i < ((width * height) - (coeff * width) - coeff); i++) {

		currentSum = 0.0;

		//if edge pixel -> left edge, right edge
		if (i % width < coeff  || (i + coeff) % width < coeff) {
			continue;
		}

		//roll through each kernel corresponding pixel surrounding the current pixel (i)

		int j = 0;

		for (int m = -coeff; m <= coeff; m++) {
			for (int c = -coeff; c <= coeff; c++) {
				//i is the current pixel. We find the pixels around i by multiplying the number of widths away the desired modifying pixel is from origin by width, plus the number of pixels horizontally
				currentSum += src[i + (m * width + c)] * k[j];
				j++;
			}
		}

		dst[i] = (uchar)abs((currentSum / avgNum));

	}


	filterTime = filter.TimeSinceLastCall();

	return filterTime;
}