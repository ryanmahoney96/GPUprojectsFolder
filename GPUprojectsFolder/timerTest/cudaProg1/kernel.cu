
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../highPerformanceTimer/highPerformanceTimer.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
//#include <string>

//use alt+b, u to build only this project

using namespace std;

typedef int ourVar_t;

void CPUTest();

bool allocCPUMemory(ourVar_t** a, ourVar_t** b, ourVar_t** c, int size, int size_of_var = sizeof(ourVar_t));
void freeCPUMemory(ourVar_t* a, ourVar_t* b, ourVar_t* c);

//double fillArrays(ourVar_t* a, ourVar_t* b, ourVar_t* c, int size_of_array, int iterations);
void fillArray(ourVar_t* a, int size_of_array);
void fill_C_Array(ourVar_t* a, int size_of_array);

void printArrays(ourVar_t* a, ourVar_t* b, ourVar_t* c, int size_of_array);

cudaError_t GPUTest();

__global__ void addKernel(ourVar_t *c, ourVar_t *a, ourVar_t *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int i = threadIdx.x;

	c[i] = a[i] + b[i];
}

int size_of_array = 5;
int iterations = 100;

const int max_num = 15;

int main(int argc, char* argv[]) {

	cout << endl;

	srand(time(NULL));

	//addKernel, The grid size, the block size, the vecs to add
	//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
	//int i = blockIDx.x * blockDimx.x + threadIdx.x;
	
	try {

		//if there is a command line argument, set the ouput variable to it
		if (argc > 1) {
			size_of_array = atoi(argv[1]);
			cout << "The size of the array is " << size_of_array << endl;


			if (argc > 2) {
				iterations = atoi(argv[2]);
				cout << "The number of iterations is " << iterations << endl;
			}
			else {
				cout << "The number of iterations is 100" << endl;
			}
		}
		else {
			cout << "No Commmand Line Argument: Size of Array Defaulting to 5, Number of Iterations deafulting to 100" << endl;
		}
		
		cout << endl;
		
		CPUTest();
		GPUTest();

	}

	catch (char * err) {
		cerr << err << endl;
	}

#ifdef _WIN32 || _WIN64
	//system("pause");
#endif
	
	return 0;
}


void CPUTest() {

	//press alt+shift+(arrow key) to vertical edit
	ourVar_t* a = nullptr;
	ourVar_t* b = nullptr;
	ourVar_t* c = nullptr;

	HighPrecisionTime full;
	HighPrecisionTime avg;
	HighPrecisionTime summing;

	double fullTime = 0.0;
	double avgTime = 0.0;
	double sumTime = 0.0;

	if (!allocCPUMemory(&a, &b, &c, size_of_array)) {
		throw("Error Allocating Memory");
	}


	avg.TimeSinceLastCall();

#pragma omp parallel for
	for (int i = 0; i < iterations; i++) {

		fillArray(a, size_of_array);
		fillArray(b, size_of_array);
		fill_C_Array(c, size_of_array);

		avgTime += avg.TimeSinceLastCall();

	}

	full.TimeSinceLastCall();

	summing.TimeSinceLastCall();

	for (int times = 0; times < iterations; times++) {
		
		for (int i = 0; i < size_of_array; i++) {
			c[i] = a[i] + b[i];
		}

		sumTime += summing.TimeSinceLastCall();
	}

	//printArrays(a, b, c, size_of_array);

	freeCPUMemory(a, b, c);

	fullTime = full.TimeSinceLastCall();

	//average here
	avgTime = avgTime / iterations;
	sumTime = sumTime / iterations;
	
	cout << "The average time for the CPU to fill all three vectors was: " << avgTime << endl;
	cout << "   ------------------------   " << endl << endl;

	cout << "The average time it took the CPU to sum the vectors was: " << sumTime << endl;
	cout << "The full CPU run (summing + freeing) was: " << fullTime << endl;

	cout << endl;
}



bool allocCPUMemory(ourVar_t** a, ourVar_t** b, ourVar_t** c, int size, int size_of_var) {

	bool retVal = true;

	int memSize = size * size_of_var;

	*a = (ourVar_t*)malloc(memSize);
	*b = (ourVar_t*)malloc(memSize);
	*c = (ourVar_t*)malloc(memSize);

	if (*a == nullptr || *b == nullptr || *c == nullptr) {
		retVal = false;
	}

	return retVal;
}

void freeCPUMemory(ourVar_t* a, ourVar_t* b, ourVar_t* c) {
	
	if (a != nullptr) {
		free(a);
	}
	if (b != nullptr) {
		free(b);
	}
	if (c != nullptr) {
		free(c);
	}
}

void fillArray(ourVar_t* a, int size_of_array) {

	ourVar_t* pa = a;

	while (pa < &a[size_of_array]) {
		*pa = rand() % max_num;

		++pa;

	}

}

void fill_C_Array(ourVar_t* a, int size_of_array) {

	ourVar_t* pa = a;

	while (pa < &a[size_of_array]) {
		*pa = 0;

		++pa;

	}

}

void printArrays(ourVar_t* a, ourVar_t* b, ourVar_t* c, int size_of_array) {

	for (int i = 0; i < size_of_array; i++) {

		cout << a[i] << " : " << b[i] << " : " << c[i] << endl;

	}

}

cudaError_t GPUTest() {

	ourVar_t* a = nullptr;
	ourVar_t* b = nullptr;
	ourVar_t* c = nullptr;

	if (!allocCPUMemory(&a, &b, &c, size_of_array)) {
		throw("Error Allocating Memory");
	}

#pragma omp parallel for
	for (int i = 0; i < iterations; i++) {

		fillArray(a, size_of_array);
		fillArray(b, size_of_array);
		fill_C_Array(c, size_of_array);

	}

	HighPrecisionTime full;
	HighPrecisionTime summing;

	double fullTime = 0.0;
	double sumTime = 0.0;
	
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int maxThreadsPerBlock = prop.maxThreadsPerBlock;
	int numberOfBlocks = size_of_array / maxThreadsPerBlock + 1;
	
	full.TimeSinceLastCall();

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		throw("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size_of_array * sizeof(ourVar_t));
	if (cudaStatus != cudaSuccess) {
		throw("cudaMalloc-1 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size_of_array * sizeof(ourVar_t));
	if (cudaStatus != cudaSuccess) {
		throw("cudaMalloc-2 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size_of_array * sizeof(ourVar_t));
	if (cudaStatus != cudaSuccess) {
		throw("cudaMalloc-3 failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size_of_array * sizeof(ourVar_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy-1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size_of_array * sizeof(ourVar_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy-2 failed!");
		goto Error;
	}

	summing.TimeSinceLastCall();

	// Launch a kernel on the GPU with one thread for each element.
	//addKernel, The grid size, the block size, the vecs to add
	for (int i = 0; i < iterations; i++) {
		addKernel <<< numberOfBlocks, maxThreadsPerBlock >>> (dev_c, dev_a, dev_b);
		sumTime += summing.TimeSinceLastCall();
	}
	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size_of_array * sizeof(ourVar_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		throw("cudaMemcpy-3 failed!");
		goto Error;
	}

	fullTime = full.TimeSinceLastCall();
	sumTime = sumTime / iterations;
	
	cout << "The average time it took the GPU to sum the vectors was: " << sumTime << endl;
	cout << "The full GPU run (allocating + copying + summing) was: " << fullTime << endl;


Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}