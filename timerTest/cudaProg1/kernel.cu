
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

bool allocMemory(ourVar_t** a, ourVar_t** b, ourVar_t** c, int size, int size_of_var = sizeof(ourVar_t));
void freeMemory(ourVar_t* a, ourVar_t* b, ourVar_t* c);

//double fillArrays(ourVar_t* a, ourVar_t* b, ourVar_t* c, int size_of_array, int iterations);
void fillArray(ourVar_t* a, int size_of_array);
void fill_C_Array(ourVar_t* a, int size_of_array);

void printArrays(ourVar_t* a, ourVar_t* b, ourVar_t* c, int size_of_array);

const int max_num = 15;

int main(int argc, char* argv[]) {

	cout << endl;

	srand(time(NULL));

	int size_of_array = 5;

	//press alt+shift+(arrow key) to vertical edit
	ourVar_t* a = nullptr;
	ourVar_t* b = nullptr;
	ourVar_t* c = nullptr;
	
	try {

		HighPrecisionTime htp;
		cudaError_t cudaStatus;

		double htp_ret = 0.0;
		int iterations = 100;

		//if there is a command line argument, set the ouput variable to it
		if (argc > 1) {
			size_of_array = atoi(argv[1]);
		}
		else {
			cout << "No Commmand Line Argument: Size of Array Defaulting to 5" << endl;
		}

		cout << "argc: " << argc << "\nargv: " << size_of_array << endl;

		if (!allocMemory(&a, &b, &c, size_of_array)) {
			throw("Error Allocating Memory");
		}

		htp.TimeSinceLastCall();

#pragma omp parallel for
		for (int i = 0; i < iterations; i++) {

			fillArray(a, size_of_array);
			fillArray(b, size_of_array);
			fill_C_Array(c, size_of_array);

			htp_ret += htp.TimeSinceLastCall();

		}

		//htp_ret = fillArrays(a, b, c, size_of_array, iterations);

		//average here
		htp_ret = htp_ret / iterations;

		/*htp.TimeSinceLastCall();

		for (int i = 0; i < size_of_array; i++) {
			c[i] = a[i] + b[i];
		}*/

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			throw("cudaDeviceReset failed!");
		}
		
		printArrays(a, b, c, size_of_array);

		cout << "The average run was: " << htp_ret << endl;


	}

	catch (char * err) {
		cerr << err << endl;
	}

	freeMemory(a, b, c);

#ifdef _WIN32 || _WIN64
	//system("pause");
#endif
	
	return 0;
}

bool allocMemory(ourVar_t** a, ourVar_t** b, ourVar_t** c, int size, int size_of_var) {

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

void freeMemory(ourVar_t* a, ourVar_t* b, ourVar_t* c) {
	
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

//double fillArrays(ourVar_t* a, ourVar_t* b, ourVar_t* c, int size_of_array, int iterations) {
//
//	double ret = 0.0;
//	HighPrecisionTime htp;
//
//	//"start" the timer
//	htp.TimeSinceLastCall();
//
//	//using omp to use as many cores as possible
//#pragma omp parallel for
//	for (int i = 0; i < iterations; i++) {
//
//		//optimized version of for loop: use a while loop with pointers
//		
//		while (a < &a[size_of_array]) {
//			*a = rand();
//			*b = rand();
//			*c = 0;
//			++a;
//			++b;
//			++c;
//		}
//
//		//take the time after each fill (averaged afterward)
//		ret += htp.TimeSinceLastCall();
//
//	}
//
//	//using omp to use as many cores as possible
//#pragma omp parallel for
//	for (int i = 0; i < iterations; i++) {
//
//		for (int j = 0; j < size_of_array; j++) {
//
//			a[j] = rand();
//			b[j] = rand();
//			c[j] = 0;
//		}
//
//		//take the time after each fill (averaged afterward)
//		ret += htp.TimeSinceLastCall();
//
//	}
//
//	return ret;
//}

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