
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "timer.h"

using namespace std;

#define USE_OMP

#if defined(_DEBUG)
#define GIGA	(1 << 20)
#else
#define GIGA	(1 << 30)
#endif

#define BMSIZE	(GIGA / 8)
#define MAX_PATTERN_LENGTH 256

__constant__ char dev_pattern[MAX_PATTERN_LENGTH];
__constant__ int dev_pattern_size;
__device__ char * dev_buffer = nullptr;
__device__ unsigned char * dev_bitmap = nullptr;

__global__ void SearchGPU_V1(char * buffer, int buffer_size, unsigned char * bitmap, int bitmap_size)
{
	//my impression: use this index to run through the buffer one char at a time
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int pIndex;

	//the pattern index
	for (pIndex = index; pIndex < dev_pattern_size; pIndex++)
	{
		char tmp = *(buffer + index + pIndex);

		if (tmp < 65 || (tmp > 90 && tmp < 97)) {
			break;
		}
		else if (tmp > 64 && tmp < 91) {
			tmp += 32;
		}

		if (tmp != *(dev_pattern + pIndex))
			break;
	}

	//if both of the words we gathered are of the same size, the pattern must match
	if (pIndex == dev_pattern_size)
	{
		int byte_number = index >> 3;
		if (byte_number < bitmap_size)
		{
			int bit_number = index % 8;

			//need atomicity
			//CUDA Atomic functions
			{
				*(bitmap + byte_number) |= (1 << bit_number);
			}
		}
	}
}

int SearchCPU_V1(char * buffer, int buffer_size, char * pattern, int pattern_size, unsigned char * bitmap, int bitmap_size)
{
	int rv = 0;

#if defined(USE_OMP)
#pragma omp parallel for
#endif
	for (int cIndex = 0; cIndex < buffer_size; cIndex++)
	{
		int pIndex;

		for (pIndex = 0; pIndex < pattern_size; pIndex++)
		{
			if (tolower(*(buffer + cIndex + pIndex)) != *(pattern + pIndex))
				break;
		}

		if (pIndex == pattern_size)
		{
			int byte_number = cIndex >> 3;
			if (byte_number < bitmap_size)
			{
				int bit_number = cIndex % 8;
#if defined(USE_OMP)
#pragma omp critical
#endif
				{
					*(bitmap + byte_number) |= (1 << bit_number);
					rv++;
				}
			}
		}
	}
	return rv;
}

/*	CStringToLower() - this function flattens a c string to all
lower case. It marches through memory until a null byte is
found. As such, some may consider this function unsafe.

By flattening the pattern, we can eliminate a tolower in
the search function - a potentially big win.

The original pointer is returned so that the function can be
used in an assignment statement.
*/
char * CStringToLower(char * s)
{
	char * rv = s;

	for (; *s != NULL; s++)
	{
		*s = tolower(*s);
	}
	return rv;
}

inline void CheckCudaAndThrow(cudaError_t t, const string & message)
{
	if (t != cudaSuccess)
		throw message;
}

int main(int argc, char * argv[])
{
	cout.imbue(locale(""));
	ifstream f("C:/Users/educ/Documents/enwiki-latest-abstract.xml");
	hptimer hpt;
	char * hst_buffer = nullptr;
	unsigned char * hst_bm = nullptr;
	unsigned char * chk_bm = nullptr;

#if defined(USE_OMP)
	cout << "OMP enabled on " << omp_get_max_threads() << " threads." << endl;
#endif

	try
	{
		if (argc < 2)
			throw string("First argument must be target string.");

		char * pattern = CStringToLower(argv[1]);
		int pattern_size = strlen(pattern);

		if (!f.is_open())
			throw string("File failed to open");

		hst_buffer = new char[GIGA];
		hst_bm = new unsigned char[BMSIZE]();
		chk_bm = new unsigned char[BMSIZE];

		hpt.TimeSinceLastCall();
		f.read(hst_buffer, GIGA);
		if (!f)
			throw string("Failed to read full buffer.");
		double read_time = hpt.TimeSinceLastCall();
		cout << GIGA << " bytes read from disk in " << read_time << " seconds at " << GIGA / read_time / double(1 << 30) << " GB / second." << endl;

		CheckCudaAndThrow(cudaSetDevice(0), string("cudaSetDevice(0) failed on line ") + to_string(__LINE__));
		CheckCudaAndThrow(cudaMalloc(&dev_buffer, GIGA), string("cudaMalloc failed on line ") + to_string(__LINE__));
		CheckCudaAndThrow(cudaMalloc(&dev_bitmap, BMSIZE), string("cudaMalloc failed on line ") + to_string(__LINE__));
		CheckCudaAndThrow(cudaMemset(dev_bitmap, 0, BMSIZE), string("cudaMemset failed on line ") + to_string(__LINE__));
		CheckCudaAndThrow(cudaMemcpyToSymbol(dev_pattern, pattern, pattern_size, 0), string("cudaMemcpyToSymbol failed on line ") + to_string(__LINE__));
		CheckCudaAndThrow(cudaMemcpyToSymbol(dev_pattern_size, &pattern_size, sizeof(int), 0), string("cudaMemcpyToSymbol failed on line ") + to_string(__LINE__));

		hpt.TimeSinceLastCall();
		CheckCudaAndThrow(cudaMemcpy(dev_buffer, hst_buffer, GIGA, cudaMemcpyHostToDevice), string("cudaMemcpy failed on line ") + to_string(__LINE__));
		double copy_time = hpt.TimeSinceLastCall();
		cout << GIGA << " data bytes copied to GPU in " << copy_time << " seconds at " << GIGA / copy_time / double(1 << 30) << " GB / second." << endl;

		hpt.TimeSinceLastCall();
		int matches_found = SearchCPU_V1(hst_buffer, GIGA, pattern, pattern_size, hst_bm, BMSIZE);
		double time_cpu = hpt.TimeSinceLastCall();
		cout << "SearchCPU_V1 found " << matches_found << " matches in " << time_cpu << " seconds.";
		cout << " Searched " << GIGA / time_cpu / double(1 << 30) << " GB / second." << endl;

		int threads_per_block = 1024;
		dim3 grid(1024, 1024);

		hpt.TimeSinceLastCall();
		SearchGPU_V1 << <grid, threads_per_block >> >(dev_buffer, GIGA, dev_bitmap, BMSIZE);
		CheckCudaAndThrow(cudaGetLastError(), string("kernel launch failed on line ") + to_string(__LINE__));
		CheckCudaAndThrow(cudaDeviceSynchronize(), string("cudaDeviceSynchronize() failed on line ") + to_string(__LINE__));
		double time_gpu = hpt.TimeSinceLastCall();

		CheckCudaAndThrow(cudaMemcpy(chk_bm, dev_bitmap, BMSIZE, cudaMemcpyDeviceToHost), string("cudaMemcpy() failed on line ") + to_string(__LINE__));

		unsigned int * bm_alias = (unsigned int *)chk_bm;
		int match_count = 0;

		for (int i = 0; i < BMSIZE / sizeof(int); i++)
		{
			unsigned int c = 0;
			unsigned int v = *(bm_alias + i);
			for (c = 0; v; c++)
			{
				v &= v - 1;
			}
			match_count += c;
		}

		cout << "SearchGPU_V1 found " << match_count << " matches in " << time_gpu << " seconds.";
		cout << " Searched " << GIGA / time_gpu / double(1 << 30) << " GB / second." << endl;
		cout << endl;
		cout << "Ratio: " << time_cpu / time_gpu << " to 1" << endl;
	}
	catch (string s)
	{
		cout << s << endl;
	}

	if (dev_buffer != nullptr)
		cudaFree(dev_buffer);

	if (dev_bitmap != nullptr)
		cudaFree(dev_bitmap);

	if (hst_buffer != nullptr)
		delete[] hst_buffer;

	if (hst_bm != nullptr)
		delete[] hst_bm;

	if (f.is_open())
		f.close();

	cudaDeviceReset();

#if defined(WIN64) || defined(WIN32)
	cout << endl;
	system("pause");
#endif

	return 0;
}