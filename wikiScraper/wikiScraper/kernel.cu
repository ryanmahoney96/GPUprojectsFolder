
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "timer.h"

#include <iostream>
#include <fstream>

using namespace std;

//gigabyte = 1 << 30;

#if defined(_DEBUG)
#define GIGA 1 << 20
#else
#define GIGA 1 << 30
#endif

const char* filepath = "C:/Users/educ/Documents/enwiki-latest-abstract.xml";

constexpr size_t BMSize = GIGA / 8;

int main(int argc, char* argv[])
{
   
	hptimer open;
	double openTime;

	char* fileBuffer = nullptr;

	open.TimeSinceLastCall();

	ifstream bigFile(filepath);

	try {
		if (argc != 2) {

		}

		if (!bigFile.is_open()) {
			throw("Failed to Open File");
		}

		fileBuffer = new char[GIGA]();

		bigFile.read(fileBuffer, GIGA);

		if (!bigFile) {
			throw("Failed to Read File");
		}

		openTime = open.TimeSinceLastCall();

		for (int i = 0; i < 100; i++) {
			
			cout << fileBuffer[i];

		}

		cout << endl << "It took " << openTime << " seconds to open and read from the file" << endl;
	}

	catch (char* err) {
		cout << err << endl;
	}

	if (fileBuffer != nullptr) {
		delete[] fileBuffer;
	}

	if (bigFile.is_open()) {
		bigFile.close();
	}

	system("pause");

    return 0;
}
