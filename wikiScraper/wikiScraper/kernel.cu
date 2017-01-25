
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "timer.h"

#include <iostream>
#include <fstream>

using namespace std;

const char* filepath = "C:/Users/educ/Documents/enwiki-latest-abstract.xml";

//giga = 1 << 30;
constexpr size_t GIGA = 1 << 30;
constexpr size_t BMSize = GIGA / 8;

int main()
{
   
	hptimer open;
	double openTime;

	char* fileBuffer = nullptr;

	open.TimeSinceLastCall();

	ifstream bigFile(filepath);

	try {
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
