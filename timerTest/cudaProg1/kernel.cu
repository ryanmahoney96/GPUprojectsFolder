
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>
//#include <string>

//use alt+b, u to build only this project

using namespace std;

typedef int ourVar_t;

bool allocMemory(ourVar_t** a, ourVar_t** b, ourVar_t** c, int size, int size_of_var = sizeof(ourVar_t));
void freeMemory(ourVar_t* a, ourVar_t* b, ourVar_t* c);

int main(int argc, char* argv[]) {

	cout << endl;

	srand(time(NULL));

	int size_of_array = 5;

	//press alt+shift+(arrow key) to vertical edit
	ourVar_t* a = nullptr;
	ourVar_t* b = nullptr;
	ourVar_t* c = nullptr;
	
	try {

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
	
		
		for (int i = 0; i < size_of_array; i++) {
			a[i] = rand() % 300 + 300;
			b[i] = rand() % 300;
			c[i] = 0;
		}

		cout << "A: " << a[0] << endl;
		cout << "B: " << b[0] << endl;
		cout << "C: " << c[0] << endl;

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