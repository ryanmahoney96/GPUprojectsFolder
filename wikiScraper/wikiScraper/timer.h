
#pragma once

#ifndef	hptimer_H
#define	hptimer_H

#if	defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

class hptimer
{
public:
	hptimer();
	double TimeSinceLastCall();
	double TotalTime();

private:

#if	defined(_WIN32) || defined(_WIN64)
	LARGE_INTEGER initializationTicks;
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER previousTicks;
#endif

};

#endif