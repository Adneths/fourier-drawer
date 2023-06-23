#pragma once

#include <complex>
#include "constant.h"
#include "core.h"
#include "FourierSeries.h"
#include "LineStrip.h"
#include "Lines.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>


class CudaFourierSeries : public FourierSeries {
public:
	CudaFourierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize);
	~CudaFourierSeries();
	float increment(size_t count, float time) override;
	void updateBuffers() override;
	void readyBuffers() override;
private:
	size_t head;
	size_t cacheSize, size;
	float dt, time;
	LineStrip* vectorLine;
	Lines* pathLine;

	cudaGraphicsResource *vectorPtr, *pathPtr;
	float *deviceMags, *devicePathCache;
	int* deviceFreqs;

	size_t lineWidth, pathBufferSize;
};

extern "C" DLL_API FourierSeries* __cdecl instantiate(LineStrip * vectorLine, Lines * pathLine, std::complex<float>*mags, int* freqs, size_t size, float dt, size_t cacheSize);