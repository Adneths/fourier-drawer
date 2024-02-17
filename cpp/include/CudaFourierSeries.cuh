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
	CudaFourierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize, int gpu, bool info);
	~CudaFourierSeries();
	float increment(size_t count, float time) override;
	void updateBuffers(glm::vec2* vecHeadPtr = nullptr) override;
	void readyBuffers() override;
	void resetTrail() override;
	void init(float time) override;
private:
	bool invalid = false;

	size_t head;
	size_t cacheSize, size;
	float dt, time;
	LineStrip* vectorLine;
	Lines* pathLine;

	cudaGraphicsResource *vectorPtr, *pathPtr;
	float *deviceMags, *devicePathCache;
	int* deviceFreqs;

	size_t lineWidth, pathBufferSize;

	float* deviceBlocks;
	void cumsum2f(float* in, float* out, size_t len);
	void cumsum2f(float* in, float* out, size_t len, size_t offset);
};

extern "C" DLL_API FourierSeries* __cdecl instantiate(LineStrip * vectorLine, Lines * pathLine, std::complex<float>*mags, int* freqs, size_t size, float dt, size_t cacheSize);