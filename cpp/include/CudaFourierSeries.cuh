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

using namespace math;

template <typename T, typename T2, typename T3>
class CudaFourierSeries : public FourierSeries<T> {
public:
	CudaFourierSeries(LineStrip<T>* vectorLine, Lines<T>* pathLine, std::complex<T>* mags, int* freqs, size_t size, T dt, size_t cacheSize, int gpu, bool info);
	~CudaFourierSeries();
	T increment(size_t count, T time) override;
	void updateBuffers(vec2<T>* vecHeadPtr = nullptr) override;
	void readyBuffers() override;
	void resetTrail(vec2<T>* vecHeadPtr = nullptr) override;
	void init(T time) override;
private:
	bool invalid = false;

	size_t head;
	size_t cacheSize, size;
	T dt, time;
	LineStrip<T>* vectorLine;
	Lines<T>* pathLine;

	cudaGraphicsResource *vectorPtr, *pathPtr;
	T *deviceMags, *devicePathCache;
	int* deviceFreqs;

	size_t lineWidth, pathBufferSize;

	T* deviceBlocks;
	void cumsum2f(T* in, T* out, size_t len);
	void cumsum2f(T* in, T* out, size_t len, size_t offset);

	size_t incrementBlockSize;
};

//extern "C" DLL_API FourierSeries* __cdecl instantiate(LineStrip * vectorLine, Lines * pathLine, std::complex<float>*mags, int* freqs, size_t size, float dt, size_t cacheSize);