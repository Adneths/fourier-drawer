#include "CudaFourierSeries.cuh"

#include "core.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>

__inline__ __device__ float2 warpAllReduceSum(float2 val) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask /= 2)
	{
		val.x += __shfl_xor(val.x, mask);
		val.y += __shfl_xor(val.y, mask);
	}
	return val;
}
__inline__ __device__ float2 blockReduceSum(float2 val, int resultantWarp) {

	static __shared__ float2 shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpAllReduceSum(val);
	if (lane == 0) shared[wid] = val;

	__syncthreads();

	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float2(0,0);
	if (wid == resultantWarp) val = warpAllReduceSum(val);

	return val;
}

__global__ void sumVector(float* mags, float* pathPtr, size_t size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size)
		return;

	float2 v = { mags[id * 2], mags[id * 2 + 1] };
	float2 val = blockReduceSum(v, 0);
	if (id == 0)
	{
		atomicAdd(&pathPtr[id * 2], val.x);
		atomicAdd(&pathPtr[id * 2 + 1], val.y);
	}
}

__global__ void cudaIncrement(float* mags, int* freqs, float* pathCache, size_t size, float dt, size_t count)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	//int lane = threadIdx.x % warpSize;
	if (id >= size)
		return;

	float2 v = { mags[id * 2], mags[id * 2 + 1] };
	float2 s = { cos(dt * freqs[id]), sin(dt * freqs[id]) };

	float2 psum = make_float2(0,0);
	for (int i = 0; i < count; i++)
	{
		v = { v.x * s.x - v.y * s.y, v.x * s.y + v.y * s.x };
		float2 val = blockReduceSum(v, i >> 5);
		if (id == i)
			psum = val;
	}
	mags[id * 2] = v.x;
	mags[id * 2 + 1] = v.y;

	if (id < count)
	{
		atomicAdd(&pathCache[id * 2], psum.x);
		atomicAdd(&pathCache[id * 2 + 1], psum.y);
	}
}

#define INCREMENT_BLOCK_SIZE 1024
CudaFourierSeries::CudaFourierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize)
	: vectorLine(vectorLine), pathLine(pathLine), dt(dt), cacheSize(cacheSize), size(size), time(0), head(0)
{
	cudaMalloc(&deviceMags, sizeof(float) * size * 2ull);
	cudaMemcpy(deviceMags, (float*)mags, sizeof(float) * size * 2ull, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceFreqs, sizeof(int) * size);
	cudaMemcpy(deviceFreqs, freqs, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&devicePathCache, sizeof(float) * cacheSize * 2ull);
	cudaGraphicsGLRegisterBuffer(&vectorPtr, vectorLine->getBuffer(), cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&pathPtr, pathLine->getBuffer(), cudaGraphicsRegisterFlagsNone);

	float* deviceStart;
	cudaMalloc(&deviceStart, sizeof(float) * 2ull);
	sumVector<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
			(deviceMags, deviceStart, size);
	float hostStart[3] = { 0 };
	cudaMemcpy(hostStart, deviceStart, sizeof(float) * 2, cudaMemcpyDeviceToHost);
	if (pathLine->isTimestamped())
		glClearBufferData(GL_ARRAY_BUFFER, GL_RGB32F, GL_RGBA, GL_FLOAT, &hostStart);
	else
		glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &hostStart);

	lineWidth = (pathLine->isTimestamped() ? 6ull : 4ull);
	pathBufferSize = pathLine->getCount() * lineWidth;
}
CudaFourierSeries::~CudaFourierSeries()
{
	cudaFree(deviceMags);
	cudaFree(deviceFreqs);
	cudaFree(devicePathCache);
	cudaGraphicsUnregisterResource(vectorPtr);
	cudaGraphicsUnregisterResource(pathPtr);
}

float CudaFourierSeries::increment(size_t count, float time)
{
	cudaMemset(devicePathCache, 0, sizeof(float) * cacheSize * 2ull);
	cudaIncrement<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
			(deviceMags, deviceFreqs, devicePathCache, size, dt, count);
	this->time += count * dt;
	return count * dt;
}

__global__ void fillVector(float* mags, size_t len, float* vector)
{

}
__global__ void fillPath(float* pathCache, size_t cacheLen, float* path, size_t pathLen, size_t head)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * 2 >= cacheLen)
		return;
	
	int index = (head + id * 4 + 2) % pathLen;
	path[index] = pathCache[id * 2];
	path[index + 1] = pathCache[id * 2 + 1];

	index = (index + 2) % pathLen;
	path[index] = pathCache[id * 2];
	path[index + 1] = pathCache[id * 2 + 1];
	
	if (id * 2 == cacheLen - 2)
	{
		index = (index + 2) % pathLen;
		path[index] = pathCache[id * 2];
		path[index + 1] = pathCache[id * 2 + 1];
	}
}
__global__ void fillPathTimestamped(float* pathCache, size_t cacheLen, float* path, size_t pathLen, float time, float dt, size_t head)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * 2 >= cacheLen)
		return;

	int index = (head + id * 6 + 3) % pathLen;
	path[index] = pathCache[id * 2];
	path[index + 1] = pathCache[id * 2 + 1];
	path[index + 2] = time + id * dt;

	index = (index + 3) % pathLen;
	path[index] = pathCache[id * 2];
	path[index + 1] = pathCache[id * 2 + 1];
	path[index + 2] = time + id * dt;
	
	if (id * 2 == cacheLen - 2)
	{
		index = (index + 3) % pathLen;
		path[index] = pathCache[id * 2];
		path[index + 1] = pathCache[id * 2 + 1];
		path[index + 2] = time + id * dt;
	}
}
#define CACHE_BLOCK_SIZE 64
void CudaFourierSeries::updateBuffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	float* ptr;
	size_t mappedSize = (size + 1) * 2 * sizeof(float);
	/*cudaGraphicsMapResources(1, &vectorPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, vectorPtr);
	cudaGraphicsUnmapResources(1, &vectorPtr);
	*/

	mappedSize = sizeof(float) * lineWidth * pathLine->getCount();
	cudaGraphicsMapResources(1, &pathPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, pathPtr);
	if (pathLine->isTimestamped())
		fillPathTimestamped<<<(cacheSize + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE>>>
		(devicePathCache, cacheSize * 2, ptr, mappedSize, time - dt * cacheSize, dt, head);
	else
		fillPath<<<(cacheSize + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE >>>
		(devicePathCache, cacheSize * 2, ptr, mappedSize, head);
	cudaGraphicsUnmapResources(1, &pathPtr);

	head = (head + lineWidth * cacheSize) % (pathBufferSize);
}

void CudaFourierSeries::readyBuffers()
{
	cudaDeviceSynchronize();
}

DLL_API FourierSeries* __cdecl instantiate(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize)
{
	return new CudaFourierSeries(vectorLine, pathLine, mags, freqs, size, dt, cacheSize);
}
