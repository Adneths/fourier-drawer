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
		val.x += __shfl_xor_sync(0xffffffff, val.x, mask);
		val.y += __shfl_xor_sync(0xffffffff, val.y, mask);
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

	if (wid == resultantWarp) {
		val = (lane < blockDim.x / warpSize) ? shared[lane] : make_float2(0, 0);
		val = warpAllReduceSum(val);
	}
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
	int tx = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tx;
	if (id >= size)
		return;

	float2 v, s;
	if (id < size)
	{
		v = { mags[id * 2], mags[id * 2 + 1] };
		s = { cos(dt * freqs[id]), sin(dt * freqs[id]) };
	}
	else
	{
		v = { 0,0 };
		s = { 0,0 };
	}

	float2 psum = make_float2(0,0);
	for (int i = 0; i < count; i++)
	{
		v = { v.x * s.x - v.y * s.y, v.x * s.y + v.y * s.x };
		float2 val = blockReduceSum(v, i >> 5);
		if (tx == i)
			psum = val;
	}
	if (id < size)
	{
		mags[id * 2] = v.x;
		mags[id * 2 + 1] = v.y;
	}

	if (tx < count)
	{
		atomicAdd(&pathCache[tx * 2], psum.x);
		atomicAdd(&pathCache[tx * 2 + 1], psum.y);
	}
}
__global__ void cudaIncrement1024(float* mags, int* freqs, float* pathCache, size_t size, float dt, size_t count)
{
	int tx = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tx;

	float2 v, s;
	if (id < size)
	{
		v = { mags[id * 2], mags[id * 2 + 1] };
		s = { cos(dt * freqs[id]), sin(dt * freqs[id]) };
	}
	else
	{
		v = { 0, 0 };
		s = { 0, 0 };
	}

	float2 psum = make_float2(0, 0);
	for (int i = 0; i < count; i++)
	{
		v = { v.x * s.x - v.y * s.y, v.x * s.y + v.y * s.x };
		float2 val = blockReduceSum(v, i >> 5);
		if (tx == i)
			psum = val;
	}
	if (id < size)
	{
		mags[id * 2] = v.x;
		mags[id * 2 + 1] = v.y;
	}

	if (tx < count)
	{
		atomicAdd(&pathCache[tx * 2], psum.x);
		atomicAdd(&pathCache[tx * 2 + 1], psum.y);
	}
}

#define CACHE_BLOCK_SIZE 64
#define INCREMENT_BLOCK_SIZE 1024
#define CUMSUM_BLOCK_SIZE 1024
__global__ void cudaCumsum2f(float2* in, float2* out, float2* blocks, int len) {
	__shared__ float2 sBlock[CUMSUM_BLOCK_SIZE];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	if (id < len)
		sBlock[tx] = in[id];
	else
		sBlock[tx] = { 0.0f, 0.0f };
	__syncthreads();

	for (int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index].x += sBlock[index - stride].x;
			sBlock[index].y += sBlock[index - stride].y;
		}
		__syncthreads();
	}

	for (int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index + stride < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index + stride].x += sBlock[index].x;
			sBlock[index + stride].y += sBlock[index].y;
		}
		__syncthreads();
	}

	if (id < len)
		out[id] = sBlock[tx];
	if (tx == 0)
		blocks[blockIdx.x] = sBlock[CUMSUM_BLOCK_SIZE - 1];
}
__global__ void cudaCumsum2f(float2* in, float2* out, int len) {
	__shared__ float2 sBlock[CUMSUM_BLOCK_SIZE];

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int tx = threadIdx.x;

	if (id < len)
		sBlock[tx] = in[id];
	else
		sBlock[tx] = { 0.0f, 0.0f };
	__syncthreads();

	for (int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index].x += sBlock[index - stride].x;
			sBlock[index].y += sBlock[index - stride].y;
		}
		__syncthreads();
	}

	for (int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		int index = (tx + 1) * stride * 2 - 1;
		if (index + stride < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index + stride].x += sBlock[index].x;
			sBlock[index + stride].y += sBlock[index].y;
		}
		__syncthreads();
	}

	if (id < len)
		out[id] = sBlock[tx];
}
__global__ void cudaFillSum2f(float2* inout, float2* blocks, size_t len) {
	__shared__ float2 sum;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIdx.x == 0)
		if (blockIdx.x > 0)
			sum = blocks[blockIdx.x-1];
		else
			sum = { 0.0f, 0.0f };
	__syncthreads();

	if (id < len)
	{
		inout[id].x += sum.x;
		inout[id].y += sum.y;
	}
}
void cumsum2f(float* in, float* out, size_t len) {
	if (len > CUMSUM_BLOCK_SIZE)
	{
		size_t blockDim = (len + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
		float* blocks;
		cudaMalloc(&blocks, sizeof(float) * blockDim * 2);
		cudaCumsum2f<<<blockDim, CUMSUM_BLOCK_SIZE>>>((float2*)in, (float2*)out, (float2*)blocks, len);
		cumsum2f(blocks, blocks, blockDim);
		cudaFillSum2f<<<blockDim, CUMSUM_BLOCK_SIZE>>>((float2*)out, (float2*)blocks, len);
		cudaFree(blocks);
	}
	else
		cudaCumsum2f<<<1, CUMSUM_BLOCK_SIZE>>>((float2*)in, (float2*)out, len);
}

void CudaFourierSeries::resetTrail()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	float* deviceStart;
	cudaMalloc(&deviceStart, sizeof(float) * 2ull);
	cudaMemset(deviceStart, 0, sizeof(float) * 2ull);
	sumVector<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
		(deviceMags, deviceStart, size);
	float hostStart[3] = { 0 };
	cudaMemcpy(hostStart, deviceStart, sizeof(float) * 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(devicePathCache + (cacheSize - 1) * 2, deviceStart, sizeof(float) * 2, cudaMemcpyDeviceToDevice);
	cudaFree(deviceStart);

	glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	if (pathLine->isTimestamped())
		glClearBufferData(GL_ARRAY_BUFFER, GL_RGB32F, GL_RGBA, GL_FLOAT, &hostStart);
	else
		glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &hostStart);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
CudaFourierSeries::CudaFourierSeries(LineStrip* vectorLine, Lines* pathLine, std::complex<float>* mags, int* freqs, size_t size, float dt, size_t cacheSize, int gpu, bool info)
	: vectorLine(vectorLine), pathLine(pathLine), cacheSize(cacheSize), size(size), dt(dt), time(0), head(0)
{
	cudaError_t err;
	cudaDeviceProp deviceProperties;
	if (err = cudaGetDeviceProperties(&deviceProperties, gpu))
	{
		printf("%s\n", cudaGetErrorString(err));
		invalid = true;
		return;
	}
	if (deviceProperties.major >= CUDA_MINIMUM_MAJOR_VERSION
		&& deviceProperties.minor >= CUDA_MINIMUM_MINOR_VERSION)
	{
		cudaSetDevice(gpu);
		if(info)
			printf("Using %s - Compute Capability %d.%d\n", deviceProperties.name, deviceProperties.major, deviceProperties.minor);
	}
	else
	{
		printf("Requires minimum compute capability of %d.%d found %d.%d\n", CUDA_MINIMUM_MAJOR_VERSION, CUDA_MINIMUM_MINOR_VERSION, deviceProperties.major, deviceProperties.minor);
		invalid = true;
		return;
	}

	cudaMalloc(&deviceMags, sizeof(float) * size * 2ull);
	cudaMemcpy(deviceMags, (float*)mags, sizeof(float) * size * 2ull, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceFreqs, sizeof(int) * size);
	cudaMemcpy(deviceFreqs, freqs, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&devicePathCache, sizeof(float) * cacheSize * 2ull);
	cudaGraphicsGLRegisterBuffer(&vectorPtr, vectorLine->getBuffer(), cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&pathPtr, pathLine->getBuffer(), cudaGraphicsRegisterFlagsNone);

	float* ptr;
	size_t mappedSize = (size + 1) * 2 * sizeof(float);
	cudaGraphicsMapResources(1, &vectorPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, vectorPtr);
	cumsum2f(deviceMags, ptr + 2, size);
	cudaGraphicsUnmapResources(1, &vectorPtr);

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


void CudaFourierSeries::init(float time) {
	cudaIncrement<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
		(deviceMags, deviceFreqs, devicePathCache, size, time, 1);
	this->time = time;
}
float CudaFourierSeries::increment(size_t count, float time)
{
	cudaMemset(devicePathCache, 0, sizeof(float) * cacheSize * 2ull);
	if (size < count)
	{
		//for (int i = 0; i < count; i += 1024)
		//	cudaIncrement1024<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
		//		(deviceMags, deviceFreqs, devicePathCache + i * 2, size, dt, std::min(1024ull, count - i));
	}
	else
	{
		//for (int i = 0; i < count; i += 1024)
		//	cudaIncrement<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
		//		(deviceMags, deviceFreqs, devicePathCache + i * 2, size, dt, std::min(1024ull, count - i));
		cudaIncrement<<<(size + INCREMENT_BLOCK_SIZE - 1) / INCREMENT_BLOCK_SIZE, INCREMENT_BLOCK_SIZE>>>
			(deviceMags, deviceFreqs, devicePathCache, size, dt, count);
	}
	this->time = time;
	return count * dt;
}

__global__ void fillPath(float* pathCache, size_t cacheLen, float* path, size_t pathLen, size_t head)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * 2 >= cacheLen)
		return;

	register float2 v = { pathCache[id * 2] , pathCache[id * 2 + 1] };

	int index = (head + id * 4 + 2) % pathLen;
	path[index] = v.x;
	path[index + 1] = v.y;

	index = (index + 2) % pathLen;
	path[index] = v.x;
	path[index + 1] = v.y;
	
	if (id * 2 == cacheLen - 2)
	{
		index = (index + 2) % pathLen;
		path[index] = v.x;
		path[index + 1] = v.y;
	}
}
__global__ void fillPathTimestamped(float* pathCache, size_t cacheLen, float* path, size_t pathLen, float time, float dt, size_t head)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * 2 >= cacheLen)
		return;

	register float3 v = { pathCache[id * 2] , pathCache[id * 2 + 1],  time + (id + 1) * dt};

	int index = (head + id * 6 + 3) % pathLen;
	path[index] = v.x;
	path[index + 1] = v.y;
	path[index + 2] = v.z;

	index = (index + 3) % pathLen;
	path[index] = v.x;
	path[index + 1] = v.y;
	path[index + 2] = v.z;
	
	if (id * 2 == cacheLen - 2)
	{
		index = (index + 3) % pathLen;
		path[index] = v.x;
		path[index + 1] = v.y;
		path[index + 2] = v.z;
	}
}

void CudaFourierSeries::updateBuffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	float* ptr;
	size_t mappedSize = (size + 1) * 2 * sizeof(float);
	cudaGraphicsMapResources(1, &vectorPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, vectorPtr);
	cumsum2f(deviceMags, ptr + 2, size);
	cudaGraphicsUnmapResources(1, &vectorPtr);

	mappedSize = sizeof(float) * lineWidth * pathLine->getCount();
	cudaGraphicsMapResources(1, &pathPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, pathPtr);
	if (pathLine->isTimestamped())
		fillPathTimestamped<<<(cacheSize + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE>>>
		(devicePathCache, cacheSize * 2, ptr, lineWidth * pathLine->getCount(), this->time, dt, head);
	else
		fillPath<<<(cacheSize + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE >>>
		(devicePathCache, cacheSize * 2, ptr, lineWidth * pathLine->getCount(), head);
	cudaGraphicsUnmapResources(1, &pathPtr);

	head = (head + lineWidth * cacheSize) % (pathBufferSize);
}

void CudaFourierSeries::readyBuffers(glm::vec2* vecHeadPtr)
{
	if (vecHeadPtr != nullptr)
		cudaMemcpy(vecHeadPtr, devicePathCache + (cacheSize - 1) * 2, sizeof(float) * 2, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}
