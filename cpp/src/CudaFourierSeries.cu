#include "CudaFourierSeries.cuh"
template class CudaFourierSeries<float, float2, float3>;
template class CudaFourierSeries<double, double2, double3>;

#include "core.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>

size_t warpsPerSM(int cudaMajorVersion, int cudaMinorVersion) {
	if (cudaMajorVersion == 8 && cudaMinorVersion != 0)
		return 48;
	if (cudaMajorVersion == 7 && cudaMinorVersion == 5)
		return 32;
	return 64;
}

template <typename T2>
__inline__ __device__ T2 warpAllReduceSum(T2 val) {
#pragma unroll
	for (unsigned int mask = warpSize / 2; mask > 0; mask /= 2)
	{
		val.x += __shfl_xor_sync(0xffffffff, val.x, mask);
		val.y += __shfl_xor_sync(0xffffffff, val.y, mask);
	}
	return val;
}
template <typename T2>
__inline__ __device__ T2 blockReduceSum(T2 val, int resultantWarp) {

	static __shared__ T2 shared[32];
	unsigned int lane = threadIdx.x % warpSize;
	unsigned int wid = threadIdx.x / warpSize;

	val = warpAllReduceSum<T2>(val);
	if (lane == 0) shared[wid] = val;

	__syncthreads();

	if (wid == resultantWarp) {
		val = (lane < blockDim.x / warpSize) ? shared[lane] : T2{0,0};
		val = warpAllReduceSum<T2>(val);
	}
	return val;
}

template <typename T, typename T2, typename T3>
__global__ void sumVector(T2* mags, T2* pathPtr, size_t size)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= size)
		return;

	//float2 v = { mags[id * 2], mags[id * 2 + 1] };
	T2 val = blockReduceSum<T2>(mags[id], 0);
	if (id == 0)
	{
		atomicAdd(&pathPtr[id].x, val.x);
		atomicAdd(&pathPtr[id].y, val.y);
	}
}

template <typename T, typename T2, typename T3>
__global__ void cudaIncrement1024(T2* mags, int* freqs, T* pathCache, size_t size, T dt, size_t count)
{
	int tx = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tx;

	T2 v, s;
	if (id < size)
	{
		//v = { mags[id * 2], mags[id * 2 + 1] };
		v = mags[id];
		s = { cos(dt * freqs[id]), sin(dt * freqs[id]) };
	}
	else
	{
		v = { 0, 0 };
		s = { 0, 0 };
	}

	//float2 psum = make_float2(0, 0);
	T a, b;
	for (unsigned int i = 0; i < count; i++)
	{
		v = { v.x * s.x - v.y * s.y, v.x * s.y + v.y * s.x };
		T2 val = blockReduceSum<T2>(v, i >> 5);
		//if (tx == i) {
		//	psum = val;
		//}
		T f = (tx & 0b1) == 0 ? val.x : val.y;
		if ((tx >> 1) == (i & 0x1ff)) {
			if ((i & 0x200) != 0)
				b = f;
			else
				a = f;
		}
	}
	if (id < size)
	{
		mags[id] = v;
		//mags[id * 2] = v.x;
		//mags[id * 2 + 1] = v.y;
	}

	if (tx < 2 * count)
		atomicAdd(&pathCache[tx], a);
	if (tx < 2 * count - 1024)
		atomicAdd(&pathCache[tx + 1024], b);

	//if (tx < count)
	//{
	//	atomicAdd(&pathCache[tx * 2], psum.x);
	//	atomicAdd(&pathCache[tx * 2 + 1], psum.y);
	//}
}

template <typename T, typename T2, typename T3>
__global__ void fillPath(T* pathCache, size_t cacheLen, T* path, size_t pathLen, size_t head)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * 2 >= cacheLen)
		return;

	register T2 v = { pathCache[id * 2] , pathCache[id * 2 + 1] };

	size_t index = (head + id * 4 + 2) % pathLen;
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
template <typename T, typename T2, typename T3>
__global__ void fillPathTimestamped(T* pathCache, size_t cacheLen, T* path, size_t pathLen, T time, T dt, size_t head)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id * 2 >= cacheLen)
		return;

	register T3 v = { pathCache[id * 2] , pathCache[id * 2 + 1],  time + (id + 1) * dt };

	size_t index = (head + id * 6 + 3) % pathLen;
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

#define CACHE_BLOCK_SIZE 64
//#define INCREMENT_BLOCK_SIZE 768
#define CUMSUM_BLOCK_SIZE 1024
template <typename T, typename T2, typename T3>
__global__ void cudaCumsum2f(T2* in, T2* out, T2* blocks, size_t len) {
	__shared__ T2 sBlock[CUMSUM_BLOCK_SIZE];

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int tx = threadIdx.x;

	if (id < len)
		sBlock[tx] = in[id];
	else
		sBlock[tx] = { 0.0f, 0.0f };
	__syncthreads();

	for (unsigned int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
		size_t index = (tx + 1) * stride * 2 - 1;
		if (index < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index].x += sBlock[index - stride].x;
			sBlock[index].y += sBlock[index - stride].y;
		}
		__syncthreads();
	}

	for (unsigned int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		size_t index = (tx + 1) * stride * 2 - 1;
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
template <typename T, typename T2, typename T3>
__global__ void cudaCumsum2f(T2* in, T2* out, size_t len) {
	__shared__ T2 sBlock[CUMSUM_BLOCK_SIZE];

	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int tx = threadIdx.x;

	if (id < len)
		sBlock[tx] = in[id];
	else
		sBlock[tx] = { 0.0f, 0.0f };
	__syncthreads();

	for (unsigned int stride = 1; stride <= CUMSUM_BLOCK_SIZE; stride *= 2) {
		size_t index = (tx + 1) * stride * 2 - 1;
		if (index < CUMSUM_BLOCK_SIZE)
		{
			sBlock[index].x += sBlock[index - stride].x;
			sBlock[index].y += sBlock[index - stride].y;
		}
		__syncthreads();
	}

	for (unsigned int stride = CUMSUM_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		size_t index = (tx + 1) * stride * 2 - 1;
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
template <typename T, typename T2, typename T3>
__global__ void cudaFillSum2f(T2* inout, T2* blocks, size_t len) {
	__shared__ T2 sum;
	unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
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
template <typename T, typename T2, typename T3>
void CudaFourierSeries<T, T2, T3>::cumsum2f(T* in, T* out, size_t len) {
	cumsum2f(in, out, len, 0);
}
template <typename T, typename T2, typename T3>
void CudaFourierSeries<T, T2, T3>::cumsum2f(T* in, T* out, size_t len, size_t offset) {
	if (len > CUMSUM_BLOCK_SIZE)
	{
		//DONE: Use pre allocated memory based on provided max length instead of malloc/free dynamically
		size_t blockDim = (len + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
		//float* blocks;
		//cudaMalloc(&blocks, sizeof(float) * blockDim * 2);
		cudaCumsum2f<T,T2,T3><<<blockDim, CUMSUM_BLOCK_SIZE>>>((T2*)in, (T2*)out, ((T2*)deviceBlocks) + offset, len);
		cumsum2f(deviceBlocks + 2ull * offset, deviceBlocks + 2ull * offset, blockDim, offset+blockDim);
		cudaFillSum2f<T,T2,T3><<<blockDim, CUMSUM_BLOCK_SIZE>>>((T2*)out, ((T2*)deviceBlocks) + offset, len);
		//cudaFree(blocks);
	}
	else
		cudaCumsum2f<T,T2,T3><<<1, CUMSUM_BLOCK_SIZE>>>((T2*)in, (T2*)out, len);
}

template <typename T, typename T2, typename T3>
void CudaFourierSeries<T, T2, T3>::resetTrail(vec2<T>* vecHeadPtr)
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	T* deviceStart;
	cudaMalloc(&deviceStart, sizeof(T) * 2ull);
	cudaMemset(deviceStart, 0, sizeof(T) * 2ull);
	sumVector<T,T2,T3><<<(size + (this->incrementBlockSize) - 1) / (this->incrementBlockSize), (this->incrementBlockSize) >> >
		((T2*)deviceMags, (T2*)deviceStart, size);
	T hostStart[3] = { 0 };
	cudaMemcpy(hostStart, deviceStart, sizeof(T) * 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(devicePathCache + (cacheSize - 1) * 2, deviceStart, sizeof(T) * 2, cudaMemcpyDeviceToDevice);
	cudaFree(deviceStart);

	pathLine->fill(hostStart);

	if (vecHeadPtr != nullptr)
		*vecHeadPtr = glm::vec2(hostStart[0], hostStart[1]);
}
template <typename T, typename T2, typename T3>
CudaFourierSeries<T, T2, T3>::CudaFourierSeries(LineStrip<T>* vectorLine, Lines<T>* pathLine, std::complex<T>* mags, int* freqs, size_t size, T dt, size_t cacheSize, int gpu, bool info)
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
	incrementBlockSize = warpsPerSM(deviceProperties.major, deviceProperties.minor) == 48 ? 768 : 1024;

	cudaMalloc(&deviceMags, sizeof(T) * size * 2ull);
	cudaMemcpy(deviceMags, (T*)mags, sizeof(T) * size * 2ull, cudaMemcpyHostToDevice);
	cudaMalloc(&deviceFreqs, sizeof(int) * size);
	cudaMemcpy(deviceFreqs, freqs, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMalloc(&devicePathCache, sizeof(T) * cacheSize * 2ull);
	cudaGraphicsGLRegisterBuffer(&vectorPtr, vectorLine->getBuffer(), cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&pathPtr, pathLine->getBuffer(), cudaGraphicsRegisterFlagsNone);

	int blocksSize = 0; int len = size;
	while (len > CUMSUM_BLOCK_SIZE) { blocksSize += len = (len + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE; }
	cudaMalloc(&deviceBlocks, sizeof(T) * blocksSize * 2ull);

	T* ptr;
	size_t mappedSize = (size + 1) * 2 * sizeof(T);
	cudaGraphicsMapResources(1, &vectorPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, vectorPtr);
	cumsum2f(deviceMags, ptr + 2, size);
	cudaGraphicsUnmapResources(1, &vectorPtr);

	lineWidth = (pathLine->isTimestamped() ? 6ull : 4ull);
	pathBufferSize = pathLine->getCount() * lineWidth;
}
template <typename T, typename T2, typename T3>
CudaFourierSeries<T, T2, T3>::~CudaFourierSeries()
{
	cudaFree(deviceMags);
	cudaFree(deviceFreqs);
	cudaFree(devicePathCache);
	cudaFree(deviceBlocks);
	cudaGraphicsUnregisterResource(vectorPtr);
	cudaGraphicsUnregisterResource(pathPtr);
}


template <typename T, typename T2, typename T3>
void CudaFourierSeries<T, T2, T3>::init(T time) {
	cudaIncrement1024<T,T2,T3><<<(size + (this->incrementBlockSize) - 1) / (this->incrementBlockSize), (this->incrementBlockSize) >>>
		((T2*)deviceMags, deviceFreqs, devicePathCache, size, time, 1);
	cudaDeviceSynchronize();
	this->time = time;
}
template <typename T, typename T2, typename T3>
T CudaFourierSeries<T, T2, T3>::increment(size_t count, T time)
{
	cudaMemset(devicePathCache, 0, sizeof(T) * cacheSize * 2ull);
	for (int i = 0; i < count; i += (this->incrementBlockSize))
	{
		cudaIncrement1024<T,T2,T3><<<(size + (this->incrementBlockSize) - 1) / (this->incrementBlockSize), (this->incrementBlockSize) >>>
			((T2*)deviceMags, deviceFreqs, devicePathCache + i * 2, size, dt, std::min(static_cast<size_t>((this->incrementBlockSize)), count - i));
	}
	this->time = time;
	return count * dt;
}

template <typename T, typename T2, typename T3>
void CudaFourierSeries<T, T2, T3>::updateBuffers(vec2<T>* vecHeadPtr)
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	T* ptr;
	size_t mappedSize = (size + 1) * 2 * sizeof(T);
	cudaGraphicsMapResources(1, &vectorPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, vectorPtr);
	cumsum2f(deviceMags, ptr + 2, size);
	cudaGraphicsUnmapResources(1, &vectorPtr);

	mappedSize = sizeof(T) * lineWidth * pathLine->getCount();
	cudaGraphicsMapResources(1, &pathPtr);
	cudaGraphicsResourceGetMappedPointer((void**)&ptr, &mappedSize, pathPtr);
	if (pathLine->isTimestamped())
		fillPathTimestamped<T,T2,T3><<<(cacheSize + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE>>>
		(devicePathCache, cacheSize * 2, ptr, lineWidth * pathLine->getCount(), this->time, dt, head);
	else
		fillPath<T,T2,T3><<<(cacheSize + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE, CACHE_BLOCK_SIZE >>>
		(devicePathCache, cacheSize * 2, ptr, lineWidth * pathLine->getCount(), head);
	cudaGraphicsUnmapResources(1, &pathPtr);

	head = (head + lineWidth * cacheSize) % (pathBufferSize);

	if (vecHeadPtr != nullptr)
		cudaMemcpy(vecHeadPtr, devicePathCache + (cacheSize - 1) * 2, sizeof(T) * 2, cudaMemcpyDeviceToHost);

	
	/*glBindBuffer(GL_ARRAY_BUFFER, pathLine->getBuffer());
	double* ptr2 = (double*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
	for (int i = 0; i < pathLine->getCount() * (pathLine->isTimestamped() ? 3ull : 2ull) * 2; i++)
		std::cout << ptr2[i] << " ";
	std::cout << std::endl << std::endl << std::endl;
	glUnmapBuffer(GL_ARRAY_BUFFER);*/
}

template <typename T, typename T2, typename T3>
void CudaFourierSeries<T, T2, T3>::readyBuffers()
{
	cudaDeviceSynchronize();
}
