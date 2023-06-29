#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "constant.h"
#include <iostream>

extern "C" {
	DLL_API int __cdecl printGPUInfo()
	{
		cudaError_t err;
		int devicesCount;
		if (err = cudaGetDeviceCount(&devicesCount))
		{
			printf("%s\n", cudaGetErrorString(err));
			return -1;
		}
		int n = 0;
		printf("Avaliable GPUs\n");
		for (int i = 0; i < devicesCount; ++i)
		{
			cudaDeviceProp deviceProperties;
			if (err = cudaGetDeviceProperties(&deviceProperties, i))
			{
				printf("%s\n", cudaGetErrorString(err));
				return -1;
			}
			if (deviceProperties.major >= CUDA_MINIMUM_MAJOR_VERSION
				&& deviceProperties.minor >= CUDA_MINIMUM_MINOR_VERSION)
				printf("[%d] %s - Compute Capability %d.%d\n", n++, deviceProperties.name, deviceProperties.major, deviceProperties.minor);
		}

		return 0;
	}
}