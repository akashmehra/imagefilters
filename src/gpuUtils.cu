#ifndef gpu_gpuUtils_cu
#define gpu_gpuUtils_cu

#include "Utils.h"
__device__ void calculateChannelOffsets(int offset, int blockIndex,
																				int blockDimension,int threadIndex,
																				int* rOffset,
																				int* gOffset,
																				int* bOffset)
{
	*rOffset = blockIndex * blockDimension + threadIndex;
	*gOffset = *rOffset + 1*offset;
	*bOffset = *rOffset + 2*offset;
}

void sendWarmUpSignal(unsigned char* h_data, const unsigned int sizeData)
{
  unsigned char* d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
  cudaFree(d_data);
}

void startSetup(int width, int height, int channels,int* problemSize, int* sizeData, int* sizeResult,gpu::Setup* setup)
{
	*problemSize = width*height*channels;
	*sizeData = *problemSize * sizeof(unsigned char);
	*sizeResult = *sizeData;

	gpu::getSetupConfig(width*height,setup);
}

#endif //gpu_gpuUtils_cu
