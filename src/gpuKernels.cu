#ifndef gpu_gpuKernels_cu
#define gpu_Kernels_cu

#include "Utils.h"
#include "Filters.h"

template <typename T>
__global__ void luminousFilterKernel(T* inputBuffer, T* outputBuffer,
																		 int width,int height,int channels,
																		 int offset,float value, 
                                     gpu::FilterType filterType)
{
  
 	int rOffset, gOffset, bOffset; 
  calculateChannelOffsets(offset, blockIdx.x, blockDim.x, threadIdx.x, 
													&rOffset,&gOffset,&bOffset);
  
	gpu::LuminousFilters<T> luminousFilters;
  outputBuffer[rOffset] = luminousFilters.apply(inputBuffer[rOffset],value,
																								filterType);
  outputBuffer[gOffset] = luminousFilters.apply(inputBuffer[gOffset],value,
																								filterType);
  outputBuffer[bOffset] = luminousFilters.apply(inputBuffer[bOffset],value,
																								filterType);
  __syncthreads();
}



template <typename T>
__global__ void colorspaceFilterKernel(T* inputBuffer, T* outputBuffer,
																			 int width,int height, int channels,
																			 int offset,float value,
																			 gpu::FilterType filterType)
{
  
 	int rOffset, gOffset, bOffset;
  
  calculateChannelOffsets(offset,blockIdx.x,blockDim.x,threadIdx.x,
                          &rOffset, &gOffset,&bOffset);
  
	gpu::ColorSpaceFilters<T> colorSpaceFilters;
  
  colorSpaceFilters.apply(inputBuffer[rOffset],
													inputBuffer[gOffset],
                          inputBuffer[bOffset], 
													outputBuffer[rOffset],
                          outputBuffer[gOffset],
													outputBuffer[bOffset],
                          value,filterType);
  __syncthreads();
}

template <typename T>
__global__ void blendFilterKernel(T* baseBuffer, T* blendBuffer, 
																	T* outputBuffer,int width,int height, 
																	int channels,int offset,float value, 
																	gpu::BlendType filterType)
{
  
 	int rOffset, gOffset, bOffset;
  
  calculateChannelOffsets(offset,blockIdx.x,blockDim.x,threadIdx.x,
                          &rOffset,&gOffset,&bOffset);
  
	gpu::BlendFilters<T> blendFilters;
  
  blendFilters.apply(baseBuffer[rOffset],
										 baseBuffer[gOffset],
										 baseBuffer[bOffset],
  									 blendBuffer[rOffset],
										 blendBuffer[gOffset],
										 blendBuffer[bOffset],
										 outputBuffer[rOffset],
                     outputBuffer[gOffset],
										 outputBuffer[bOffset],
                     value,filterType);
  __syncthreads();
}

template <typename T>
__global__ void convolutionFilterKernel(T* inputBuffer, T* outputBuffer, 
																				int* kernel,int width,int height,
																				int kernelSize,int normal,
																				int offset)
{
	int rOffset, gOffset, bOffset;
	calculateChannelOffsets(offset, blockIdx.x, blockDim.x, threadIdx.x,
													&rOffset,&gOffset,&bOffset);
	
	gpu::ConvolutionFilters<T> convolutionFilters;
  
  convolutionFilters.applyConvolution(inputBuffer,outputBuffer,
																			kernel,width,height,kernelSize, 
																			normal,rOffset,0);
  
	convolutionFilters.applyConvolution(inputBuffer, outputBuffer,
  																		kernel,width,height, kernelSize, 
  																		normal, gOffset, 1);

	convolutionFilters.applyConvolution(inputBuffer, outputBuffer,
  																		kernel,width,height, kernelSize, 
																			normal, bOffset, 2);             								
	__syncthreads();
}

#endif // gpu_gpuKernels_cu
