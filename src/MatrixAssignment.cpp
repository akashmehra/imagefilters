//
//  MatrixAssignment.cpp
//  ImageProcessingFilters
//
//  Created by Akash Mehra on 4/19/12.
//  Copyright (c) 2012 New York University. All rights reserved.
//

#include <iostream>
#include "Constants.h"

template<typename T>
__device__ T DomainCheck(const T& pixel)
{
  return pixel-(pixel-1);
}


template <typename T>
__global__ processingKernel(T* inputBuffer, T* outputBuffer, int width, int height, const int offset)
{
  
  int redChannelOffset = blockIdx.x * blockDim + threadIdx.x;
  int greenChannelOffset = redChannelOffset + 1*offset;
  int blueChannelOffset = redChannelOffset + 2*offset;
  //__shared__ T blockA[BLOCK_SIZE][BLOCK_SIZE];
  //int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  //baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * width;
  //blockA[threadIdx.y][threadIdx.x] = A[baseIdx];
  //__synchthreads();
  //B[baseIdx] = A[baseIdx];//blockA[threadIdx.y][threadIdx.x];
  
  outputBuffer[redChannelOffset] = DomainCheck(inputBuffer[redChannelOffset]);
  outputBuffer[greenChannelOffset] = DomainCheck(inputBuffer[greenChannelOffset]);
  outputBuffer[blueChannelOffset] = DomainCheck(inputBuffer[blueChannelOffset]);
}

struct Setup
{
  int threads;
  int blocks;
};

unsigned int powerOf2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


void getSetupConfig(unsigned int problemSize, struct Setup* setup)
{
  int threads = powerOf2(problemSize);
  threads = threads <= MAX_THREADS ? threads : MAX_THREADS;
  
  int blocks = (problemSize+threads-1)/threads;
  setup->threads = threads;
  setup->blocks = blocks;
}

void runKernel(unsigned char* h_data, const unsigned int problemSize)
{
  int sizeData = problemSize*sizeof(int);
  int sizeResult = sizeData;
  
  Setup setup;
  getSetupConfig(problemSize,&setup);
  
  unsigned char* *d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
  
  unsigned char* *d_result;
  cudaMalloc((void**)&d_result,sizeData);
  
  dim3 dimGrid(setup.blocks,1,1);
  dim3 dimBlock(setup.threads,1,1);
  int sizeSharedMem = problemSize;
  processingKernel<<<dimGrid,dimBlock>>>(d_data,d_result,problemSize);
  
  cudaMemcpy(result,d_result,sizeResult,
             cudaMemcpyDeviceToHost);
  
  cudaFree(d_data);
  cudaFree(d_result);
}
