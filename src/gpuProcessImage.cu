#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include <stdio.h>
#include "CImg.h"

#include "cpuProcessImage.h"
#include "Filters.h"
#include "Utils.h"
#include "Constants.h"
using namespace cimg_library;

__device__ void calculateChannelOffsets(int offset, int blockIndex, int blockDimension, 
                                        int threadIndex,int* redChannelOffset, 
                                        int* greenChannelOffset, int* blueChannelOffset)
{
	*redChannelOffset = blockIndex * blockDimension + threadIndex;
	*greenChannelOffset = *redChannelOffset + 1*offset;
	*blueChannelOffset = *redChannelOffset + 2*offset;
}


template <typename T>
__global__ void colorspaceFilterKernel(T* inputBuffer, T* outputBuffer, int width, 
                                       int height, int channels,int offset, 
                                       float value, gpu::FilterType filterType)
{
  
 	int redChannelOffset, greenChannelOffset, blueChannelOffset;
  
  calculateChannelOffsets(offset,blockIdx.x,blockDim.x,threadIdx.x,
                          &redChannelOffset, &greenChannelOffset, &blueChannelOffset);
  
	gpu::ColorSpaceFilters<T> colorSpaceFilters;
  
  colorSpaceFilters.apply(inputBuffer[redChannelOffset],inputBuffer[greenChannelOffset],
                          inputBuffer[blueChannelOffset], outputBuffer[redChannelOffset],
                          outputBuffer[greenChannelOffset],outputBuffer[blueChannelOffset],
                          value,filterType);
  __syncthreads();
}

template <typename T>
__global__ void luminousFilterKernel(T* inputBuffer, T* outputBuffer, int width, 
                                     int height, int channels,int offset, float value , 
                                     gpu::FilterType filterType)
{
  
 	int redChannelOffset, greenChannelOffset, blueChannelOffset; 
  calculateChannelOffsets(offset, blockIdx.x, blockDim.x, threadIdx.x, &redChannelOffset, &greenChannelOffset, &blueChannelOffset);
  
	gpu::LuminousFilters<T> luminousFilters;
  outputBuffer[redChannelOffset] =   luminousFilters.apply(inputBuffer[redChannelOffset],value,filterType);
  outputBuffer[greenChannelOffset] = luminousFilters.apply(inputBuffer[greenChannelOffset],value,filterType);
  outputBuffer[blueChannelOffset] =  luminousFilters.apply(inputBuffer[blueChannelOffset],value,filterType);
  __syncthreads();
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

template <typename T>
void callKernel(void(*kernel)(T*,T*,int,int,int,int,float,gpu::FilterType), 
                int blocks, int threads, 
          			T* inputBuffer,T* outputBuffer,
								int width, int height, int channels, int offset, 
                float value,gpu::FilterType filterType) 
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
  kernel<<<dimGrid,dimBlock>>>(inputBuffer, outputBuffer, width, height, channels, offset, value,filterType);
}


void getSetupConfig(unsigned int problemSize, struct Setup* setup)
{
  int threads = powerOf2(problemSize);
  threads = threads <= MAX_THREADS ? threads : MAX_THREADS;
  
  int blocks = (problemSize)/threads;
	std::cout << "inside setup config, blocks: " << blocks << std::endl;
	if(problemSize % threads != 0)
	{
		blocks +=1;
	}
	std::cout << "inside setup config, blocks: " << blocks << std::endl;
	
  setup->threads = threads;
  setup->blocks = blocks;
}

void sendWarmUpSignal(unsigned char* h_data, const unsigned int sizeData)
{
  unsigned char* d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
  cudaFree(d_data);
}

void runKernel(unsigned char* h_data, unsigned char* h_result,int width, int height, int channels)
{
  int problemSize = width*height*channels;
  int sizeData = problemSize*sizeof(unsigned char);
  int sizeResult = sizeData;
  
  Setup setup;
  getSetupConfig(width*height,&setup);
  std::cout << "Blocks: " << setup.blocks << std::endl;
  std::cout << "Threads: " << setup.threads << std::endl;
  
  unsigned char *d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
  
  unsigned char* d_result;
  cudaMalloc((void**)&d_result,sizeData);
  
  int offset = width*height;
  //  callKernel<unsigned char>(luminousFilterKernel,setup.blocks,setup.threads,
  //                          d_data,d_result,width,height,channels,offset, BRIGHTNESS_VALUE, gpu::LUMINOUS_FILTER_BRIGHTNESS);
  callKernel<unsigned char>(colorspaceFilterKernel,setup.blocks,setup.threads,
                            d_data,d_result,width,height,channels,offset,
                            SATURATION_VALUE, gpu::COLORSPACE_FILTER_SATURATION);
  
  cudaMemcpy(h_result,d_result,sizeResult,
             cudaMemcpyDeviceToHost);
  
  cudaFree(d_data);
  cudaFree(d_result);
}


void printMetaData(const gpu::Image& image)
{
  std::cout << "Image Metadata:" << std::endl;
  std::cout << "width: " << image.width << ", height: " << image.height << 
  ", size: " << image.size << ", spectrum: " << image.spectrum << std::endl;
}

int main(int argc, char* argv[])
{
  //cimg::imagemagick_path("/opt/local/bin/convert");
  if(argc == 3)
  {
    std::string filename = argv[1];
    std::string outputFilename = argv[2];
    CImg<unsigned char> image(filename.c_str());
    CImgDisplay mainDisplay(image,"Image",0);
    
    gpu::Image imgInfo(image.width(),image.height(),image.width()*image.height(),image.spectrum());
    printMetaData(imgInfo);
    
    /*
     <summary> 
     1. Allocate Buffers
     2. Get Meta information from the image and assign that to ImageInfo object.
     3. Copy image into Input Buffer (unroll operation).
     4. Perform the operation.
     */
    
    unsigned char* inputBuffer = new unsigned char[imgInfo.spectrum*imgInfo.size];
    unsigned char* outputBuffer = new unsigned char[imgInfo.spectrum*imgInfo.size];
    
    timeval tim;
    
    double dTime1 = gpu::getTime(tim);
    
    gpu::unroll(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,
                inputBuffer);
    
    sendWarmUpSignal(inputBuffer,imgInfo.width*imgInfo.height*imgInfo.spectrum); 
    double dTime2 = gpu::getTime(tim);
    std::cout << "time taken for performing warm up: " << dTime2 - dTime1 << std::endl;
    runKernel(inputBuffer,outputBuffer,imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,imgInfo.height,1,
                                    imgInfo.spectrum,0);
    
    double dTime3 = gpu::getTime(tim);
    std::cout << "time taken for GPU operation: " << dTime3 - dTime2 << std::endl;
    
    outputImage.save_jpeg(outputFilename.c_str());
    CImgDisplay darkDisplay(outputImage,"Output Image",0);
    
    while(!(mainDisplay.is_closed()))
    {
      mainDisplay.wait();
    }
    delete[] inputBuffer;
    delete[] outputBuffer;
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " <image-filename> <output-filename>" << std::endl;
  }
}
