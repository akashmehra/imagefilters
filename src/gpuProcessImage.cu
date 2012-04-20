#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include <stdio.h>
#include "CImg.h"

#include "processImage.h"
#include "Filters.h"
#include "Utils.h"
#include "Constants.h"
using namespace cimg_library;

template<typename T>
__device__ T DomainCheck(T pixel)
{
  PIXEL_DOMAIN_CHECK(pixel);
  return pixel;
}

template <typename T>
__global__ void processingKernel(T* inputBuffer, T* outputBuffer, int width, 
                            int height, int channels,const int offset)
{
  
  int redChannelOffset = blockIdx.x * blockDim.x + threadIdx.x;
  int greenChannelOffset = redChannelOffset + 1*offset;
  int blueChannelOffset = redChannelOffset + 2*offset;
  float brightnessVal = 0.0f;
  outputBuffer[redChannelOffset] = (T)DomainCheck<float>(50+inputBuffer[redChannelOffset]);
  outputBuffer[greenChannelOffset] = (T)DomainCheck<float>(50+inputBuffer[greenChannelOffset]);
  outputBuffer[blueChannelOffset] = (T)DomainCheck<float>(50+inputBuffer[blueChannelOffset]);
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


void getSetupConfig(unsigned int problemSize, struct Setup* setup)
{
  int threads = powerOf2(problemSize);
  threads = threads <= MAX_THREADS ? threads : MAX_THREADS;
  
  int blocks = (problemSize)/threads;
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

void runKernel(unsigned char* h_data, unsigned char* h_result,const unsigned int width, const unsigned int height, unsigned int channels)
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
  
  dim3 dimGrid(setup.blocks,1,1);
  dim3 dimBlock(setup.threads,1,1);
  int sizeSharedMem = problemSize;
  int offset = width*height;
  processingKernel<unsigned char><<<dimGrid,dimBlock>>>(d_data,d_result,width,height,channels, offset);
  
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
    
    gpu::ImageProcessing<unsigned char> imp;
    timeval tim;
    
    double dTime1 = gpu::getTime(tim);
    
    gpu::unroll(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,
                inputBuffer);
    
    sendWarmUpSignal(inputBuffer,imgInfo.width*imgInfo.height*imgInfo.spectrum); 
    double dTime2 = gpu::getTime(tim);
    std::cout << "time taken for unrolled version: " << dTime2 - dTime1 << std::endl;
    runKernel(inputBuffer,outputBuffer,imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    //imp.saturation(S_VALUE,inputBuffer, outputBuffer, 
      //             imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,imgInfo.height,1,
                                    imgInfo.spectrum,0);
    
    double dTime3 = gpu::getTime(tim);
    std::cout << "time taken for unrolled version: " << dTime3 - dTime2 << std::endl;
    
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
