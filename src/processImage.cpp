#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include "CImg.h"

#include "processImage.h"
#include "Filters.h"
#include "Utils.h"

using namespace cimg_library;


template <typename T>
__global__ processingKernel(T* inputBuffer, T* outputBuffer, int width, 
                            int height, int channels,const int offset)
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

void runKernel(unsigned char* h_data, unsigned char* h_result,const unsigned int width, const unsigned int height, unsigned int channels)
{
  int problemSize = width*height*channels;
  int sizeData = problemSize*sizeof(unsigned char);
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
  int offset = width*height;
  processingKernel<<<dimGrid,dimBlock>>>(d_data,d_result,width,height, channels, offset);
  
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
  cimg::imagemagick_path("/opt/local/bin/convert");
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
    
    runKernel(inputBuffer,outputBuffer,imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    //imp.saturation(S_VALUE,inputBuffer, outputBuffer, 
      //             imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,imgInfo.height,1,
                                    imgInfo.spectrum,0);
    
    double dTime2 = gpu::getTime(tim);
    std::cout << "time taken for unrolled version: " << dTime2 - dTime1 << std::endl;
    
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
