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
                          &rOffset, &gOffset, 
													&bOffset);
  
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
                          &rOffset,
													&gOffset,
													&bOffset);
  
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
													&rOffset,
													&gOffset, &bOffset
												);
	gpu::ConvolutionFilters<T> convolutionFilters;
  
  convolutionFilters.applyConvolution(inputBuffer, outputBuffer,
																			kernel,width,height, kernelSize, 
																			normal, rOffset, 0);
  
	/*convolutionFilters.applyConvolution(inputBuffer, outputBuffer,
  																		kernel,width,height, kernelSize, 
  																		normal, gOffset, 1);


	convolutionFilters.applyConvolution(inputBuffer, outputBuffer,
  																		kernel,width,height, kernelSize, 
																			normal, bOffset, 2);             								
 */
	 __syncthreads();
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
  kernel<<<dimGrid,dimBlock>>>(inputBuffer, outputBuffer, 
															 width, height, channels, offset, 
															 value,filterType);
}

template <typename T>
void callConvolutionKernel(void(*kernel)(T*,T*,int*,int,int,int,int,int),
			                		 int blocks, int threads, 
      			    					 T* inputBuffer,T* outputBuffer,int* convKernel,
													 int width, int height, int kernelSize, int normal,
													 int offset) 
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
  kernel<<<dimGrid,dimBlock>>>(inputBuffer, outputBuffer,convKernel, 
															 width, height, kernelSize, normal,
															 offset); 
}

template <typename T>
void callBlendKernel(void(*kernel)(T*,T*,T*,int,int,int,int,float,
																	gpu::BlendType), 
                		 int blocks, int threads, 
          					 T* baseBuffer,T* blendBuffer,T* outputBuffer,
										 int width, int height, int channels, int offset, 
                		 float value,gpu::BlendType filterType) 
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
  kernel<<<dimGrid,dimBlock>>>(baseBuffer, blendBuffer,outputBuffer, 
															 width, height, channels, offset, 
															 value,filterType);
}


void sendWarmUpSignal(unsigned char* h_data, const unsigned int sizeData)
{
  unsigned char* d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
  cudaFree(d_data);
}

void runKernel(unsigned char* h_data, unsigned char* h_result,
							 int width, int height, int channels)
{
  int problemSize = width*height*channels;
  int sizeData = problemSize*sizeof(unsigned char);
  int sizeResult = sizeData;
  
  gpu::Setup setup;
  gpu::getSetupConfig(width*height,&setup);
  std::cout << "Blocks: " << setup.blocks << std::endl;
  std::cout << "Threads: " << setup.threads << std::endl;
  
  unsigned char *d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
  
  unsigned char* d_result;
  cudaMalloc((void**)&d_result,sizeData);
  
  int offset = width*height;
  
	timeval tim;
  double dTime1 = gpu::getTime(tim);
	/*callKernel<unsigned char>(luminousFilterKernel,setup.blocks,setup.threads,
                            d_data,d_result,width,height,channels,offset,
														BRIGHTNESS_VALUE, gpu::LUMINOUS_FILTER_BRIGHTNESS);
  */
  double dTime2 = gpu::getTime(tim);
  std::cout << "time taken for brightness on GPU: " << dTime2 - dTime1 << std::endl;
	/*callKernel<unsigned char>(colorspaceFilterKernel,setup.blocks,setup.threads,
                            d_result,d_result,width,height,channels,offset,
                            SATURATION_VALUE,gpu::COLORSPACE_FILTER_SATURATION);
  
	*/
  double dTime3 = gpu::getTime(tim);
  std::cout << "time taken for saturation on GPU: " << dTime3 - dTime2 << std::endl;
	/*callBlendKernel<unsigned char>(blendFilterKernel,setup.blocks,setup.threads,
                            d_data,d_result,d_result,width,height,channels,
														offset,SATURATION_VALUE,
														gpu::BLEND_FILTER_LINEARLIGHT);
  */

  int convKernel[]={-1,0,1,-2,0,2,-1,0,1};
	int kernelSize = sizeof(convKernel)/sizeof(*convKernel);
	callConvolutionKernel(convolutionFilterKernel,setup.blocks,setup.threads,d_data,d_result,
												convKernel,width,height,channels,kernelSize,offset);
	
	cudaMemcpy(h_result,d_result,sizeResult,cudaMemcpyDeviceToHost);
  double dTime4 = gpu::getTime(tim);
  std::cout << "time taken for convolution on GPU: " << dTime4 - dTime3 << std::endl;
  
  cudaFree(d_data);
  cudaFree(d_result);
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
