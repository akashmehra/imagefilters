#include <ctime>
#include <cmath>
#include <iostream>
#include "gpuUtils.cu"
#include "gpuKernels.cu"
#include "cpuProcessImage.h"
#include "Filters.h"
#include "Constants.h"
using namespace cimg_library;

template <typename T>
void callLumAndColorKernel(void(*kernel)(T*,T*,int,int,int,int,float,gpu::FilterType), 
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
void callConvolutionKernel(void(*kernel)(T*,T*,int*,
																				int,int,int,int,int),
			                		 int blocks, int threads, 
      			    					 T* inputBuffer,T* outputBuffer,int* convKernel,
													 int width, int height, int kernelSize, int normal,
													 int offset) 
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
	std::cout << "CONVOLUTION KERNEL" << std::endl;
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

void runBlendKernel(const gpu::Setup& setup,
                    unsigned char* d_data,unsigned char* d_blendData,
                    unsigned char* d_result,
                    int width, int height, int channels, int offset,
                    float value, gpu::BlendType filterType)
{
  callBlendKernel<unsigned char>(blendFilterKernel,setup.blocks,setup.threads,
                                 d_data,d_blendData,d_result,width,height,
                                 channels,offset,value,filterType);
}

void runSepiaKernel(const gpu::Setup& setup,
                    unsigned char* d_data, unsigned char* d_result,
                    int width, int height, int channels, int offset)
{
  callLumAndColorKernel<unsigned char>(colorspaceFilterKernel,setup.blocks,setup.threads,
                                       d_data,d_result,width,height,channels,offset,
                                       0.0f,gpu::COLORSPACE_FILTER_SEPIA);

}


void runSaturationKernel(const gpu::Setup& setup,
                         unsigned char* d_data, unsigned char* d_result,
                         int width, int height, int channels, int offset
                        ) 
{
  callLumAndColorKernel<unsigned char>(colorspaceFilterKernel,setup.blocks,setup.threads,
                                       d_data,d_result,width,height,channels,offset,
                                       SATURATION_VALUE, 
                                       gpu::COLORSPACE_FILTER_SATURATION);

}

void runBrightnessKernel(const gpu::Setup& setup,
                         unsigned char* d_data, unsigned char* d_result,
                         int width, int height, int channels, int offset)
{
  callLumAndColorKernel<unsigned char>(luminousFilterKernel,setup.blocks,setup.threads,
                                       d_data,d_result,width,height,channels,offset,
                                       BRIGHTNESS_VALUE, 
                                       gpu::LUMINOUS_FILTER_BRIGHTNESS);

}

void runContrastKernel(const gpu::Setup& setup,
                       unsigned char* d_data, unsigned char* d_result,
                       int width, int height, int channels, int offset)
{
  callLumAndColorKernel<unsigned char>(luminousFilterKernel,setup.blocks,setup.threads,
                                       d_data,d_result,width,height,channels,offset,
                                       CONTRAST_VALUE, 
                                       gpu::LUMINOUS_FILTER_CONTRAST);

}


void runConvolutionKernel(const gpu::Setup& setup, 
                          unsigned char* d_data, unsigned char* d_result,
                          int* d_kernel,int kernelWidth,
                          int width, int height, int channels, int offset)
{

  callConvolutionKernel<unsigned char>(convolutionFilterKernel,setup.blocks,setup.threads,
                                       d_data,d_result,d_kernel,
                                       width,height,kernelWidth,1,offset);

}

void process(unsigned char* h_data, unsigned char* h_result,
             int width, int height, int channels)
{
  int problemSize, sizeData, sizeResult;
  gpu::Setup setup;
  startSetup(width, height, channels,&problemSize, &sizeData, &sizeResult, &setup);

  std::cout << "Blocks: " << setup.blocks << std::endl;
  std::cout << "Threads: " << setup.threads << std::endl;

  unsigned char *d_data;
  cudaMalloc((void**)&d_data,sizeData);
  cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);

  unsigned char* d_result;
  cudaMalloc((void**)&d_result,sizeData);

  int offset = width*height;


  //runLuminousKernel(setup,d_data,d_result,h_result,width,height,channels,offset);
  //runConvolutionKernel(setup,d_data,d_result,h_result,d_kernel,width,height,channels,offset);

  /*callKernel<unsigned char>(colorspaceFilterKernel,setup.blocks,setup.threads,
    d_result,d_result,width,height,channels,offset,
    SATURATION_VALUE,gpu::COLORSPACE_FILTER_SATURATION);


    double dTime3 = gpu::getTime(tim);
    std::cout << "time taken for saturation on GPU: " << dTime3 - dTime2 << std::endl;
    callBlendKernel<unsigned char>(blendFilterKernel,setup.blocks,setup.threads,
    d_data,d_result,d_result,width,height,channels,
    offset,SATURATION_VALUE,
    gpu::BLEND_FILTER_LINEARLIGHT);
   */

  //int h_kernel[]={-1,0,1,-2,0,2,-1,0,1};
  //int h_kernel[] = {2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2};

  cudaMemcpy(h_result,d_result,sizeResult,cudaMemcpyDeviceToHost);
  //cudaFree(d_kernel); 
  cudaFree(d_data);
  cudaFree(d_result);
}



























































