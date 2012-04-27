#ifndef gpu_cpuProcessingImage_h
#define gpu_cpuProcessingImage_h

#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include "CImg.h"

#include "Filters.h"
#include "Constants.h"

using namespace cimg_library;


namespace gpu
{
  
  template<typename T>
  class ImageProcessing
  {
  private:
		LuminousFilters<T> luminousFilter;
		ColorSpaceFilters<T> colorSpaceFilter;
        BlendFilters<T> blendFilter;
        ConvolutionFilters<T>convolutionFilter;
  public:
    
    void applyLuminousFilter    (T *inputBuffer,
                                 T *outputBuffer, 
                                 int imageWidth,
                                 int imageHeight, 
                                 int spectrum, 
                                 float value,
                                 gpu::FilterType filterType);  

     
    void applyColorSpaceFilter  (T *inputBuffer, 
                                 T *outputBuffer, 
                                 int imageWidth,
                                 int imageHeight, 
                                 int spectrum,
                                 float value,
                                 gpu::FilterType filterType);  
      
    void applyBlendFilter       (T *baseBuffer, 
                                 T *blendBuffer, 
                                 T *destinationBuffer, 
                                 int imageWidth,
                                 int imageHeight, 
                                 int spectrum,
                                 float value,
                                 gpu::BlendType blendType);
      
    void applyConvolution(T *inputBuffer, 
                          T *outputBuffer, 
                          int *kernel, 
                          int imageWidth,
                          int imageHeight, 
                          int spectrum,
                          int kernelSize,
                          int normal);
  };

  template<typename T>
  void ImageProcessing<T>::applyLuminousFilter(T *inputBuffer, 
                                               T *outputBuffer, 
                                               int imageWidth,
                                               int imageHeight, 
                                               int spectrum,
                                               float value,
                                               gpu::FilterType filterType)                                       
    {
        for(int channel = 0; channel < spectrum ; ++channel)
        {
            for(int j = 0; j < imageWidth; ++j)
            {
                for(int i = 0; i < imageHeight; ++i)
                {
                    int k = channel*imageWidth*imageHeight +  i * imageWidth + j;
                    outputBuffer[k] = luminousFilter.apply(inputBuffer[k], value, filterType);
                }
            }
        }
    }
    
    template<typename T>
    void ImageProcessing<T>::applyColorSpaceFilter(T *inputBuffer, 
                                                   T *outputBuffer, 
                                                   int imageWidth,
                                                   int imageHeight, 
                                                   int spectrum,
                                                   float value,
                                                   gpu::FilterType filterType)                                       
    {
            for(int j = 0; j < imageWidth; ++j)
            {
                for(int i = 0; i < imageHeight; ++i)
                {
                    int r = 0*imageWidth*imageHeight +  i * imageWidth + j;
                    int g = 1*imageWidth*imageHeight +  i * imageWidth + j;
                    int b = 2*imageWidth*imageHeight +  i * imageWidth + j;

                    colorSpaceFilter.apply(inputBuffer[r],inputBuffer[g],inputBuffer[b],
                                           outputBuffer[r],outputBuffer[g],outputBuffer[b], value, filterType);
                }
            }
    }
    
    template<typename T>
    void ImageProcessing<T>::applyBlendFilter(T *baseBuffer, 
                                              T *blendBuffer, 
                                              T *destinationBuffer, 
                                              int imageWidth,
                                              int imageHeight, 
                                              int spectrum,
                                              float value,
                                              gpu::BlendType blendType)                                   
    {
        //for(int channel = 0; channel < spectrum ; ++channel)
        //{
            for(int j = 0; j < imageWidth; ++j)
            {
                for(int i = 0; i < imageHeight; ++i)
                {
                    int r = 0*imageWidth*imageHeight +  i * imageWidth + j;
                    int g = 1*imageWidth*imageHeight +  i * imageWidth + j;
                    int b = 2*imageWidth*imageHeight +  i * imageWidth + j;
                    
                    blendFilter.apply(baseBuffer[r],baseBuffer[g],baseBuffer[b],
                                           blendBuffer[r],blendBuffer[g],blendBuffer[b],
                                           destinationBuffer[r],destinationBuffer[g],destinationBuffer[b],value,blendType);
                }
            }
        //}
    }
    
    template<typename T>
    void ImageProcessing<T>::applyConvolution(T *inputBuffer, 
                                              T *outputBuffer, 
                                              int *kernel, 
                                              int imageWidth,
                                              int imageHeight, 
                                              int spectrum,
                                              int kernelSize,
                                              int normal)                                   
    {
        
        
        for(int channel = 0; channel < spectrum ; ++channel)
        {
        	for(int j = 0; j < imageWidth; ++j)
        	{
            for(int i = 0; i < imageHeight; ++i)
            {
            int offset = channel*imageWidth*imageHeight +  i * imageWidth + j;
            convolutionFilter.applyConvolution(inputBuffer,outputBuffer,kernel,imageWidth,imageHeight,kernelSize,normal,offset,channel);
            }
        	}
    		}
		}
}

#endif
