#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include "CImg.h"

#include "Filters.h"
#include "blendFilters.h"
#include "Constants.h"

using namespace cimg_library;


namespace gpu
{
  template <typename T>
  static  void unroll(CImg<T>& image, unsigned int width, unsigned int height,
                      unsigned int spectrum, T* buffer)
  {
    {
      for(unsigned int k = 0;k<spectrum;++k)
      {
        for(unsigned int j=0;j<height;++j)
        {
          for(int i=0;i<width;++i)
          {
            buffer[k*height*width+j*width+i] = image(i,j,0,k);
          }
        }
      }
    }

  }
  
  struct Image
  {
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned int spectrum;
    
    Image(unsigned int width_,unsigned int height,
          unsigned int size_,unsigned int spectrum_)
    :width(width_),
    height(height),
    size(size_),
    spectrum(spectrum_)
    {}
  };
  
  
  template<typename T>
  class ImageProcessing
  {
  private:
    LuminousFilters<T> luminousFilter;
    ColorSpaceFilters<T> colorSpaceFilter;
    BlendFilters<T> blendFilter;
  public:
    
    void applyLuminousFilter    (T *inputBuffer,
                                 T *outputBuffer, 
                                 int imageWidth,
                                 int imageHeight, 
                                 int spectrum, 
                                 float value,
                                 gpu::LuminousFilterTypes filterType);  

     
    void applyColorSpaceFilter  (T *inputBuffer, 
                                 T *outputBuffer, 
                                 int imageWidth,
                                 int imageHeight, 
                                 int spectrum,
                                 float value,
                                 gpu::ColorSpaceFilterTypes filterType);  
      
    void applyBlendFilter       (T *baseBuffer, 
                                 T *blendBuffer, 
                                 T *destinationBuffer, 
                                 int imageWidth,
                                 int imageHeight, 
                                 int spectrum,
                                 float value,
                                 gpu::BlendType filterType);
  };

  template<typename T>
  void ImageProcessing<T>::applyLuminousFilter(T *inputBuffer, 
                                               T *outputBuffer, 
                                               int imageWidth,
                                               int imageHeight, 
                                               int spectrum,
                                               float value,
                                               gpu::LuminousFilterTypes filterType)                                       
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
                                                   gpu::ColorSpaceFilterTypes filterType)                                       
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
                                              gpu::BlendType filterType)                                   
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
                                           destinationBuffer[r],destinationBuffer[g],destinationBuffer[b],value,filterType);
                }
            }
        //}
    }
    
    
    
  
}
