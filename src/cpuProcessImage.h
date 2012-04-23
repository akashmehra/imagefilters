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
  public:
    
    void adjustContrast(T *inputBuffer, T *outputBuffer, int imageWidth,
                        int imageHeight, int spectrum, float value);
    void adjustBrightness(T *inputBuffer, T *outputBuffer, int imageWidth,
                          int imageHeight, int spectrum, float value); 
    void sepia(T *inputBuffer, T *outputBuffer, int imageWidth,
               int imageHeight, int spectrum);
    void saturation(float sValue,T *inputBuffer, T *outputBuffer, 
                    int imageWidth, int imageHeight, int spectrum);  
  };
  
  template <typename T>
  void ImageProcessing<T>::adjustContrast(T *inputBuffer, T *outputBuffer, int imageWidth,
                                          int imageHeight, int spectrum,float value)                                       
  {
    for(int channel = 0; channel < spectrum ; ++channel)
    {
      for(int j = 0; j < imageWidth; ++j)
      {
        for(int i = 0; i < imageHeight; ++i)
        {
          int k = channel*imageWidth*imageHeight +  i * imageWidth + j;
          outputBuffer[k] = luminousFilter.apply(inputBuffer[k], value, LUMINOUS_FILTER_CONTRAST);
        }
      }
    }
  }
  
  template <typename T>
  void ImageProcessing<T>::adjustBrightness(T *inputBuffer, T *outputBuffer, int imageWidth,
                                          int imageHeight, int spectrum,float value)                                       
  {
    for(int channel = 0; channel < spectrum ; ++channel)
    {
      for(int j = 0; j < imageWidth; ++j)
      {
        for(int i = 0; i < imageHeight; ++i)
        {
          int k = channel*imageWidth*imageHeight +  i * imageWidth + j;
          outputBuffer[k] = luminousFilter.apply(inputBuffer[k], value, LUMINOUS_FILTER_BRIGHTNESS);
        }
      }
    }
  }
  
  template <typename T>
  void ImageProcessing<T>::sepia(T *inputBuffer, T *outputBuffer, 
                                int imageWidth, int imageHeight, int spectrum)  
  {
    if(spectrum == 3)
    {
      for(int j = 0; j < imageWidth; ++j)
      {
        for(int i = 0; i < imageHeight; ++i)
        {
          int redChannel = 0*imageWidth*imageHeight +  i * imageWidth + j;
          int greenChannel = 1*imageWidth*imageHeight +  i * imageWidth + j;
          int blueChannel = 2*imageWidth*imageHeight +  i * imageWidth + j;
          
          colorSpaceFilter.sepia(inputBuffer[redChannel],inputBuffer[greenChannel],inputBuffer[blueChannel],
                                 outputBuffer[redChannel],outputBuffer[greenChannel],outputBuffer[blueChannel]);
        }
      }
    }
  }
  
  template <typename T>
  void ImageProcessing<T>::saturation(float sValue, T *inputBuffer, T *outputBuffer, 
                                      int imageWidth, int imageHeight, int spectrum)  
  {
    if(spectrum == 3)
    {
      for(int j = 0; j < imageWidth; ++j)
      {
        for(int i = 0; i < imageHeight; ++i)
        {
          int redChannel = 0*imageWidth*imageHeight +  i * imageWidth + j;
          int greenChannel = 1*imageWidth*imageHeight +  i * imageWidth + j;
          int blueChannel = 2*imageWidth*imageHeight +  i * imageWidth + j;
          
          colorSpaceFilter.saturation(inputBuffer[redChannel],inputBuffer[greenChannel],
                                      inputBuffer[blueChannel],outputBuffer[redChannel],
                                      outputBuffer[greenChannel],outputBuffer[blueChannel],sValue);
        }
      }
    }
  }
  
}
