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
  public:
    void applyFilter(const CImg<T>& image, const Image& imgInfo,
                     CImg<T>* destinationImage, LuminousFilters<T>* filterObject);
    
    void applyFilter(const CImg<T>& image, const Image& imgInfo,
                     T* buffer, LuminousFilters<T>* filterObject);
    
    void applyFilter(const CImg<T>& image, const Image& imgInfo,
                     CImg<T>* destinationImage, ColorSpaceFilters<T>* filterObject);
    
    void applyFilter(const CImg<T>& image, const Image& imgInfo,
                     T* buffer, ColorSpaceFilters<T>* filterObject);
    
    void unroll(CImg<T>& image, unsigned int width, unsigned int height,
                unsigned int spectrum, T* buffer);
  };
  
  template <typename T>
  void ImageProcessing<T>::applyFilter(const CImg<T>& image, const Image& imgInfo,
                                       CImg<T>* destinationImage,LuminousFilters<T>* filterObject)
  {
    for(int row = 0; row < imgInfo.width; ++row)
    {
      for(int col = 0; col < imgInfo.height; ++col)
      {
        for(int channel = 0; channel < imgInfo.spectrum ; ++channel)
        {
          destinationImage->atXYZ(row,col,0,channel) = filterObject->apply(image(row,col,0,channel));
        }
      }
    }
  }
  
  template <typename T>
  void ImageProcessing<T>::applyFilter(const CImg<T>& image,const Image& imgInfo,
                                       T* buffer,LuminousFilters<T>* filterObject)
                                       
  {
    for(int channel = 0; channel < imgInfo.spectrum ; ++channel)
    {
      for(int j = 0; j < imgInfo.width; ++j)
      {
        for(int i = 0; i < imgInfo.height; ++i)
        {
          int k = channel*imgInfo.width*imgInfo.height +  i * imgInfo.width + j;
          buffer[k] = filterObject->apply(image(j,i,0,channel));
        }
      }
    }
  }
  
  template <typename T>
  void ImageProcessing<T>::applyFilter(const CImg<T>& image,const Image& imgInfo,
                                       T* buffer,ColorSpaceFilters<T>* filterObject)
  
  {
      for(int j = 0; j < imgInfo.width; ++j)
      {
        for(int i = 0; i < imgInfo.height; ++i)
        {
          int redChannel = 0*imgInfo.width*imgInfo.height +  i * imgInfo.width + j;
          int greenChannel = 1*imgInfo.width*imgInfo.height +  i * imgInfo.width + j;
          int blueChannel = 2*imgInfo.width*imgInfo.height +  i * imgInfo.width + j;
          T pixelR = image(j,i,0,0);
          T pixelG = image(j,i,0,1);
          T pixelB = image(j,i,0,2);
          filterObject->GetSepia(pixelR,pixelG,pixelB,buffer[redChannel],buffer[greenChannel],
                                  buffer[blueChannel]);
        }
      }
  }
  
  template <typename T>
  void ImageProcessing<T>::unroll(CImg<T>& image,unsigned int width,
                                   unsigned int height,unsigned int spectrum,T* buffer)
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
