#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include "CImg.h"
using namespace cimg_library;

#define BOUND_BUFFER 256
#define MILLION 1000000
#define THOUSAND 1000
#define MILLION_D 1000000.0

namespace gpu
{

  template<typename T>
  struct ContrastFilter
  {
  private:
    T pixelDictionary[256];
  public:
    ContrastFilter(float cValue)
    {
      for(int i=0;i<256;++i)
      {
	if(cValue > 1)
	{
	  float diff = cValue - 1;
	  cValue = 1 + diff/2;
	}
	int val = (int)((float)i*cValue + (-cValue*128 + 128));   
	if(val < 0) val = 0;
	if(val > 255) val = 255;
	pixelDictionary[i] = val;
      }
    }
    T apply(const T& pixel) {return pixelDictionary[pixel];}

  };
  

  template<typename T>
  struct BrightnessFilter
  {
  private:
    T pixelDictionary[256];
  public:
    BrightnessFilter(float cValue)
    {
      for(int i=0;i<256;++i)
      {
	int val = (int)(i*cValue);
	if(val < 0) val = 0;
	if(val > 255) val = 255;
	pixelDictionary[i] = val;
      }
    }
    T apply(const T& pixel) {return pixelDictionary[pixel];}

  };

  int diff_ms(timeval t1, timeval t2)
  {
    return (((t1.tv_sec - t2.tv_sec) * MILLION) + 
            (t1.tv_usec - t2.tv_usec))/THOUSAND;
  }
  
  double getTime(timeval& tim)
  {
    gettimeofday(&tim,NULL);
    return tim.tv_sec+(tim.tv_usec/MILLION_D);  
  }
  
  struct Image
  {
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned int spectrum;
    
    Image(unsigned int width_,
	unsigned int height,
	unsigned int size_,
	unsigned int spectrum_)
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
    void applyFilter(const CImg<T>& image,
		     const Image& imgInfo,
		     CImg<T>* destinationImage,
		     BrightnessFilter<T>* filterObject
		     );
    
    void applyFilter(const CImg<T>& image,
		     const Image& imgInfo,
		     T* buffer,
		     BrightnessFilter<T>* filterObject
		     );
    
    void unroll(CImg<T>& image, 
		unsigned int width,
		unsigned int height,
		unsigned int spectrum,
		T* buffer);
  };
  
  template <typename T>
  void ImageProcessing<T>::applyFilter(const CImg<T>& image,
				       const Image& imgInfo,
				       CImg<T>* destinationImage,
				       BrightnessFilter<T>* filterObject)
    {
      for(int row = 0; row < imgInfo.width; ++row)
      {
	for(int col = 0; col < imgInfo.height; ++col)
	{
	  for(int channel = 0; channel < imgInfo.spectrum ; ++channel)
	  {
	    destinationImage->atXYZ(row,col,0,channel) = filterObject->apply(image(row,col,0,channel));
	    // destinationImage->atXYZ(row,col,0,channel) = Filters<unsigned char>::brightness(1.1,destinationImage->atXYZ(row,col,0,channel));
	  }
	}
      }
    }
  
  template <typename T>
  void ImageProcessing<T>::applyFilter(const CImg<T>& image,
				       const Image& imgInfo,
				       T* buffer,
				       BrightnessFilter<T>* filterObject)
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
  void ImageProcessing<T>::unroll(CImg<T>& image, 
			       unsigned int width,
			       unsigned int height,
			       unsigned int spectrum,
			       T* buffer)
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
