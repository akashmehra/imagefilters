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
    void displayReshapedImage(const CImg<T>& image,
			      const Image& imgInfo,
			      CImg<T>* destinationImage);
    void unroll(CImg<T>& image, 
		unsigned int width,
		unsigned int height,
		unsigned int spectrum,
		T* buffer);
  };
  
  template <typename T>
  void ImageProcessing<T>::displayReshapedImage(const CImg<T>& image,
			  const Image& imgInfo,
			  CImg<T>* destinationImage)
    {
      for(int row = 0; row < imgInfo.width; ++row)
      {
	for(int col = 0; col < imgInfo.height; ++col)
	{
	  for(int channel = 0; channel < imgInfo.spectrum ; ++channel)
	  {
	    destinationImage->atXYZ(row,col,0,channel) = image(row,col,0,0)>>1;
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
