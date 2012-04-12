#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include "CImg.h"
using namespace cimg_library;

#define BOUND_BUFFER 256
#define MILLION 1000000
#define THOUSAND 1000
#define MILLION_D 1000000.0

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

void displayReshapedImage(const CImg<unsigned char>& image,
			  const Image& imgInfo,
			  CImg<unsigned char>* destinationImage)
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

void unroll(CImg<unsigned char>& image, 
	    unsigned int width,
	    unsigned int height,
	    unsigned int spectrum,
	    unsigned char* buffer)
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

void printMetaData(const Image& image)
{
  std::cout << "Image Metadata:" << std::endl;
  std::cout << "width: " << image.width << ", height: " << image.height << 
  ", size: " << image.size << ", spectrum: " << image.spectrum << std::endl;
}

int main(int argc, char* argv[])
{
  cimg::imagemagick_path("/opt/local/bin/convert");
  if(argc == 2)
  {
    std::string filename = argv[1];
    
    CImg<unsigned char> image(filename.c_str());
    CImgDisplay mainDisplay(image,"Image",0);

    struct Image imgInfo(image.width(),image.height(),image.width()*image.height(),image.spectrum());
    printMetaData(imgInfo);
    
    unsigned char* buffer = new unsigned char[imgInfo.spectrum*imgInfo.size];
    
    timeval tim;
    
    double dTime1 = getTime(tim);
    
    unroll(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,buffer);
    
    CImg<unsigned char> outputImage(buffer,imgInfo.width,imgInfo.height,1,imgInfo.spectrum,0);
    //CImg<unsigned char> outputImage(imgInfo.width,imgInfo.height,1,imgInfo.spectrum);
    //displayReshapedImage(image,imgInfo,&outputImage);
    double dTime2 = getTime(tim);
    std::cout << "time taken for unrolled version: " << dTime2 - dTime1 << std::endl;

    CImgDisplay darkDisplay(outputImage,"Output Image",0);

    while(!(mainDisplay.is_closed()))
    {
      mainDisplay.wait();
    }
    delete[] buffer;
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " <image-filename>" << std::endl;
  }
}
