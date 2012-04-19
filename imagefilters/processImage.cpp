#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include "CImg.h"

#include "processImage.h"
#include "Filters.h"
#include "Utils.h"

using namespace cimg_library;

void printMetaData(const gpu::Image& image)
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

    gpu::Image imgInfo(image.width(),image.height(),image.width()*image.height(),image.spectrum());
    printMetaData(imgInfo);
    
    unsigned char* buffer = new unsigned char[imgInfo.spectrum*imgInfo.size];
    
    timeval tim;
    
    double dTime1 = gpu::getTime(tim);
    gpu::ImageProcessing<unsigned char> imp;
    imp.unroll(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,buffer);
    
    //CImg<unsigned char> outputImage(imgInfo.width,imgInfo.height,1,imgInfo.spectrum);
    
    //    brightnessPtr = bFilter.apply;
    //gpu::LuminousFilters<unsigned char> lFilter(0.0,1.2,gpu::LUMINOUS_FILTER_BRIGHTNESS);
    //imp.applyFilter(image,imgInfo,buffer,&lFilter);
    
    gpu::ColorSpaceFilters<unsigned char> clsp;
    imp.applyFilter(image, imgInfo, buffer, &clsp);
    
    CImg<unsigned char> outputImage(buffer,imgInfo.width,imgInfo.height,1,imgInfo.spectrum,0);
    
    double dTime2 = gpu::getTime(tim);
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
