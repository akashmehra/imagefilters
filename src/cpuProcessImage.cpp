#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include "CImg.h"

#include "cpuProcessImage.h"
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
//  cimg::imagemagick_path("/opt/local/bin/convert");
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
    
    gpu::ImageProcessing<unsigned char> imp;
    timeval tim;
    
    double dTime1 = gpu::getTime(tim);
    
    gpu::unroll(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,
                inputBuffer);
   
		imp.adjustBrightness(inputBuffer, outputBuffer, imgInfo.width, imgInfo.height, imgInfo.spectrum, BRIGHTNESS_VALUE); 
    double dTime2 = gpu::getTime(tim);
    std::cout << "time taken for brightness: " << dTime2 - dTime1 << std::endl;
    imp.saturation(S_VALUE,outputBuffer, outputBuffer, 
                   imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,imgInfo.height,1,
                                    imgInfo.spectrum,0);
    
    double dTime3 = gpu::getTime(tim);
    std::cout << "time taken for saturation: " << dTime3 - dTime2 << std::endl;
    
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
