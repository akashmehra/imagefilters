#include <iostream>
#include "CImg.h"
#include "gpuProcessImage.cu"
#include "Utils.h"
#include "cpuProcessImage.h"

int main(int argc, char* argv[])
{
  //cimg::imagemagick_path("/opt/local/bin/convert");
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
    
    timeval tim;
    
    double dTime1 = gpu::getTime(tim);
    
    gpu::unrollMatrix(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,
                inputBuffer);
    
    sendWarmUpSignal(inputBuffer,imgInfo.width*imgInfo.height*imgInfo.spectrum); 
    double dTime2 = gpu::getTime(tim);
    std::cout << "time taken for performing warm up: " << dTime2 - dTime1 << std::endl;
    process(inputBuffer,outputBuffer,imgInfo.width, imgInfo.height, imgInfo.spectrum);
    
    CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,imgInfo.height,1,
                                    imgInfo.spectrum,0);
    
    double dTime3 = gpu::getTime(tim);
    std::cout << "time taken for GPU operation: " << dTime3 - dTime2 << std::endl;
    
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
