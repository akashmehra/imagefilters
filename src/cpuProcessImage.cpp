#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include "CImg.h"
#include <sys/types.h>
#include <sys/stat.h>

#include "cpuProcessImage.h"
#include "Filters.h"
#include "Utils.h"

using namespace cimg_library;
using namespace gpu;

int main(int argc, char* argv[])
{
  std::cout << "-------------------RUNNING IMAGE FILTERS APP ON GPU---------------------------" << std::endl;
  gpu::Options options;                                                                                                  
  bool validArguments	= parseCommandLine(argc, argv, &options);
  if(validArguments)                                                                                                     
  {                                                                                                                      
    std::vector<std::string> fileList;                                                                                   
    std::string directoryPath = options.directoryPath;                                                                   

    gpu::readDirectory(directoryPath,&fileList);                                                                         
    std::vector<std::string>::iterator it = fileList.begin();                                                            
    int count = 0;                                                                                                       

    int* warmupBuffer =(int*) calloc(1000, sizeof(int));                                                                 

    timeval tim;                                                                                                         

    std::cout << "Starting File I/O using CPU." << std::endl;
    double fileIOTime = 0.0;                                                                                             

    double configurationTime = 0.0;

    double executionTime = 0.0;                                                                                          
    double dTime1,dTime2,dTime3;
    for(;it != fileList.end();++it)                                                                                      
    {                                                                                                                    
      std::string imageFilename = directoryPath+*it;                                                                     
      std::string extension = imageFilename.substr(imageFilename.length()-4,4);                                          
      if(extension == JPG_EXTENSION || extension == JPEG_EXTENSION)
      {                                                                                                                  
        std::string filename = imageFilename.substr(0,imageFilename.length()-4);	                                       
        int indexOfSlash = imageFilename.find_last_of("/");
        std::cout << "filename: " << filename << std::endl;                                                              
        std::cout << "Reading Image from Disk." << std::endl;                                                            
        dTime1 = gpu::getTime(tim);

        std::string outputFilename = imageFilename.substr(indexOfSlash,                                                  
                                                          imageFilename.length()-4-indexOfSlash);                        

        CImg<unsigned char> image(imageFilename.c_str());                                                                

        dTime2 = gpu::getTime(tim);                                                                                      
        fileIOTime += dTime2 - dTime1;                                                                                   

        std::cout << "Time taken to read from disk:  " << dTime2 - dTime1 << std::endl;

        dTime1 = gpu::getTime(tim);                                                                                      
        std::cout << "Unrolling Image and setting up blocks and threads." << std::endl;
        int width = image.width();                                                                                       
        int height = image.height();
        int channels = image.spectrum();                                                                                 
        gpu::Image imgInfo(image.width(),image.height(),image.width()*image.height(),image.spectrum());                  

        /*
           <summary>                                                                                                     
           1. Allocate Buffers                                                                                           
           2. Get Meta information from the image and assign that to ImageInfo object.                                   
           3. Copy image into Input Buffer (unroll operation).                                                           
           4. Perform the operation.                                                                                     
           */                                                                                                              

        unsigned char* h_data = new unsigned char[imgInfo.spectrum*imgInfo.size];                                        
        unsigned char* h_result = new unsigned char[imgInfo.spectrum*imgInfo.size];                                      
        gpu::unrollMatrix(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,                                           
                          h_data);                                                                                       
        int problemSize, sizeData, sizeResult;                                                                           

        dTime2 = gpu::getTime(tim);                                                                                      

        configurationTime += dTime2 - dTime1;                                                                            

        std::cout << "Done configuring the problem.\nTime taken: " << dTime2 - dTime1 << std::endl;

        dTime1 = gpu::getTime(tim);
        ImageProcessing <unsigned char>ip;
        switch(options.filterFlag)
        {									

        case gpu::BRIGHTNESS:
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.2,LUMINOUS_FILTER_BRIGHTNESS);
                break;

        case gpu::CONTRAST:
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.5,LUMINOUS_FILTER_CONTRAST);
                break;
        case gpu::CONVOLUTION:
                ip.applyConvolution(h_data,h_result,options.convolutionKernel,width,height,channels,options.kernelSize,0);
                break;
        case gpu::BLEND:
                ip.applyBlendFilter(h_data,h_data,h_result, width,height,channels,1.0f,options.blendMode);
                break;
        case gpu::SATURATION:
                ip.applyColorSpaceFilter(h_data,h_result, width,height,channels,50.2f,COLORSPACE_FILTER_SATURATION);
                break;
        case gpu::SEPIA:
                ip.applyColorSpaceFilter(h_data,h_result, width,height,channels,0,COLORSPACE_FILTER_SEPIA);
                break;
        case gpu::BLACKWHITE:
                ip.applyColorSpaceFilter(h_data,h_result, width,height,channels,0,COLORSPACE_FILTER_BW);
                break;
        case gpu::BRIGHTNESS_CONTRAST:
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.2f,LUMINOUS_FILTER_BRIGHTNESS);
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.5f,LUMINOUS_FILTER_CONTRAST);
                break;
        case gpu::BLACKWHITE_BRIGHTNESS:
                ip.applyColorSpaceFilter(h_data,h_result, width,height,channels,0,COLORSPACE_FILTER_BW);
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.2f,LUMINOUS_FILTER_BRIGHTNESS);
                break;
        case gpu::BRIGHTNESS_SATURATION:
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.2f,LUMINOUS_FILTER_BRIGHTNESS);
                ip.applyColorSpaceFilter(h_data,h_result, width,height,channels,50.2f,COLORSPACE_FILTER_SATURATION);
                break;
        case gpu::CONTRAST_SEPIA:
                ip.applyLuminousFilter(h_data,h_result,width,height,channels,1.2f,LUMINOUS_FILTER_BRIGHTNESS);
                ip.applyColorSpaceFilter(h_data,h_result, width,height,channels,0.0f,COLORSPACE_FILTER_SEPIA);
                break;
        }
        dTime2 = gpu::getTime(tim);
        executionTime += dTime2 - dTime1;
        std::cout << "Done with execution on CPU.\nTime Taken: " << dTime2 - dTime1	<< std::endl;


        CImg<unsigned char> outputImage(h_result,imgInfo.width,imgInfo.height,1,
                                        imgInfo.spectrum,0);

        dTime1 = dTime2;

        std::cout << "Writing to Disk"<<std::endl;	
        std::string outputDirectory = directoryPath+"output/";
        int outDir = mkdir(outputDirectory.c_str(),0777);

        if(outDir == 0 || errno == EEXIST)
        {
          outputImage.save_jpeg((directoryPath+"/output/"+outputFilename+extension).c_str());
          dTime2 = gpu::getTime(tim);
          fileIOTime += dTime2 - dTime1;
          std::cout << "Time for Disk Write: " << dTime2 - dTime1 << std::endl;
        }
        else
        {
          if(errno != EEXIST)
          {
            std::cout << "Error creating output directory" << std::endl;
          }
        }

        delete[] h_data;
        delete[] h_result;
      }
    }

    std::cout << "File I/O time: " << fileIOTime << std::endl;
    std::cout << "Configuration time: " << configurationTime << std::endl;
    std::cout << "Execution time: " << executionTime << std::endl;
    std::cout << "CPU Utilization: " << (double)executionTime/(fileIOTime+configurationTime+executionTime) << std::endl;
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " -filter [optional] <image-directory>" << std::endl;
  }
  std::cout << "---------------------ENDING IMAGE APP ON GPU--------------------------------------" << std::endl;
}
