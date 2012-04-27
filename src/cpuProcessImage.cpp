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


int main(int argc, char* argv[])
{
//  cimg::imagemagick_path("/opt/local/bin/convert");
  if(argc == 2)
  {
		std::vector<std::string> fileList;
		std::string directoryPath = argv[1];
		gpu::readDirectory(directoryPath,&fileList);
		std::vector<std::string>::iterator it = fileList.begin();
   	int count = 0;
    int convKernel[] = {-2,-1,0,-1,1,1,0,1,2};	
		for(;it != fileList.end();++it)
		{
			std::string imageFilename = directoryPath+*it;
			std::string extension = imageFilename.substr(imageFilename.length()-4,4); 
			if(extension == JPG_EXTENSION || extension == JPEG_EXTENSION)
			{
				if(count % 100 == 0)
				{
					std::cout << "Frame " << count << std::endl;
				}
				++count;
				//std::cout << "filename : " << imageFilename << std::endl;
				std::string filename = imageFilename.substr(0,imageFilename.length()-4);	
   			int indexOfSlash = imageFilename.find_last_of("/");
				std::string outputFilename = imageFilename.substr(indexOfSlash,imageFilename.length()-4-indexOfSlash);//+"_output";
				//std::cout << "output filename: " << outputFilename << std::endl;
    		CImg<unsigned char> image(imageFilename.c_str());
    		//CImgDisplay mainDisplay(image,"Image",0);

    		gpu::Image imgInfo(image.width(),image.height(),image.width()*image.height(),image.spectrum());
    		gpu::printMetaData(imgInfo);
    
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
    
    		gpu::unrollMatrix(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,
                inputBuffer);
   
      	//imp.applyLuminousFilter(inputBuffer, outputBuffer, imgInfo.width, imgInfo.height, imgInfo.spectrum, BRIGHTNESS_VALUE,gpu::LUMINOUS_FILTER_BRIGHTNESS); 
    		//imp.applyColorSpaceFilter(inputBuffer, outputBuffer, imgInfo.width, imgInfo.height, imgInfo.spectrum, S_VALUE,gpu::COLORSPACE_FILTER_SATURATION); 
      	//int convKernel[]={3,3,3,3,3,3,3,3,3};
				imp.applyConvolution(inputBuffer,outputBuffer,convKernel, imgInfo.width, imgInfo.height, imgInfo.spectrum,3,1);
    
				//double dTime2 = gpu::getTime(tim);
    		//std::cout << "time taken for convolution: " << dTime2 - dTime1 << std::endl;
    
      	//imp.applyBlendFilter(inputBuffer,inputBuffer,outputBuffer, imgInfo.width, imgInfo.height, imgInfo.spectrum, 1.0,gpu::BLEND_FILTER_LINEARLIGHT); 

    		CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,
																			imgInfo.height,1,
                                    	imgInfo.spectrum,0);
    
    		//double dTime3 = gpu::getTime(tim);
    		//std::cout << "time taken for blend: " << dTime3 - dTime2 << std::endl;
				std::string outputDirectory = directoryPath+"output/";
   			int outDir = mkdir(outputDirectory.c_str(),0777);
			 	if(outDir == 0 || errno == EEXIST)
			 	{	
					//std::cout << filename << std::endl;	
    			outputImage.save_jpeg((directoryPath+"/output/"+outputFilename+extension).c_str());
			 	}
			 	else
			 	{
					if(errno != EEXIST)
					{
			 			std::cout << "Error creating output directory" << std::endl;
					}
			 	}
    		//CImgDisplay darkDisplay(outputImage,"Output Image",0);
    
    		/*while(!(mainDisplay.is_closed()))
    		{
      		mainDisplay.wait();
    		}*/
    		delete[] inputBuffer;
    		delete[] outputBuffer;
			}
		}
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " <image-filename> <output-filename>" << std::endl;
  }
}


