#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <cstdlib>
#include "CImg.h"

#include "gpuProcessImage.cu"
#include "Utils.h"
#include "cpuProcessImage.h"


int main(int argc, char* argv[])
{

		gpu::Options options;
	 	bool validArguments	= parseCommandLine(argc, argv, &options);	 
  	if(validArguments)
		{
			std::vector<std::string> fileList;
  		std::string directoryPath = options.directoryPath;

  		gpu::readDirectory(directoryPath,&fileList);
  		std::vector<std::string>::iterator it = fileList.begin();
     	int count = 0;
  		
			int warmpupBuffer = calloc(1000, sizeof(float));

      timeval tim;
      double dTime1 = gpu::getTime(tim);
					
			std::cout << "Sending warm up signal to GPU." << std::endl;
   	 	
			sendWarmUpSignal(warmupBuffer,1000*sizeof(float)); 

			double dTime2 = gpu::getTime(tim);
			double warmpupTime = dTime2 - dTime1;
    	std::cout << "time taken for performing warm up: " << warmupTime << std::endl;
			
   	 	delete[] warmpupBuffer;
			
			std::cout << "Starting File I/O using CPU." << std::endl;
			double fileIOTime = 0.0;

			double configurationTime = 0.0;

			double executionTime = 0.0;

			for(;it != fileList.end();++it)
  		{
  			std::string imageFilename = directoryPath+*it;
  			std::string extension = imageFilename.substr(imageFilename.length()-4,4); 
  			if(extension == JPG_EXTENSION || extension == JPEG_EXTENSION)
  			{
  				std::string filename = imageFilename.substr(0,imageFilename.length()-4);	
     			int indexOfSlash = imageFilename.find_last_of("/");
  				
					std::cout << "Reading Image from Disk." << std::endl;
					dTime1 = gpu::getTime(tim);
	
					std::string outputFilename = imageFilename.substr(indexOfSlash,
																														imageFilename.length()-4-indexOfSlash);
      		dTime2 = gpu::getTime(tim);
					fileIOTime += dTime2 - dTime1;

					std::cout << "Time taken to read from disk:  " << dTime2 - dTime1 << std::endl;

					
					dTime1 = gpu::getTime(tim);
					CImg<unsigned char> image(imageFilename.c_str());
  
					std::cout << "Unrolling Image and setting up blocks and threads." << std::endl;
					gpu::Image imgInfo(image.width(),image.height(),image.width()*image.height(),image.spectrum());
      		
					/*
       			<summary> 
       			1. Allocate Buffers
       			2. Get Meta information from the image and assign that to ImageInfo object.
       			3. Copy image into Input Buffer (unroll operation).
       			4. Perform the operation.
       		*/
      
      		unsigned char* inputBuffer = new unsigned char[imgInfo.spectrum*imgInfo.size];
      		unsigned char* outputBuffer = new unsigned char[imgInfo.spectrum*imgInfo.size];
      
      		gpu::unrollMatrix(image,imgInfo.width,imgInfo.height,imgInfo.spectrum,
                  inputBuffer);
    			
    			

					int problemSize, sizeData, sizeResult;
          gpu::Setup setup;
          startSetup(width, height, channels,&problemSize, &sizeData, &sizeResult, &setup);
          dTime2 = gpu::getTime(tim);
					
					configurationTime += dTime2 - dTime1;
			
          std::cout << "Blocks: " << setup.blocks << ", Threads: " << setup.threads << std::endl;
					std::cout << "Done configuring the problem.\nTime taken: " << dTime2 - dTime1 << std::endl;


					dTime1 = gpu::getTime(tim);
					std::cout << "Starting memory allocation and data transfer from Host to Device." << std::endl;
          unsigned char *d_data;
          cudaMalloc((void**)&d_data,sizeData);
          cudaMemcpy(d_data,h_data,sizeData,cudaMemcpyHostToDevice);
          
          unsigned char* d_result;
          cudaMalloc((void**)&d_result,sizeData);
          
					dTime2 = gpu::getTime(tim);
					std::cout << "Done transferring data.\nTime taken: " << dTime2 - dTime1 <<std::endl;
					
					dTime1 = gpu::getTime(tim);
					std::cout << "Begining execution on GPU." << std::endl;
          int offset = width*height;
          switch(filterFlag)
					{									

						/*
						case BRIGHTNESS:
          		runBrightnessKernel(setup,d_data,d_result,h_result,
															width,height,channels,offset);
          		break;
						
						case CONTRAST:
          		runContrastKernel(setup,d_data,d_result,h_result,
																width,height,channels,offset);
          		break;
						*/


						case CONVOLUTION:
							cudaMalloc((void**)&d_kernel,sizeKernel);
							cudaMemcpy(d_kernel,h_kernel,sizeKernel,cudaMemcpyHostToDevice);
							runConvolutionKernel(setup,d_data,d_result,h_result,d_kernel
																width,height,channels,offset);
          		cudaFree(d_kernel); 
              break;
						/*
						case BLEND:
							runBlendKernel(setup,d_data,d_result,h_result,
                             width,height,channels,offset);

							break;

						case SATURATION:
							runSaturationKernel(setup,d_data,d_result,h_result,
          	                      width,height,channels,offset);
							break;

						case SEPIA:
							runSepiaKernel(setup,d_data,d_result,h_result,
                             width,height,channels,offset);
							break;
							*/

					}
					dTime2 = gpu::getTime(tim);
					executionTime += dTime2 - dTime1;
					std::cout << "Done with execution on GPU.\nTime Taken: " << dTime2 - dTime1	<< std::endl;

					dTime1 = dTime2;					
          cudaMemcpy(h_result,d_result,sizeResult,cudaMemcpyDeviceToHost);
          cudaFree(d_data);
          cudaFree(d_result);

					dTime2 = gpu::getTime(tim);
					configurationTime += dTime2 - dTime1;
					std::cout << "Data transferred back to Host.\nTime taken: " << dTime2 - dTime1 << std::endl;
					


    			CImg<unsigned char> outputImage(outputBuffer,imgInfo.width,imgInfo.height,1,
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
					
    			delete[] inputBuffer;
    			delete[] outputBuffer;
			}
			std::cout << "File I/O time: " << fileIOTime << std::endl;
			std::cout << "Configuration time: " << configurationTime << std::endl;
			std::cout << "Execution time: " << executionTime << std::endl;
			std::cout << "GPU Utilization: " << (double)executionTime/(fileIOTime+configurationTime+executionTime) << std::endl;
		}
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " <image-filename> <output-filename>" << std::endl;
  }
}
