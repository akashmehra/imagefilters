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

    int* warmupBuffer =(int*) calloc(1000, sizeof(int));

    timeval tim;
    double dTime1 = gpu::getTime(tim);

    std::cout << "Sending warm up signal to GPU." << std::endl;

    sendWarmUpSignal(warmupBuffer,1000*sizeof(int)); 

    double dTime2 = gpu::getTime(tim);
    double warmupTime = dTime2 - dTime1;
    std::cout << "time taken for performing warm up: " << warmupTime << std::endl;

    delete[] warmupBuffer;

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
        std::cout << "filename: " << filename << std::endl;
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
        std::cout << "Applying Convolution Filter..." << std::endl;
       	int h_kernel[] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
        int sizeKernel = sizeof(h_kernel)/sizeof(*h_kernel);
        int windowSize = static_cast<int>(sqrt(sizeKernel));
        int* d_kernel;
        cudaMalloc((void**)&d_kernel,windowSize);
        cudaMemcpy(d_kernel,h_kernel,windowSize,cudaMemcpyHostToDevice);
        
				switch(options.filterFlag)
        {									

        case gpu::BRIGHTNESS:
          runBrightnessKernel(setup,d_data,d_result,
                              width,height,channels,offset);
          break;

        case gpu::CONTRAST:
          runContrastKernel(setup,d_data,d_result,
                            width,height,channels,offset);
          break;

        case gpu::CONVOLUTION:
            runConvolutionKernel(setup,d_data,d_result,
                                 d_kernel,windowSize,
                                 width,height,channels,offset);
            break;
        
        case gpu::BLEND:
            runBlendKernel(setup,d_data,d_data,d_result,
                           width,height,channels,offset,1.2f,options.blendMode);

            break;

        case gpu::SATURATION:
            runSaturationKernel(setup,d_data,d_result,
                                width,height,channels,offset);
            break;

        case gpu::SEPIA:
            runSepiaKernel(setup,d_data,d_result,
                           width,height,channels,offset);
            break;
        }
        dTime2 = gpu::getTime(tim);
        executionTime += dTime2 - dTime1;
        std::cout << "Done with execution on GPU.\nTime Taken: " << dTime2 - dTime1	<< std::endl;

        dTime1 = dTime2;					
        cudaMemcpy(h_result,d_result,sizeResult,cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        cudaFree(d_result);
        cudaFree(d_kernel);

        dTime2 = gpu::getTime(tim);
        configurationTime += dTime2 - dTime1;
        std::cout << "Data transferred back to Host.\nTime taken: " << dTime2 - dTime1 << std::endl;



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
    std::cout << "GPU Utilization: " << (double)executionTime/(fileIOTime+configurationTime+executionTime) << std::endl;
  }
  else
  {
    std::cout << "Usage: " << argv[0] << " -filter [optional] <image-directory>" << std::endl;
  }
}
