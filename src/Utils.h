//
//  Utils.h
//  ImageProcessingFilters
//
//  Created by Akash Mehra on 4/19/12.
//  Copyright (c) 2012 New York University. All rights reserved.
//

#ifndef gpu_Utils_h
#define gpu_Utils_h

#include <iostream>
#include "Constants.h"
#include "CImg.h"
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/types.h>
#include <cerrno> 
#include <sstream>

namespace gpu 
{
  enum FilterType
  {
    LUMINOUS_FILTER_CONTRAST, 
    LUMINOUS_FILTER_BRIGHTNESS,
    COLORSPACE_FILTER_SATURATION,
    COLORSPACE_FILTER_SEPIA,
    COLORSPACE_FILTER_BW,
  };

  enum BlendType
  {
    BLEND_FILTER_NORMAL,
    BLEND_FILTER_LIGHTEN,
    BLEND_FILTER_DARKEN,
    BLEND_FILTER_MULTIPLY,
    BLEND_FILTER_AVERAGE,
    BLEND_FILTER_ADD,
    BLEND_FILTER_SUBTRACT,
    BLEND_FILTER_DIFFERENCE,
    BLEND_FILTER_NEGATION,
    BLEND_FILTER_SCREEN,
    BLEND_FILTER_EXCLUSION,
    BLEND_FILTER_OVERLAY,
    BLEND_FILTER_SOFTLIGHT,
    BLEND_FILTER_HARDLIGHT,
    BLEND_FILTER_COLORDODGE,
    BLEND_FILTER_COLORBURN,
    BLEND_FILTER_LINEARDODGE,
    BLEND_FILTER_LINEARBURN,
    BLEND_FILTER_LINEARLIGHT,
    BLEND_FILTER_VIVIDLIGHT,
    BLEND_FILTER_PINLIGHT,
    BLEND_FILTER_HARDMIX
  };
  enum FilterFlag
  {
    BRIGHTNESS,
    CONTRAST,
    CONVOLUTION,
    BLEND,
    SATURATION,
    SEPIA,
    BLACKWHITE,
    BRIGHTNESS_CONTRAST,
    BLACKWHITE_BRIGHTNESS,
    BRIGHTNESS_SATURATION,
    CONTRAST_SEPIA,
  };

  enum ConvolutionKernel
  {
    GAUSSIAN,
    EMBOSSED,
    MOTIONBLUR,
    SHARPEN,
    EDGEDETECTION,
    EDGEENHANCE,
  };

  static void initEdgeDetectionKernel(int* kernel,int size)
  {
    int edgeKernel[] = {0,1,0,1,-4,1,0,1,0};
    std::copy(edgeKernel,edgeKernel+size,kernel);
  }

  static void initSharpenKernel(int* kernel,int size)
  {
    int sharpenKernel[] = {0,-1,0,-1,5,-1,0,-1,0};
    std::copy(sharpenKernel,sharpenKernel+size,kernel);
  }

  static void initEdgeEnhanceKernel(int* kernel,int size)
  {
    int edgeEnhanceKernel[] = {0,0,0,-1,1,0,0,0,0};
    std::copy(edgeEnhanceKernel,edgeEnhanceKernel+size,kernel);
  }

  static void initEmbossedKernel(int* kernel,int size)
  {
    int embossedKernel[] = {-2,-1,0,-1,1,1,0,1,2}; 
    std::copy(embossedKernel,embossedKernel+size,kernel);
  }

  static void initMotionBlurKernel(int* kernel,int size)
  {
    int motionBlurKernel[] = {1,0,0,0,1,0,0,0,1};;
    std::copy(motionBlurKernel,motionBlurKernel+size,kernel);
  }


  static void initGaussianKernel(int* kernel,int size)
  {
    int gaussianKernel[] = {1,1,1,1,1,1,1,1,1};
    std::copy(gaussianKernel,gaussianKernel+size,kernel);
  }

  void initConvolutionKernel(int* kernel, int size, ConvolutionKernel kernelType)
  {
    switch(kernelType)
    {
    case gpu::GAUSSIAN:
      gpu::initGaussianKernel(kernel,size);
      break;
    case gpu::EMBOSSED:
      gpu::initEmbossedKernel(kernel,size);
      break;
    case gpu::MOTIONBLUR:
      gpu::initMotionBlurKernel(kernel,size);
      break;
    case gpu::SHARPEN:
      gpu::initSharpenKernel(kernel,size);
      break;
    case gpu::EDGEDETECTION:
      gpu::initEdgeDetectionKernel(kernel,size);
      break;
    case gpu::EDGEENHANCE:
      gpu::initEdgeEnhanceKernel(kernel,size);
      break;
    }
  }




  class Options
  {
    public:
      gpu::FilterFlag filterFlag;
      std::string directoryPath;
      ConvolutionKernel convolutionKernelType;
      gpu::BlendType blendMode;
      int* convolutionKernel;
      int kernelSize;
      std::string errorMessage;
      bool isConvolutionOp;

      Options(FilterFlag filterFlag_,
              std::string directoryPath_,
              ConvolutionKernel convolutionKernelType_,
              int kernelSize_)
        :filterFlag(filterFlag_),
        directoryPath(directoryPath_),
        convolutionKernelType(convolutionKernelType_),
        kernelSize(kernelSize_)
    {
      convolutionKernel = new int[kernelSize*kernelSize];
    }

      Options(const Options& options_)
      {
        filterFlag = options_.filterFlag;
        directoryPath = options_.directoryPath;
        convolutionKernelType = options_.convolutionKernelType;
        std::copy(options_.convolutionKernel,
                  options_.convolutionKernel + options_.kernelSize,
                  convolutionKernel);
        kernelSize = options_.kernelSize;
        errorMessage = options_.errorMessage;
      }

      Options& operator= (const Options& options_)
      {
        if(this != &options_)
        {
          filterFlag = options_.filterFlag;
          directoryPath = options_.directoryPath;
          convolutionKernelType = options_.convolutionKernelType;
          int* newKernel = new int[options_.kernelSize*options_.kernelSize];
          std::copy(options_.convolutionKernel,
                    options_.convolutionKernel + options_.kernelSize,
                    newKernel);
          delete[] convolutionKernel;
          convolutionKernel = newKernel;
          kernelSize = options_.kernelSize;
          errorMessage = options_.errorMessage;
        }
        return *this;
      }

      Options():convolutionKernel(0){}
      ~Options(){if(convolutionKernel) delete[] convolutionKernel;}
  };

  static bool convertToInt(const char* argument, int* value)
  {
    std::stringstream ss;
    ss	<< std::string(argument);
    std::cout << argument << std::endl;
    bool convertSuccess = !(ss >> *value).fail();
    std::cout << convertSuccess << std::endl;
    return convertSuccess;
  }


  static bool parseCommandLine(int argc, char* argv[], Options* options)
  {
    bool validArguments = false;
    std::cout << argv[1] << std::endl;
    std::string str(argv[1]);
    if(str  == "-filter")
    {
      int value;
      bool convertSuccess = convertToInt(argv[2],&value);
      if(convertSuccess)
      {
        options->filterFlag = (FilterFlag)value;
        std::cout << "value: " << value << std::endl;
        if(options->filterFlag == BRIGHTNESS
           ||options->filterFlag == SEPIA
           || options->filterFlag == CONTRAST
           || options->filterFlag == SATURATION
           || options->filterFlag == BLACKWHITE 
           || options->filterFlag == BLACKWHITE_BRIGHTNESS 
           || options->filterFlag == BRIGHTNESS_CONTRAST
           || options->filterFlag == BRIGHTNESS_SATURATION
           || options->filterFlag == CONTRAST_SEPIA
          )
        {
          options->directoryPath = std::string(argv[3]);
          validArguments = true;
        }
        else if(options->filterFlag == BLEND)
        {
          int blendValue;
          bool convertSuccess = convertToInt(argv[3], &blendValue);
          if(convertSuccess)
          {
            if(blendValue >= 0 && blendValue <=21)
            {
              options->blendMode = (BlendType)blendValue;
              options->directoryPath = argv[4];	
              validArguments = true;
            }
            else
            {
              options->errorMessage = "Blend Value should be between 0 and 21 (both inclusive).";
            }
          }
          else
          {
            options->errorMessage = "Blend Value should be integer.";
          }
        }
        else if(options->filterFlag == CONVOLUTION)
        {
          int convKernel;
          bool convertSuccess = convertToInt(argv[3],&convKernel);	
          if(convertSuccess)
          {
            int kSize;
            convertSuccess = convertToInt(argv[4],&kSize);
            if(convKernel >=0 && convKernel <=5)
            {
              options->convolutionKernelType = (ConvolutionKernel)convKernel;
              std::cout << "------CONV KERNEL TYPE: " << options->convolutionKernelType << std::endl;
              if(convertSuccess)
              {
                if(kSize == 3)
                {
                  options->kernelSize = kSize;
                  options->directoryPath = argv[5];
                  options->isConvolutionOp = true;	
                  options->convolutionKernel = new int[options->kernelSize*options->kernelSize];
                  initConvolutionKernel(options->convolutionKernel,options->kernelSize*options->kernelSize,options->convolutionKernelType);
                  validArguments = true;
                }
                else
                {
                  options->errorMessage = "Convolution kernel size should be 3.";
                }	
              }
              else
              {
                options->errorMessage = "Convolution kernel size should be integer.";
              }
            }
            else
            {
              options->errorMessage = "Convolution kernel value should be between 0 and 5 (both inclusive).";
            }
          }
          else
          {
            options->errorMessage = "Convolution kernel type should be an integerbetween 0 and 5 (both inclusive).";
          }
        }
        else
        {
          options->errorMessage = "Filter type should be a number between 0 and 5 (both inclusive).";
        }
      }
      else
      {
        options->errorMessage = "Filter type should be a number between 0 and 5 (both inclusive).";
      }
    }
    std::cout << "directory: " << options->directoryPath << std::endl;
    std::cout << "filter flag: " << options->filterFlag << std::endl;
    return validArguments;
  }


  static void readDirectory(std::string directoryPath,
                            std::vector<std::string>* fileList)
  {
    DIR* dir;
    struct dirent* dirp;
    if((dir = opendir(directoryPath.c_str())) == NULL)
    {
      std::cout << "Error Opening directory, error: " << errno << std::endl;
    }

    while((dirp = readdir(dir)) != NULL)
    {
      fileList->push_back(std::string(dirp->d_name));
    }
    closedir(dir);
  }


  template <typename T>
    static void unrollMatrix(cimg_library::CImg<T>& image, unsigned int width, unsigned int height,
                             unsigned int spectrum, T* buffer)
    {
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

  struct Image
  {
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned int spectrum;

    Image(unsigned int width_,unsigned int height,
          unsigned int size_,unsigned int spectrum_)
      :width(width_),
      height(height),
      size(size_),
      spectrum(spectrum_)
    {}
  };

  struct Setup
  {
    int threads;
    int blocks;
  };

  static unsigned int powerOf2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
  }

  static void getSetupConfig(unsigned int problemSize, struct Setup* setup)
  {
    int threads = powerOf2(problemSize);
    threads = threads <= MAX_THREADS ? threads : MAX_THREADS;

    int blocks = (problemSize)/threads;
    if(problemSize % threads != 0)
    {
      blocks +=1;
    }

    setup->threads = threads;
    setup->blocks = blocks;
  }

  static void printMetaData(const Image& image)
  {
    std::cout << "Image Metadata:" << std::endl;
    std::cout << "width: " << image.width << ", height: " << image.height << 
      ", size: " << image.size << ", spectrum: " << image.spectrum << std::endl;
  }



  static int diff_ms(timeval t1, timeval t2)
  {
    return (((t1.tv_sec - t2.tv_sec) * MILLION) + 
            (t1.tv_usec - t2.tv_usec))/THOUSAND;
  }

  static double getTime(timeval& tim)
  {
    gettimeofday(&tim,NULL);
    return tim.tv_sec+(tim.tv_usec/MILLION_D);  
  }
}


#endif
