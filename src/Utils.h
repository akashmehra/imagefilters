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
    INVERTBLEND,
  };

  enum ConvolutionKernel
  {
    GAUSSIAN,
    LAPLACIAN,
    EMBOSSED,
    MOTIONBLUR,
    SHARPEN,
  };

  class Options
  {
    public:
      gpu::FilterFlag filterFlag;
      std::string directoryPath;
      ConvolutionKernel convolutionKernelType;
      gpu::BlendType blendMode;
      int* convolutionKernel;
      int kernelSize;

      Options(FilterFlag filterFlag_,
              std::string directoryPath_,
              ConvolutionKernel convolutionKernelType_,
              int kernelSize_)
        :filterFlag(filterFlag_),
        directoryPath(directoryPath_),
        convolutionKernelType(convolutionKernelType_),
        kernelSize(kernelSize_)
    {
      convolutionKernel = new int[kernelSize];
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
      }

      Options& operator= (const Options& options_)
      {
        if(this != &options_)
        {
          filterFlag = options_.filterFlag;
          directoryPath = options_.directoryPath;
          convolutionKernelType = options_.convolutionKernelType;
          int* newKernel = new int[options_.kernelSize];
          std::copy(options_.convolutionKernel,
                    options_.convolutionKernel + options_.kernelSize,
                    newKernel);
          delete[] convolutionKernel;
          convolutionKernel = newKernel;
          kernelSize = options_.kernelSize;	
        }
        return *this;
      }

      Options(){}
      ~Options(){}//delete[] convolutionKernel;}
};


static bool parseCommandLine(int argc, char* argv[], Options* options)
{
  // executablename -filter filterType [blendValue|convolutionkerneltype] [convolutionkernelsize] directorypath.
  // filterType: BRIGHTNESS:0, SEPIA:5, CONTRAST:1, SATURATION:4, CONVOLUTION:2, BLEND:3, these are all integers.
  // blendValue: 0-13, convolutionkerneltype 0-4
  // convolutionkernelsize 3,5,7 are common sizes.
  // directory where images are stored, output will be stored in the same directory under output directory: directoryPath/output/

  bool validArguments = false;
  std::cout << argv[1] << std::endl;
  std::string str(argv[1]);
  if(str  == "-filter")
  {
    std::stringstream ss;
    ss	<< std::string(argv[2]);
    int value;
    ss >> value;
    options->filterFlag = (FilterFlag)value;
    std::cout << "value: " << value << std::endl;
    if(options->filterFlag == BRIGHTNESS
       ||options->filterFlag == SEPIA
       || options->filterFlag == CONTRAST
       || options->filterFlag == SATURATION)
    {
      options->directoryPath = std::string(argv[3]);
      validArguments = true;
    }
    else if(options->filterFlag == BLEND)
    {
      std::stringstream ss;
      ss	<< std::string(argv[3]);
      int blendValue;
      ss >> blendValue;
      options->blendMode = (BlendType)blendValue;
      options->directoryPath = argv[4];	
      validArguments = true;
    }
    else if(options->filterFlag == CONVOLUTION)
    {
      std::stringstream ss;
      ss	<< argv[4];
      int kSize;
      ss >> kSize;
      ss << argv[3];
      int convKernel;
      ss >> convKernel;
      options->convolutionKernelType = (ConvolutionKernel)convKernel;
      options->kernelSize = kSize*kSize;
      options->directoryPath = argv[5];
      options->convolutionKernel = new int[options->kernelSize];
      validArguments = true;
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
