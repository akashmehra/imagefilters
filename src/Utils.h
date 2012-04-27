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
namespace gpu 
{

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
