//
//  Utils.h
//  ImageProcessingFilters
//
//  Created by Akash Mehra on 4/19/12.
//  Copyright (c) 2012 New York University. All rights reserved.
//

#ifndef gpu_Utils_h
#define gpu_Utils_h

namespace gpu 
{

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

	static void printMetaData(const gpu::Image& image)
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
