//
//  Constants.h
//  ImageProcessingFilters
//
//  Created by Akash Mehra on 4/19/12.
//  Copyright (c) 2012 New York University. All rights reserved.
//

#ifndef gpu_Constants_h
#define gpu_Constants_h

#define BUFFER_SIZE 256
#define MILLION 1000000
#define THOUSAND 1000
#define MILLION_D 1000000.0
#define UNSIGNED_CHAR_MAX 255
#define UNSIGNED_CHAR_MIN 0
#define SIGNED_CHAR_MAX 128
#define PIXEL_DOMAIN_CHECK(T){if(T>255)T=255;else if (T<0)T=0;}

#endif
