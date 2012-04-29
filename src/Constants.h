//
//  Constants.h
//  ImageProcessingFilters
//
//  Created by Akash Mehra on 4/19/12.
//  Copyright (c) 2012 New York University. All rights reserved.
//

#ifndef gpu_Constants_h
#define gpu_Constants_h

#include <cmath>
#include <string>


#define BUFFER_SIZE 256
#define MILLION 1000000
#define THOUSAND 1000
#define MILLION_D 1000000.0
#define UNSIGNED_CHAR_MAX 255
#define UNSIGNED_CHAR_MIN 0
#define S_VALUE 50
#define SIGNED_CHAR_MAX 128
#define MAX_THREADS 256
#define BRIGHTNESS_VALUE 1.2f
#define CONTRAST_VALUE 1.5f
#define SATURATION_VALUE 50.2f
#define PIXEL_DOMAIN_CHECK(T){if(T>255)T=255;else if (T<0)T=0;}

#define GAUSSIAN_KERNEL  {1,1,1,1,1,1,1,1,1};
#define SHARPNESS_KERNEL {0,-1,0,-1,5,-1,0,-1,0};
#define EMBOSSED_KERNEL  {-2,-1,0,-1,1,1,0,1,2};
#define EDGE_DETECTION   {0,1,0,1,-4,1,0,1,0};
#define EDGE_ENHANCE 		 {0,0,0,-1,1,0,0,0,0};
#define MOTION_BLUR 		 {1,0,0,0,1,0,0,0,1};

static const std::string JPG_EXTENSION = ".jpg";
static const std::string JPEG_EXTENSION = ".jpeg";



/**
 * Blend filters are particularly useful in combining
 * two images together. Photoshop, Gimp implement
 * exactly the same way as implemented below.
 * These blend modes work on predefined blend functions
 * Reference::HTTP://INLANDSTUDIOS.COM/EN/?P=851
 * */


#define maximum(B,L) (((B>L) ? B:L))
#define minimum(B,L) (((B<L) ? B:L))
#define ChannelBlend_Normal(B,L)     (B)
#define ChannelBlend_Lighten(B,L)    (((L > B) ? L:B))
#define ChannelBlend_Darken(B,L)     (((L > B) ? B:L))
#define ChannelBlend_Multiply(B,L)   (((B * L) / 255))
#define ChannelBlend_Average(B,L)    (((B + L) / 2))
#define ChannelBlend_Add(B,L)        ((minimum(255, (B + L))))
#define ChannelBlend_Subtract(B,L)   (((B + L < 255) ? 0:(B + L - 255)))
#define ChannelBlend_Difference(B,L) ((abs(B - L)))
#define ChannelBlend_Negation(B,L)   ((255 - abs(255 - B - L)))
#define ChannelBlend_Screen(B,L)     ((255 - (((255 - B) * (255 - L)) >> 8)))
#define ChannelBlend_Exclusion(B,L)  ((B + L - 2 * B * L / 255))

#define ChannelBlend_Overlay(B,L)    (((L < 128) ? (2 * B * L / 255):(255 - 2 * (255 - B) * (255 - L) / 255)))

#define ChannelBlend_SoftLight(B,L)  (((L < 128)?(2*((B>>1)+64))*((float)L/255):(255-(2*(255-((B>>1)+64))*(float)(255-L)/255))))

#define ChannelBlend_HardLight(B,L)  (ChannelBlend_Overlay(L,B))
#define ChannelBlend_ColorDodge(B,L) (((L == 255) ? L:minimum(255, ((B << 8 ) / (255 - L)))))

#define ChannelBlend_ColorBurn(B,L)  (((L == 0) ? L:maximum(0, (255 - ((255 - B) << 8 ) / L))))

#define ChannelBlend_LinearDodge(B,L)(ChannelBlend_Add(B,L))
#define ChannelBlend_LinearBurn(B,L) (ChannelBlend_Subtract(B,L))
#define ChannelBlend_LinearLight(B,L)((L < 128)?ChannelBlend_LinearBurn(B,(2 * L)):ChannelBlend_LinearDodge(B,(2 * (L - 128))))

#define ChannelBlend_VividLight(B,L) ((L < 128)?ChannelBlend_ColorBurn(B,(2 * L)):ChannelBlend_ColorDodge(B,(2 * (L - 128))))

#define ChannelBlend_PinLight(B,L)   ((L < 128)?ChannelBlend_Darken(B,(2 * L)):ChannelBlend_Lighten(B,(2 * (L - 128))))

#define ChannelBlend_HardMix(B,L)    (((ChannelBlend_VividLight(B,L) < 128) ? 0:255))


#endif //gpu_Constants_h
