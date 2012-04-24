//
//  FilterTemplates.h
//  ImageProcessingFilters
//
//  Created by Akash Mehra on 4/19/12.
//  Copyright (c) 2012 New York University. All rights reserved.
//

#ifndef gpu_FilterTemplates_h
#define gpu_FilterTemplates_h

#include "Constants.h"


#ifdef GCC_COMPILATION
#define FUNCTION_PREFIX 
#else 
#define FUNCTION_PREFIX __host__ __device__
#endif


namespace gpu 
{
  
  struct Pixel 
  {
    unsigned char r;
    unsigned char g;
    unsigned char b;
  };
  
  //////////////////////////////////////////////
  ////LUMINOUS FILTERS
  //////////////////////////////////////////////
  enum LuminousFilterTypes
  {
    LUMINOUS_FILTER_CONTRAST, 
    LUMINOUS_FILTER_BRIGHTNESS,
  };
    
  enum ColorSpaceFilterTypes
  {
    COLORSPACE_FILTER_SATURATION,
    COLORSPACE_FILTER_SEPIA,
  };
  
  template<typename T>
  class LuminousFilters
  {
    //////////////////////////////////////////////
    private:
      LuminousFilterTypes filterType;
      T contrast(const T& pixel, float cValue);
      T brightness(const T& pixel, float bValue);
      float cValue, bValue;
    //////////////////////////////////////////////
    public:
      FUNCTION_PREFIX T apply(const T& pixel,
                              float value, 
                              LuminousFilterTypes filterType);
    //////////////////////////////////////////////
  };
 
  //////////////////////////////////////////////
  ///LUMINOUS FILTER TYPE CONTRAST TEMPLATE
  ///cValue Range:
  template<typename T>
  FUNCTION_PREFIX T LuminousFilters<T>::contrast(const T& pixel, float cValue)
  {
    if(cValue > 1)
    {
      float diff = cValue - 1;
      cValue = 1 + diff/2;
    }
    int val = (int)((float)pixel*cValue + (-cValue*SIGNED_CHAR_MAX + SIGNED_CHAR_MAX));   
    PIXEL_DOMAIN_CHECK(val);
    return val;
  }
    
  //////////////////////////////////////////////
  ///LUMINOUS FILTER TYPE BRIGHTNESS TEMPLATE
  ///bValue Range:
  template<typename T>
  FUNCTION_PREFIX T LuminousFilters<T>::brightness(const T& pixel, float bValue)
  {
    int val = (int)(pixel*bValue);
    PIXEL_DOMAIN_CHECK(val);
    return val;
  }
  
    
    
  template<typename T>
  FUNCTION_PREFIX T LuminousFilters<T>::apply(const T& pixel, float value, LuminousFilterTypes filterType) 
  {
    switch(filterType)
    {
      case LUMINOUS_FILTER_CONTRAST:    return contrast(pixel, value);
      case LUMINOUS_FILTER_BRIGHTNESS:  return brightness(pixel, value);
    }
  }
  //////////////////////////////////////////////
  ///LUMINOUS FILTERS END  
  //////////////////////////////////////////////

  
   //////////////////////////////////////////////
   ///COLORSPACE FILTERS  
   //////////////////////////////////////////////  
  template <typename T>
  class ColorSpaceFilters
  {
    //////////////////////////////////////////////  
    private:
      //////////////////////////////////////////////
      ///COLORSPACE FILTER TYPE SATURATION
      ///sValue Range:
      FUNCTION_PREFIX void saturation(T& pixelR, 
                                      T& pixelG,
                                      T& pixelB,
                                      T& pixelOutputR,
                                      T& pixelOutputG,
                                      T& pixelOutputB,
                                      float sValue);
      
      //////////////////////////////////////////////
      ///COLORSPACE FILTER TYPE NORMALIZE
      ///POINT RANGES:0-255;
      FUNCTION_PREFIX void NormalizePixel(float whitePoint,
                                          float blackPoint,
                                          float outputWhitePoint,
                                          float outputBlackPoint,
                                          T& pixel, 
                                          T& pixelOutput);
      
      //////////////////////////////////////////////
      ///COLORSPACE FILTER TYPE FUNCTION
      FUNCTION_PREFIX void ApplyFunctionOnPixel(float *curveFunction,
                                                T& pixel,
                                                T& pixelOutput); 
      
      //////////////////////////////////////////////
      ///COLORSPACE FILTER TYPE BW
      FUNCTION_PREFIX void BlackNWhite(T &pixelR,
                                       T &pixelG,
                                       T &pixelB,
                                       T &pixelOutputR,
                                       T& pixelOutputG,
                                       T& pixelOutputB);
      
      //////////////////////////////////////////////
      ///COLORSPACE FILTER TYPE SEPIA
      FUNCTION_PREFIX void sepia(T &pixelR, 
                                 T &pixelG,
                                 T &pixelB,
                                 T &pixelOutputR,
                                 T &pixelOutputG,
                                 T &pixelOutputB);
      
      
	public:
		FUNCTION_PREFIX void apply(T& pixelR, T& pixelG,T& pixelB,
                               T& pixelOutputR, T& pixelOutputG,T& pixelOutputB,float sValue, ColorSpaceFilters filterType);

  };
  
	template<typename T>
	FUNCTION_PREFIX void ColorSpaceFilters<T>::apply(T& pixelR, 
                                                     T& pixelG,
                                                     T& pixelB,
                                                     T& pixelOutputR, 
                                                     T& pixelOutputG,
                                                     T& pixelOutputB,
                                                     float sValue,
                                                     ColorSpaceFilters filterType)
	{
		switch(filterType)
		{
        case COLORSPACE_FILTER_SATURATION:
        saturation(pixelR,
                   pixelG, 
                   pixelB,
                   pixelOutputR,
                   pixelOutputG, 
                   pixelOutputB,
                   sValue);
        break;
                
        case COLORSPACE_FILTER_SEPIA:
		sepia(pixelR, 
              pixelG, 
              pixelB,
              pixelOutputR, 
              pixelOutputG, 
              pixelOutputB);
        break;
		}
	}
	
	template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::saturation(T& pixelR,T& pixelG,
                                                        T& pixelB,T& pixelOutputR,T& pixelOutputG,
                                                        T& pixelOutputB,float sValue)
  {
    /***Filter can be implemented inplace.***/
    //sValue can be between -1 and 1.
    //1 means no change. 0 signifies black & white. 2 signifies max saturation.
    sValue = sValue/100;
    float temp = 0.0f;
    float bwValue=pixelR*0.33+pixelG*0.33+pixelB*0.33;
    
    ///Adjust Saturation of Every Channel    
    if(pixelR>bwValue)
    {
      temp=pixelR+(pixelR-bwValue)*sValue; 
    } 
    else
    {
      temp = pixelR-(bwValue-pixelR)*sValue;
    }
    PIXEL_DOMAIN_CHECK(temp);    
    pixelOutputR=temp;
    
    if(pixelG>bwValue)
    {
      temp = pixelG+(pixelG-bwValue)*sValue;
    }           
    else
    {
      temp = pixelG-(bwValue-pixelG)*sValue;
    }
    PIXEL_DOMAIN_CHECK(temp);
    pixelOutputG=temp;
    
    if(pixelB>bwValue)
    {
      temp=pixelB+(pixelB-bwValue)*sValue;            
    }
    else
    {
      temp=pixelB-(bwValue-pixelB)*sValue;
    }
    PIXEL_DOMAIN_CHECK(temp);
    pixelOutputB=temp;  
  }
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::NormalizePixel(float whitePoint,float blackPoint,
                                                            float outputWhitePoint,
                                                            float outputBlackPoint,T& pixel,
                                                            T& pixelOutput)
  {
    /***Filter can be implemented inplace.***/
    //Values for both all the input values can be between 0-255;
    //DEFAULT VALUES:
    //BLACK POINT:0   WHITE POINT:255   OUTPUT WHITE POINT:255 OUTPUT BLACK POINT:0
    //can be used to perform histogram normalization.
  } 
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::ApplyFunctionOnPixel(float *curveFunction,
                                                                  T& pixel,T& pixelOutput)
  {
    /***Filter can be implemented inplace. ***/
    ///Curve function defines the the output values from 0-255 got after Spline fitting
    ///input points
    pixelOutput=curveFunction[pixel];
    
  }
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::BlackNWhite(T &pixelR,T &pixelG, 
                                                         T &pixelB,T &pixelOutputR,
                                                         T& pixelOutputG,T& pixelOutputB)
  {
    /***Filter can be implemented inplace ***/
    ///BW filter is basically 0.6*R + 0.35*G + 0.5*B
    float value = 0.6*pixelR + 0.35*pixelG + 0.05*pixelB;
    pixelOutputR=pixelOutputG=pixelOutputB=value;
  }
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::sepia(T &pixelR, T &pixelG,T &pixelB,T &pixelOutputR,
                                                   T &pixelOutputG,T &pixelOutputB)                       
  {
    /***Filter can be implemented inplace ***/
    ///BW filter is basically 0.6*R + 0.35*G + 0.5*B
    ///sepia we basically add some in RedChannel, and siginficantly less in Green Channel
    
    float temp = 1.25 * pixelR;
    PIXEL_DOMAIN_CHECK(temp);
    pixelOutputR = temp;
    
    temp = 1.05 * pixelG;
    PIXEL_DOMAIN_CHECK(temp);
    pixelOutputG = temp;
    
    temp = 1.25 * pixelB;
    PIXEL_DOMAIN_CHECK(temp);
    pixelOutputB = temp;
    
  }
  
  // ColorSpace Filters End.
  
}



#endif //gpu_FilterTemplates_h
