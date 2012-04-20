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

namespace gpu 
{
  
  struct Pixel 
  {
    unsigned char r;
    unsigned char g;
    unsigned char b;
  };
 
  // Luminous Filters Begin
  enum FilterType
  {
    LUMINOUS_FILTER_CONTRAST, 
    LUMINOUS_FILTER_BRIGHTNESS,
  };
  
  template<typename T>
  class LuminousFilters
  {
  private:
    T contrast(const T& pixel, float cValue);
    T brightness(const T& pixel, float bValue);
    float cValue, bValue;
    FilterType filterType;
  public:
    T apply(const T& pixel,float value, FilterType filterType);
  };
  
  template<typename T>
  T LuminousFilters<T>::contrast(const T& pixel, float cValue)
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
  
  template<typename T>
  T LuminousFilters<T>::brightness(const T& pixel, float bValue)
  {
    int val = (int)(pixel+bValue);
    PIXEL_DOMAIN_CHECK(val);
    return val;
  }
  
  template<typename T>
  T LuminousFilters<T>::apply(const T& pixel, float value, FilterType filterType) 
  {
    switch(filterType)
    {
      case LUMINOUS_FILTER_CONTRAST:
        return contrast(pixel, value);
        break;
      case LUMINOUS_FILTER_BRIGHTNESS:
        return brightness(pixel, value);
        break;
    }
  }
  // Luminous Filters End.
  
  
  
  // ColorSpace Filters Begin.
  template <typename T>
  class ColorSpaceFilters
  {
  public:
    void saturation(float sValue, T& pixelR, T& pixelG,T& pixelB,
                T& pixelOutputR, T& pixelOutputG,T& pixelOutputB);
    
    void NormalizePixel(float whitePoint,float blackPoint,float outputWhitePoint,
                        float outputBlackPoint,T& pixel, T& pixelOutput);
    
    void ApplyFunctionOnPixel(float *curveFunction,T& pixel,T& pixelOutput);                             
    
    void BlackNWhite(T &pixelR,T &pixelG,T &pixelB,T &pixelOutputR,
              T& pixelOutputG,T& pixelOutputB);
    
    void Sepia(T &pixelR, T &pixelG,T &pixelB,T &pixelOutputR,
                  T &pixelOutputG,T &pixelOutputB);
  };
  
  template<typename T>
  void ColorSpaceFilters<T>::saturation(float sValue,T& pixelR,T& pixelG,
                                     T& pixelB,T& pixelOutputR,T& pixelOutputG,
                                     T& pixelOutputB)
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
  void ColorSpaceFilters<T>::NormalizePixel(float whitePoint,float blackPoint,float outputWhitePoint,
                                  float outputBlackPoint,T& pixel,T& pixelOutput)
  {
    /***Filter can be implemented inplace.***/
    //Values for both all the input values can be between 0-255;
    //DEFAULT VALUES:
    //BLACK POINT:0   WHITE POINT:255   OUTPUT WHITE POINT:255 OUTPUT BLACK POINT:0
    //can be used to perform histogram normalization.
  } 
  
  template<typename T>
  void ColorSpaceFilters<T>::ApplyFunctionOnPixel(float *curveFunction,T& pixel,T& pixelOutput)
  {
    /***Filter can be implemented inplace. ***/
    ///Curve function defines the the output values from 0-255 got after Spline fitting
    ///input points
    pixelOutput=curveFunction[pixel];
    
  }
  
  template<typename T>
  void ColorSpaceFilters<T>::BlackNWhite(T &pixelR,T &pixelG, T &pixelB,T &pixelOutputR,
                         T& pixelOutputG,T& pixelOutputB)
  {
    /***Filter can be implemented inplace ***/
    ///BW filter is basically 0.6*R + 0.35*G + 0.5*B
    float value = 0.6*pixelR + 0.35*pixelG + 0.05*pixelB;
    pixelOutputR=pixelOutputG=pixelOutputB=value;
  }
  
  template<typename T>
  void ColorSpaceFilters<T>::Sepia(T &pixelR, T &pixelG,T &pixelB,T &pixelOutputR,
                            T &pixelOutputG,T &pixelOutputB)                       
  {
    /***Filter can be implemented inplace ***/
    ///BW filter is basically 0.6*R + 0.35*G + 0.5*B
    ///Sepia we basically add some in RedChannel, and siginficantly less in Green Channel
    
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
