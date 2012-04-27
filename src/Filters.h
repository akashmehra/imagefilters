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
#include "Utils.h"

#ifdef GCC_COMPILATION
	#define FUNCTION_PREFIX 
#else 
	#define FUNCTION_PREFIX __host__ __device__
#endif


namespace gpu 
{
  
  //FILTERS
  
 
  template<typename T>
  class LuminousFilters
  {
  private:
    FilterType filterType;
    T contrast(const T& pixel, float cValue);
    T brightness(const T& pixel, float bValue);
    
  public:
    T apply(const T& pixel,float value, 
            FilterType filterType);  
  };
  
  
  /*
   * LUMINOUS FILTER TYPE CONTRAST TEMPLATE
   * cValue Range:
   * */
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
  
  /**
   * LUMINOUS FILTER TYPE BRIGHTNESS TEMPLATE
   * bValue Range:
   * */
  template<typename T>
  FUNCTION_PREFIX T LuminousFilters<T>::brightness(const T& pixel, float bValue)
  {
    int val = (int)(pixel*bValue);
    PIXEL_DOMAIN_CHECK(val);
    return val;
  }
  
  
  
  template<typename T>
  FUNCTION_PREFIX T LuminousFilters<T>::apply(const T& pixel, float value, FilterType filterType) 
  {
    switch(filterType)
    {
      case LUMINOUS_FILTER_CONTRAST:
				return contrast(pixel, value);
      case LUMINOUS_FILTER_BRIGHTNESS:
				return brightness(pixel, value);
    }
  }
  
  
  template <typename T>
  class ColorSpaceFilters
  {
    
  private:	
    /**
     * <summary>Colorspace filter type saturation</summary>
     * <parameters>sValue Range:</parameters>
     * */
    FUNCTION_PREFIX void saturation(T& pixelR, 
                                    T& pixelG,
                                    T& pixelB,
                                    T& pixelOutputR,
                                    T& pixelOutputG,
                                    T& pixelOutputB,
                                    float sValue);
    
    
    /**
     * <summary>Colorspace filter type normalize, POINT RANGES:0-255.</summary>
     * */
    FUNCTION_PREFIX void NormalizePixel(float whitePoint,
                                        float blackPoint,
                                        float outputWhitePoint,
                                        float outputBlackPoint,
                                        T& pixel, 
                                        T& pixelOutput);
    
    /**
     * Colorspace filter type function
     * */
    FUNCTION_PREFIX void ApplyFunctionOnPixel(float *curveFunction,
                                              T& pixel,
                                              T& pixelOutput); 
    
    /**
     * Colorspace filter type bw
     * */
    FUNCTION_PREFIX void BlackNWhite(T &pixelR,
                                     T &pixelG,
                                     T &pixelB,
                                     T &pixelOutputR,
                                     T& pixelOutputG,
                                     T& pixelOutputB);
    
    /**
     * Colorspace filter type sepia
     * */
    FUNCTION_PREFIX void sepia(T &pixelR,T &pixelG,T &pixelB,
                               T &pixelOutputR,
                               T &pixelOutputG,
                               T &pixelOutputB);
    
    
	public:
		FUNCTION_PREFIX void apply(T& pixelR, T& pixelG,T& pixelB,
                               T& pixelOutputR, T& pixelOutputG,
															 T& pixelOutputB,float sValue, 
															 FilterType filterType);
    
  };
  
	template<typename T>
	FUNCTION_PREFIX void ColorSpaceFilters<T>::apply(T& pixelR,T& pixelG,
                                                   T& pixelB,
                                                   T& pixelOutputR, 
                                                   T& pixelOutputG,
                                                   T& pixelOutputB,
                                                   float sValue,
                                                   FilterType filterType)
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
    /**
     * Filter can be implemented inplace.
     * sValue can be between -1 and 1.
     * 1 means no change. 0 signifies black & white. 2 signifies max saturation.
     * */
    sValue = sValue/100;
    float temp = 0.0f;
    float bwValue=pixelR*0.33+pixelG*0.33+pixelB*0.33;
    
    //Adjust Saturation of Every Channel    
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
    /**
     *	Filter can be implemented inplace.
     *	Values for both all the input values can be between 0-255;
     *	DEFAULT VALUES:
     *	BLACK POINT:0   WHITE POINT:255   OUTPUT WHITE POINT:255 OUTPUT BLACK POINT:0
     *	Can be used to perform histogram normalization.
     * */
  } 
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::ApplyFunctionOnPixel(float *curveFunction,
                                                                  T& pixel,T& pixelOutput)
  {
    /**
     * Filter can be implemented inplace.
     * Curve function defines the the output values from 0-255 got after Spline fitting input points.
     * pixelOutput=curveFunction[pixel];
     * */
    
  }
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::BlackNWhite(T &pixelR,T &pixelG, 
                                                         T &pixelB,T &pixelOutputR,
                                                         T& pixelOutputG,T& pixelOutputB)
  {
    /**
     * Filter can be implemented inplace
     * BW filter is basically 0.6*R + 0.35*G + 0.5*B
     * */
    float value = 0.6*pixelR + 0.35*pixelG + 0.05*pixelB;
    pixelOutputR=pixelOutputG=pixelOutputB=value;
  }
  
  template<typename T>
  FUNCTION_PREFIX void ColorSpaceFilters<T>::sepia(T &pixelR, T &pixelG,T &pixelB,T &pixelOutputR,
                                                   T &pixelOutputG,T &pixelOutputB)                       
  {
    /**
     * Filter can be implemented inplace 
     * BW filter is basically 0.6*R + 0.35*G + 0.5*B
     * sepia we basically add some in RedChannel, and siginficantly less in Green Channel
     * */
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

  //Blend Filters Begin.
	template<typename T>
  class BlendFilters
  {
  private:
    void normal        (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void lighten       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void darken        (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void multiply      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void average       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void add           (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void subtract      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void difference    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void negation      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void screen        (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void exclusion     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void overlay       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void softLight     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void hardLight     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void colorDodge    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void colorBurn     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void linearLight   (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void linearDodge   (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void linearBurn    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void vividLight    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void pinLight      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
    void hardMix       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);

    float alpha;
    BlendType blendType;
  public:
    FUNCTION_PREFIX void apply(T& pixelBaseR, T& pixelBaseG,T& pixelBaseB,
                                T& pixelBlendR,T& pixelBlendG, T& pixelBlendB, 
                                T& pixelDestinationR,T& pixelDestinationG,T& pixelDestinationB,
                                float alpha,BlendType blendType);
   };

   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::normal(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Normal(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
    
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::lighten(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Lighten(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::darken(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Darken(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }

   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::multiply(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Multiply(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }

   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::average(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Average(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T> 
   FUNCTION_PREFIX void BlendFilters<T>::add(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       /*float blendValue=ChannelBlend_Add(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;*/
   }
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::subtract(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Subtract(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::difference(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Difference(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::negation(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Negation(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::screen(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Screen(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::exclusion(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Exclusion(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T> 
   FUNCTION_PREFIX void BlendFilters<T>::overlay(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_Overlay(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::softLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_SoftLight(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::hardLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_HardLight(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::colorDodge(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_ColorDodge(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::colorBurn(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_ColorBurn(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::linearDodge(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_LinearDodge(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T> 
   FUNCTION_PREFIX void BlendFilters<T>::linearBurn(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_LinearBurn(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::linearLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_LinearLight(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::vividLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_VividLight(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::pinLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_PinLight(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
   FUNCTION_PREFIX void BlendFilters<T>::hardMix(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
   {
       float blendValue=ChannelBlend_HardMix(baseLayer,blendLayer);
       float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
       PIXEL_DOMAIN_CHECK(blendValueNormalized);
       destination=(T)blendValueNormalized;
   }
   
   template<typename T>
	 FUNCTION_PREFIX void BlendFilters<T>::apply(T& pixelBaseR, 
                                                    T& pixelBaseG,
                                                    T& pixelBaseB,
                                                    T& pixelBlendR, 
                                                    T& pixelBlendG,
                                                     T& pixelBlendB, 
                                                     T& pixelDestinationR, 
                                                     T& pixelDestinationG,
                                                     T& pixelDestinationB,
                                                     float alpha,
                                                     BlendType blendType)
	 {
		 switch(blendType)
     {
        case BLEND_FILTER_NORMAL:      
             normal(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
             normal(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
             normal(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;

        case BLEND_FILTER_LIGHTEN:       
            lighten(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            lighten(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            lighten(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_DARKEN:       
            darken(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            darken(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            darken(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_MULTIPLY:     
            multiply(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            multiply(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            multiply(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_AVERAGE:      
            average(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            average(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            average(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_ADD:        
            add(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            add(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            add(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_SUBTRACT:    
            subtract(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            subtract(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            subtract(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_DIFFERENCE:   
            difference(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            difference(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            difference(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_NEGATION:     
            negation(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            negation(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            negation(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_SCREEN:       
            screen(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            screen(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            screen(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_EXCLUSION:   
            exclusion(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            exclusion(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            exclusion(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;   
            
        case BLEND_FILTER_OVERLAY:      
            overlay(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            overlay(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            overlay(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_SOFTLIGHT:   
            softLight(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            softLight(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            softLight(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;  
        
        case BLEND_FILTER_HARDLIGHT:    
            hardLight(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            hardLight(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            hardLight(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_COLORDODGE:    
            colorDodge(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            colorDodge(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            colorDodge(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_COLORBURN:     
            colorBurn(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            colorBurn(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            colorBurn(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_LINEARDODGE:   
            normal(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            normal(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            normal(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_LINEARBURN:    
            linearBurn(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            linearBurn(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            linearBurn(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_LINEARLIGHT: 
            linearLight(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            linearLight(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            linearLight(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_VIVIDLIGHT:   
            
            vividLight(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            vividLight(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            vividLight(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_PINLIGHT:     
            pinLight(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            pinLight(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            pinLight(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
        case BLEND_FILTER_HARDMIX:     
            hardMix(pixelBaseR,pixelBlendR,pixelDestinationR,alpha);
            hardMix(pixelBaseG,pixelBlendG,pixelDestinationG,alpha);
            hardMix(pixelBaseB,pixelBlendB,pixelDestinationB,alpha);return;
            
      }
	}


    template<typename T>
    class ConvolutionFilters
    {
    public:
        FUNCTION_PREFIX void applyConvolution(T* inputBuffer, 
                                              T* outputBuffer,
                                              int* kernel,
                                              int imageWidth,
                                              int imageHeight,
                                              int kernelSize,
                                              int normal,
                                              int offset,
                                              int channel);
    };

    template<typename T>
    FUNCTION_PREFIX void ConvolutionFilters<T>::applyConvolution(T* inputBuffer, 
                                                                 T* outputBuffer,
                                                                 int* kernel,
                                                                 int imageWidth,
                                                                 int imageHeight,
                                                                 int kernelSize,
                                                                 int normal,
                                                                 int offset,
                                                                 int channel)
    {
      int startingOffset=offset-(imageWidth*(kernelSize/2))-kernelSize/2;
        int boundaryBase  =channel*imageWidth*imageHeight;
        int boundaryLimit =channel*imageWidth*imageHeight+imageWidth*imageHeight;
        
        int sum=0;
        int count=0;
        int _offset=startingOffset-1;
        int calc=1;
        for (int c=1;c<=kernelSize*kernelSize;c++)
        {
            if(calc==kernelSize) {_offset+=(imageWidth-kernelSize); calc=1;}
            else _offset+=1;
            if (_offset>boundaryBase && _offset<boundaryLimit)
            {
             sum+=inputBuffer[_offset]*kernel[c-1];
             count+=kernel[c-1];
            }
            calc+=1;
        }
        if (count==0)count=1;
        sum=sum/count;
        PIXEL_DOMAIN_CHECK(sum);
        outputBuffer[offset]=sum;
    
    }
}

#endif //gpu_FilterTemplates_h
