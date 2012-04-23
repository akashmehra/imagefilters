/////////////////////////////////////////////////////////////
////BLEND FILTERS ARE PARTICULARLY USEFUL IN COMBINING/////// 
//////TWO IMAGES TOGETHER. PHOTOSHOP, GIMP IMPLEMENT/////////
/////////EXACTLY THE SAME WAY AS IMPLEMENTED BELOW///////////
/////////////////////////////////////////////////////////////

/***THESE BLEND MODES WORK ON PREDEFINED BLEND FUNCTIONS ***/
/////////////////////////////////////////////////////////////
/***    REFERENCE::http://inlandstudios.com/en/?p=851    ***/
/////////////////////////////////////////////////////////////


#define ChannelBlend_Normal(B,L)     (B)
#define ChannelBlend_Lighten(B,L)    (((L > B) ? L:B))
#define ChannelBlend_Darken(B,L)     (((L > B) ? B:L))
#define ChannelBlend_Multiply(B,L)   (((B * L) / 255))
#define ChannelBlend_Average(B,L)    (((B + L) / 2))
#define ChannelBlend_Add(B,L)        ((min(255, (B + L))))
#define ChannelBlend_Subtract(B,L)   (((B + L < 255) ? 0:(B + L - 255)))
#define ChannelBlend_Difference(B,L) ((abs(B - L)))
#define ChannelBlend_Negation(B,L)   ((255 - abs(255 - B - L)))
#define ChannelBlend_Screen(B,L)     ((255 - (((255 - B) * (255 - L)) >> 8)))
#define ChannelBlend_Exclusion(B,L)  ((B + L - 2 * B * L / 255))
#define ChannelBlend_Overlay(B,L)    (((L < 128) ? (2 * B * L / 255):(255 - 2 * (255 - B) * (255 - L) / 255)))
#define ChannelBlend_SoftLight(B,L)  (((L < 128)?(2*((B>>1)+64))*((float)L/255):(255-(2*(255-((B>>1)+64))*(float)(255-L)/255))))
#define ChannelBlend_HardLight(B,L)  (ChannelBlend_Overlay(L,B))
#define ChannelBlend_ColorDodge(B,L) (((L == 255) ? L:min(255, ((B << 8 ) / (255 - L)))))
#define ChannelBlend_ColorBurn(B,L)  (((L == 0) ? L:max(0, (255 - ((255 - B) << 8 ) / L))))
#define ChannelBlend_LinearDodge(B,L)(ChannelBlend_Add(B,L))
#define ChannelBlend_LinearBurn(B,L) (ChannelBlend_Subtract(B,L))
#define ChannelBlend_LinearLight(B,L)((L < 128)?ChannelBlend_LinearBurn(B,(2 * L)):ChannelBlend_LinearDodge(B,(2 * (L - 128))))
#define ChannelBlend_VividLight(B,L) ((L < 128)?ChannelBlend_ColorBurn(B,(2 * L)):ChannelBlend_ColorDodge(B,(2 * (L - 128))))
#define ChannelBlend_PinLight(B,L)   ((L < 128)?ChannelBlend_Darken(B,(2 * L)):ChannelBlend_Lighten(B,(2 * (L - 128))))
#define ChannelBlend_HardMix(B,L)    (((ChannelBlend_VividLight(B,L) < 128) ? 0:255))



#ifndef gpu_FilterTemplates_h
#define gpu_FilterTemplates_h 

#include "Constants.h" 
#ifdef GCC_COMPILATION
	#define FUNCTION_PREFIX 
#elif 
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
    
   template<typename T>
    class BlendFilters
    {
    private:
        T normal        (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T lighten       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T darken        (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T multiply      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T average       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T add           (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T subtract      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T difference    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T negation      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T screen        (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T exclusion     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T overlay       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T softLight     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T hardLight     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T colorDodge    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T colorBurn     (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T linearLight   (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T linearDodge   (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T linearBurn    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T vividLight    (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T pinLight      (const T& baseLayer,const T& blendLayer, T& destination, float alpha);
        T hardMix       (const T& baseLayer,const T& blendLayer, T& destination, float alpha);

        float alpha;
        BlendType blendType;
    public:
        T apply (const T& baseLayer,const T& blendLayer,T& destination, float alpha, BlendType blendType);
    };

    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::normal(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Normal(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::lighten(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Lighten(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::darken(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Darken(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }

    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::multiply(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Multiply(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }

    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::average(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Average(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T> 
    FUNCTION_PREFIX T BlendFilters<T>::add(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Add(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::subtract(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Subtract(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::difference(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Difference(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::negation(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Negation(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::screen(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Screen(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::exclusion(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Exclusion(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T> 
    FUNCTION_PREFIX T BlendFilters<T>::overlay(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_Overlay(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::softLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_SoftLight(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::hardLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_HardLight(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::colorDodge(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_ColorDodge(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::colorBurn(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_ColorBurn(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::linearDodge(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_LinearDodge(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T> 
    FUNCTION_PREFIX T BlendFilters<T>::linearBurn(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_LinearBurn(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::linearLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_LinearLight(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::vividLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_VividLight(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::pinLight(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_PinLight(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
    FUNCTION_PREFIX T BlendFilters<T>::hardMix(const T& baseLayer,const T& blendLayer, T& destination, float alpha)
    {
        float blendValue=ChannelBlend_HardMix(baseLayer,blendLayer);
        float blendValueNormalized=(1.0-alpha)*baseLayer+alpha*blendValue;
        PIXEL_DOMAIN_CHECK(blendValue);
        destination=(T)blendValue;
    }
    
    template<typename T>
	FUNCTION_PREFIX void ColorSpaceFilters<T>::apply(T& pixelBaseR, 
                                                     T& pixelBaseG,
                                                     T& pixelBaseB,
                                                     T& pixelBlendR, 
                                                     T& pixelBlendG,
                                                     T& pixelBlendB, 
                                                     T& pixelDestinationR, 
                                                     T& pixelDestinationG,
                                                     T& pixelDestinationB,
                                                     float alpha,
                                                     BlendFilters blendType)
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
    
}

