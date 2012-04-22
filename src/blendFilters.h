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
 
   enum FilterType
   {
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

}

