import random
import numpy as np
import itertools

def downsample_slice(img_array, ds_factor, is_mask = False):
    
    '''
    
    Function Info:
        Takes in a 3D array and then downsamples in the z-direction by 
        a user-specified downsampling factor
        
    Parameters:
    -----------
    img_array : float32 array
        3D numpy array for now (xres x yres x zres)
    ds_factor : int
        Downsampling factor
    is_mask : bool
        Set to True if downsampling a mask and binarizing (unlike for image)

    Returns:
    --------
    final : float32 array
        3D numpy array for now (xres x yres x zres//ds_factor)
        
    Example:
    --------        
    input_image  = numpy.random.rand(4,4,4)
    input_mask   = (a > 0.5)*1
    output_image = downsample_slice(input_mask, ds_factor = 2, is_mask = False)
    output_mask  = downsample_slice(input_mask, ds_factor = 2, is_mask = True)

    Created By:
    -----------  
    Akshay Chaudhari, akshaysc@stanford.edu, October 16th 2018

    '''
        
    img_array = np.transpose(img_array,(2,0,1))
    L = list(img_array)
    
    def grouper(iterable, n):
         args = [iter(iterable)] * n
         return itertools.zip_longest(fillvalue=0, *args)
     
    final = np.array([sum(x) for x in grouper(L, ds_factor)])    
    final = np.transpose(final, (1,2,0))
    
#     Binarize if it is a mask
    if is_mask is True:
        final = (final >= 1)*1

    return final 

