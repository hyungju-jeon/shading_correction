import os 
import sys
sys.path.append('../')

import pandas as pd
import math
import numpy as np
import ants

from zimg import *
from utils import img_util
from utils import brain_info

if __name__ == "__main__":

    # load allen
    allen_volume_ZImg = ZImg('/users/Yoonkyoung/Desktop/average_template/average_template_10_coronal.tif')
    allen_volume = allen_volume_ZImg.data[0]
    allen_volume = np.squeeze(allen_volume)
    allen_ants = ants.from_numpy(allen_volume.astype('float32'))
    
    #load 3d
    DAPI_volume_ZImg = ZImg('/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_Align_DAPI.tiff')
    DAPI_volume = DAPI_volume_ZImg.data[0]
    DAPI_volume = np.squeeze(DAPI_volume)
    DAPI_ants = ants.from_numpy(DAPI_volume.astype('float32'))
    
    # set spacing
    allen_ants.set_spacing( (10., 10., 10. ) )
    DAPI_ants.set_spacing( (50., 5., 5.) )
    
    # downsample
    # allen_ants = ants.resample_image(allen_ants, (50, 10, 10), False)
    DAPI_ants = ants.resample_image(DAPI_ants, (50, 10, 10), False)
    
    # make new volume: DAPI_reg_volume = np.zeros[dapi_slice#, allen_x, allen_y,]
    DAPI_reg_volume = np.zeros((234, 800, 1140))

    # first registration strategy
    for idx in range(234):
        display(f'Running {idx} slice')
        
        fixedImg = ants.from_numpy(allen_ants[5*idx+225, :, :].astype('float32'))
        movingImg = ants.from_numpy(DAPI_ants[idx, :, :].astype('float32'))
        
        # mytx = ants.registration(fixed=fixedImg , moving=movingImg, type_of_transform='Translation' )
        mytx = ants.registration(fixed=fixedImg , moving=movingImg, type_of_transform='Translation', restrict_deformation, '1x0')
        warped_moving = mytx['warpedmovout']
        
        
        DAPI_reg_volume[idx, :, :] = warped_moving.numpy().astype('uint16')
        
        
    img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_DAPI_Allen_1.tiff', img_data = DAPI_reg_volume)
    
    
    
    
    
    #%% Load allen reference 3D volume
    """
    1. If the allen reference volume is not in coronal, moveaxis to coronal
    2. Convert the volume into 3D ants image
    """
    
    #%% Load our 3D aligned volume
    """
    1.  Change the voxel spacing for allen reference 3D volume and 3D aligned volume this can be done using set_spacing function in ants.
        Allens reference brain is isometric 10x10x10
        Our data is (original voxel resolution * downsample ratio) x (original voxel resolution * downsample ratio) x 50
    2.  3D registration often takes quite long, and we don't want to waste too much time while testing/practicing
        Downsample the both volume image using resample_image functionn in ants
    """
    #%% First registration strategy
    """
    1.  In this strategy, we will assume that the cutting angle and the size of brain are similar
    2.  This enables us to find one-to-one correspondence between a slice in reference volume and in our aligned volume
    3.  Open two volume using ATLAS, and you will notice that the starting region in two volumes are different (allen will probably contain more regions)
    4.  Find the a slice on reference volume that corresponds (most similar) to the first slice of aligned volume
    5.  Then you can perform 2D registration for each subsequent slices 
    """
    #%% Second registration strategy
    """
    1.  In this strategy, we will estimate cutting angle difference and volume difference
    2.  First perform 3D registration between two volume with reference volume as moving image
        In this step, we will use Similarity registration. 
    3.  Once the 3D registration is done, perform 2D slice registration with transformed reference volume as fixed_image    
        In this step, we will use Rigid registration
    4.  Repeat step 2-3 several times 
    """