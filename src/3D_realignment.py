import os 
import sys
sys.path.append('../')

import pandas as pd
import math
import numpy as np
import ants
import cv2
# import imutils
import math 
import re
import matplotlib.pyplot as plt

from zimg import *
from utils import img_util
from utils import brain_info
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    
    os.chdir('/Users/yoonkyoung/Desktop/041715_JK359-2_L/')


    DAPI_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/prealign_vol.mhd')
    DAPI_volume = DAPI_volume_ZImg.data[0]
    DAPI_volume = np.squeeze(DAPI_volume)
    DAPI_ants = ants.from_numpy(DAPI_volume.astype('uint32'))
    DAPI_ants.set_spacing( (50., 25., 25. ))
    # DAPI_ants = ants.resample_image(DAPI_ants, (400, 25, 25), False)
    
    template_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/test_template_0308_2.tiff')
    template_volume = template_volume_ZImg.data[0]
    template_volume = np.squeeze(template_volume)
    template_ants = ants.from_numpy(template_volume.astype('uint32'))
    template_ants.set_spacing( (50., 25., 25. ))
    # template_ants = ants.resample_image(template_ants, (100, 25, 25), False)
    
    org_template_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/ARA_rotated_25.mhd')
    org_template_volume = org_template_volume_ZImg.data[0]
    org_template_volume = np.squeeze(org_template_volume)
    org_template_ants = ants.from_numpy(org_template_volume.astype('uint32'))
    org_template_ants.set_spacing( (25., 25., 25. ))

    
    for i in range(4):
        display(f'Running {i}')
        
        for idx in range(len(DAPI_volume)):
            display(f'Running {idx} slice')
            
        
            fixedImg = ants.from_numpy(template_ants[idx, :, :].astype('uint32'))
            movingImg = ants.from_numpy(DAPI_ants[idx, :, :].astype('uint32'))
            
            # translation
            mytx = ants.registration(fixed=fixedImg , moving=movingImg, initial_transform = 'identity',
                                  type_of_transform='Translation', restrict_deformation= '1x0')
        
            warped_moving = mytx['warpedmovout']
            DAPI_ants[idx, :, :] = warped_moving.numpy().astype('uint16')
            
            movingImg = ants.from_numpy(DAPI_ants[idx, :, :].astype('uint32'))
            
            # # rigid
            # mytx = ants.registration(fixed=fixedImg , moving=movingImg, initial_transform = 'identity',
            #                       type_of_transform='Rigid', restrict_deformation= '0.001x1x0.0001')
            
            # warped_moving = mytx['warpedmovout']
            # DAPI_ants[idx, :, :] = warped_moving.numpy().astype('uint16')

        # 3D volume affine registration
        mytx = ants.registration(fixed=DAPI_ants, moving=org_template_ants, type_of_transform='Affine', 
                                 restrict_deformation='0.0001x1x0.0001x1x1x0.0001x0.0001x0.0001x1x1x1x1', 
                                 grad_step=0.25, write_composite_transform=False, verbose=True)
    
        warped_moving = mytx['warpedmovout']
        template_ants = warped_moving.numpy().astype('uint16')
            
    trans_template = DAPI_ants.numpy().astype('uint16')
        
    img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_sample_0308_2.tiff', img_data = trans_template)