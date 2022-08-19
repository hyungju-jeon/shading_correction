import os 
import sys
sys.path.append('../')

import pandas as pd
import math
import numpy as np
import ants
import cv2
import imutils
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
    # DAPI_ants = ants.resample_image(DAPI_ants, (50, 25, 25), False)
        
        
    template_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/ARA_rotated_25.mhd')
    template_volume = template_volume_ZImg.data[0]
    template_volume = np.squeeze(template_volume)
    template_ants = ants.from_numpy(template_volume.astype('uint32'))
    template_ants.set_spacing( (25., 25., 25. ))
    
    cutting_angle = []
    average_angle = []
    parameters = []
    
    DAPI_volume = DAPI_ants.numpy().astype('uint16')
   
    slices = template_volume.shape[0]
    resolution = template_ants.spacing[0]
    height = DAPI_volume.shape[1]
    width = DAPI_volume.shape[2] 
   
    for idx in range(21):
   
        slice_idx = 106 + idx
        print(f'{slice_idx}')
        corresponding_slice_idx = 212 + 56 + 2*idx   #40-50 (slice difference)
        print(f'{corresponding_slice_idx}')
        
        fixed_volume = np.zeros((slices, height, width))
        fixed_volume[corresponding_slice_idx,:,:] = DAPI_volume[slice_idx,:,:] 
        fixed_volume[corresponding_slice_idx+1,:,:] = DAPI_volume[slice_idx,:,:] 
        fixed_ants = ants.from_numpy(fixed_volume.astype('uint32'))
        fixed_ants.set_spacing((25., 25., 25.))
        
        fixed_mask = np.zeros((slices, height, width), dtype=np.bool)
        fixed_mask[corresponding_slice_idx:corresponding_slice_idx+2,:,:] = True
        fixed_mask = ants.from_numpy(fixed_mask.astype('uint32'))
        fixed_mask.set_spacing((25., 25., 25.))
        
        
        # mytx = ants.registration(fixed=fixed_ants, moving=template_ants, type_of_transform='Similarity', 
        #                           restrict_deformation='1x1x0.00001x1x1x0.00001x0.00001x0.00001x1x1x1x1', 
        #                           aff_sampling=64, aff_random_sampling_rate=0.75,
        #                           grad_step=0.25, write_composite_transform=False, verbose=True)
        
        
        mytx = ants.registration(fixed=fixed_ants, moving=template_ants, type_of_transform='Affine', 
                                  restrict_deformation='1x1x0.00001x1x1x0.00001x0.00001x0.00001x1x1x1x1', 
                                  aff_sampling=64, aff_random_sampling_rate=0.75, mask=fixed_mask,
                                  grad_step=0.25, write_composite_transform=False, verbose=True)
        
        a = mytx['fwdtransforms']
        a = ' '.join(map(str, a))
        
        tx2 = ants.read_transform(a)
        parameters.append(tx2.parameters)
        
        rot_mat = (np.reshape(tx2.parameters[0:9], (3,3)))
        r = R.from_matrix(rot_mat)
        
        angle = r.as_euler('zyx', degrees=True)
        cutting_angle.append(angle)
        
        display(angle)
        
        print(cutting_angle)
    
        parameter_avg = np.mean(parameters, axis = 0)
        avg = np.mean(cutting_angle, axis=0)
        average_angle.append(avg)
    print(average_angle)
    

    new_parameter = parameter_avg.copy()

    new_parameter[9] +=  56*25      # 40-70
    # new_parameter[11] -=10*25
    
    new_tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3, parameters=new_parameter)
    ants.write_transform(new_tx, 'cutting_angle.mat')
    trans = ants.apply_transforms(fixed=DAPI_ants, moving=template_ants, transformlist='cutting_angle.mat')
    trans = ants.resample_image(trans, (50, 25, 25), False)

    trans_template = trans.numpy().astype('uint16')
    img_util.write_img(filename= '/Users/yoonkyoung/Desktop/041715_JK359-2_L/test_template_0308_2.tiff', img_data = trans_template)
    
    
    # new_tx = ants.new_ants_transform(precision='float', dimension=3, transform_type='AffineTransform', parameters=parameter_avg)
    # new_tx.apply_to_image(template_ants,reference=DAPI_ants)
    # template_ants = ants.resample_image(template_ants, (50, 25, 25), False)
    
    # trans_template = template_ants.numpy().astype('uint16')
    # img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/trans_template.tiff', img_data = trans_template)
    
