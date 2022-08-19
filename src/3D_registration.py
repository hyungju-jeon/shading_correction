import os 
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import ants

from zimg import *
from utils import img_util
from utils import brain_info

if __name__ == "__main__":
    
    os.chdir('/Users/yoonkyoung/Desktop/041715_JK359-2_L/')


    DAPI_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/test_sample_0308_2.tiff')
    DAPI_volume = DAPI_volume_ZImg.data[0]
    DAPI_volume = np.squeeze(DAPI_volume)
    DAPI_ants = ants.from_numpy(DAPI_volume.astype('uint32'))
    DAPI_ants.set_spacing( (50., 25., 25. ))
    # DAPI_ants = ants.resample_image(DAPI_ants, (100, 25, 25), False)
    
    # template_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/test_template_0308_2.tiff')
    # template_volume = template_volume_ZImg.data[0]
    # template_volume = np.squeeze(template_volume)
    # template_ants = ants.from_numpy(template_volume.astype('uint32'))
    # template_ants.set_spacing( (50., 25., 25. ))
    # template_ants = ants.resample_image(template_ants, (100, 25, 25), False)
        
    template_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/ARA_rotated_25.mhd')
    template_volume = template_volume_ZImg.data[0]
    template_volume = np.squeeze(template_volume)
    template_ants = ants.from_numpy(template_volume.astype('uint32'))
    template_ants.set_spacing( (25., 25., 25. ))
    template_ants = ants.resample_image(template_ants, (50, 25, 25), False)
    # data-to-template(ARA_rotated_25)
    
    # # affine    (template-to-data)
    # mytx_affine = ants.registration(fixed=DAPI_ants, moving=template, type_of_transform='Affine', 
    #                               restrict_deformation='0.0001x1x0.0001x1x1x0.0001x0.0001x0.0001x1x1x1x1', 
    #                               grad_step=0.25, write_composite_transform=False, verbose=True)
    # template2 = mytx_affine['warpedmovout']
    
    # SyNOnly 
    # mytx_syn = ants.registration(fixed=DAPI_ants, moving=template2, restrict_deformation='0x1x1', 
    #                              type_of_transform='SyNOnly', initial_transform = 'identity',
    #                              grad_step=0.25, write_composite_transform=False, verbose=True)
    
 
    # affine    (data-to-template)
    mytx_affine = ants.registration(fixed=template_ants, moving=DAPI_ants, type_of_transform='Affine', 
                                  restrict_deformation='1x1x0.00001x1x1x0.00001x0.00001x0.00001x1x1x1x1', 
                                  grad_step=0.25, write_composite_transform=False, verbose=True)
    DAPI_ants2 = mytx_affine['warpedmovout']
 
    
    # SyNOnly
    mytx_syn = ants.registration(fixed=template_ants, moving=DAPI_ants2, restrict_deformation='0x1x1', 
                                 type_of_transform='SyNOnly', initial_transform = 'identity',
                                 grad_step=0.25, write_composite_transform=False, verbose=True)
    
    warped_moving = mytx_syn['warpedmovout']

    trans_template = warped_moving.numpy().astype('uint16')
        
    img_util.write_img(filename= '/Users/yoonkyoung/Desktop/041715_JK359-2_L/registered_sample_0309.tiff', 
                       img_data = trans_template)
    
    
    # #SyNAggro
    # mytx = ants.registration(fixed=DAPI_ants, moving=template_ants, type_of_transform='SyNAggro')
    
    # warped_moving = mytx['warpedmovout']

    # trans_template = warped_moving.numpy().astype('uint16')
        
    # img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/synaggro.tiff', 
    #                     img_data = trans_template)
    
    
    # #SyNAbs
    # mytx = ants.registration(fixed=DAPI_ants, moving=template_ants, type_of_transform='antsRegistrationSyN[b]')
    
    # warped_moving = mytx['warpedmovout']

    # trans_template = warped_moving.numpy().astype('uint16')
        
    # img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/cutting_template_antsSyN.tiff', 
    #                    img_data = trans_template)