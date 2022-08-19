import os
import sys
sys.path.append('../')

from utils import brain_info
from utils import img_util
from zimg import *
import numpy as np
import ants

if __name__ == "__main__":

    os.chdir('/users/Yoonkyoung/Desktop/041715_JK359-2_L/')

    stack_volume_ZImg = ZImg('/users/Yoonkyoung/Desktop/041715_JK359-2_L/041715_JK359-2_L_3d.tiff')
    stack_volume = stack_volume_ZImg.data[0]
    DAPI_volume = stack_volume[1,:,:,:]
  
    
    for idx in range(233):
        
        display(f'Running {idx} slice')
        fixedImg = ants.from_numpy(DAPI_volume[idx,:,:].astype('float32'))
        movingImg = ants.from_numpy(DAPI_volume[idx+1,:,:].astype('float32'))
        
        mytx = ants.registration(fixed=fixedImg , moving=movingImg, type_of_transform='Rigid',restrict_deformation, '1x1x0' )
        warped_moving = mytx['warpedmovout']
        
        DAPI_volume[idx+1,:,:] = warped_moving.numpy().astype('uint16')
        
        
        # apply transform to other channel
        for ch in range(stack_volume.shape[0]):
            if ch == 1:
                continue
            fixedImg = ants.from_numpy(stack_volume[ch,idx,:,:].astype('float32'))
            movingImg = ants.from_numpy(stack_volume[ch,idx+1,:,:].astype('float32'))
          
            trans = ants.apply_transforms(fixed=fixedImg, moving=movingImg, transformlist=mytx['fwdtransforms']) 
        
            # update stack_volume
            stack_volume[ch,idx+1,:,:] = trans.numpy().astype('uint16')

    img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_DAPI.tiff', img_data = stack_volume)
    

