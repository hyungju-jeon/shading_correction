import os
import sys
sys.path.append('../')

from utils import brain_info
from utils import img_util
from zimg import *
import numpy as np
import ants

if __name__ == "__main__":

    os.chdir('/users/Yoonkyoung/Desktop/PV-zsGreen serial/')

    slice_list = []

    info_file = '/users/Yoonkyoung/Desktop/PV-zsGreen serial/info.txt'
    brain_info = brain_info.read_brain_info(info_file)
    
    filename = brain_info['filename']
    scenes = brain_info['scene']
            
        
    for idx in range(len(filename)):
        
        czi_name = filename[idx]
        scene_idx = scenes[idx]
        
        imgObj = ZImg(czi_name, scene = int(scene_idx) - 1, xRatio = 16, yRatio = 16)
        img = imgObj.data[0]
        img = img_util.pad_img(img, des_height = 987, des_width = 1332)
            
        slice_list.append(img)

                
stack_volume = np.stack(slice_list, axis=1)
stack_volume = np.squeeze(stack_volume)

print(stack_volume.shape)

img_util.write_img(filename= '/users/Yoonkyoung/Desktop/PV-zsGreen serial/stack_volume.tiff', img_data = stack_volume)

