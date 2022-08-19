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

    slice_list = []
    
    img_height = 0
    img_width = 0
    scale_ratio = 16

    info_file = '/users/Yoonkyoung/Desktop/041715_JK359-2_L/info.txt'
    brain_info = brain_info.read_brain_info(info_file)
    
    filename = brain_info['filename']
    scene = brain_info['scene']
    
    for czi_name in filename:
        czi_img_info = ZImg.readImgInfos(czi_name)
        
        for scene_idx in range(len(czi_img_info)):
            imgObj = ZImg(czi_name, scene = scene_idx, xRatio = 16, yRatio = 16)
            img = imgObj.data[0]    
              

            if czi_img_info[scene_idx].height>img_height:
                
                img_height = max(czi_img_info[scene_idx].height, img_height)
                img_width = max(czi_img_info[scene_idx].width, img_width)
                
                img_height = int(np.ceil(img_height / scale_ratio))
                img_width = int(np.ceil(img_width / scale_ratio))
                    
    print(img_height, img_width)
    



