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

from zimg import *
from utils import img_util
from utils import brain_info


def rotate_points(pts:np.ndarray, center_pts:np.ndarray, rot_angle:int):
    matc = np.array([[1,0,center_pts[0]],[0,1,center_pts[1]],[0,0,1]])
    matr = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0, 0, 1]])
    inv_matc = np.linalg.inv(matc)

    tform = np.dot(np.dot(matc, matr), inv_matc)
    aug_pts = pts.copy()
    aug_pts.append(1)

    res_pts = np.dot(tform, np.transpose(aug_pts))

    return res_pts[:-1]


if __name__ == "__main__":
    
    os.chdir('/users/Yoonkyoung/Desktop/041715_JK359-2_L/')
    
    
    info_file = '/users/Yoonkyoung/Desktop/041715_JK359-2_L/info.txt'
    info_file_midline = '/users/Yoonkyoung/Desktop/041715_JK359-2_L/info_midline.txt'
    brain_infofile = brain_info.read_midline_info(info_file_midline, ratio = 16)  #ratio = 32
    brain_infotxt = brain_info.read_brain_info(info_file)
    
    filename = brain_infofile['filename']
    scenes = brain_infofile['scene']
    pts = brain_infofile['pts']    
    flipped = brain_infotxt['flipped']
        
    img_height = 0
    img_width = 0
    scale_ratio = 16   #ratio = 32
    
    rotated_list = []
    rotated_pad_list = []
    
    
    for idx in range(len(filename)): 
        display(f'Running {idx} slice')
        
        czi_name = filename[idx]
        scene_idx = scenes[idx]
        flip = int(flipped[idx])
        
        czi_img_info = ZImg.readImgInfos(czi_name)
        
        imgObj = ZImg(czi_name, scene = int(scene_idx) - 1, xRatio = 16, yRatio = 16)  #ratio = 32
        img = imgObj.data[0]   
        
        DAPI_img = img[1, :, :, :]
        DAPI_img = np.squeeze(DAPI_img)
        
        # load midpts
        x1 = pts[idx][0]
        x2 = pts[idx][1]
        y1 = pts[idx][2]
        y2 = pts[idx][3]
            
        pts1 = [x1, y1]
        pts2 = [x2, y2]
        
        center_pts = [img.shape[3]/2, img.shape[2]/2]

        theta = 0 
        degree = 0

        # compute angle         
        slope = (x1 - x2)/(y1 - y2)
        theta = math.atan(slope)
        degree = -math.degrees(theta)
        
        #flip image
        if flip == 1:
            DAPI_img = np.fliplr(DAPI_img)
        
        # rotate image    
        rotated = imutils.rotate(DAPI_img, degree)
        
        # rotate midline pts
        new_pts = rotate_points(pts1, center_pts, theta)
        new_pts2 = rotate_points(pts2, center_pts, theta)
            
        # compute difference between transformed midline points and center of image
        trans_pts = (new_pts[0] - center_pts[0])
    
        # shift image 
        rows,cols = rotated.shape
        M = np.float32([[1, 0, -trans_pts], [0, 1, 0]])
        new_img = cv2.warpAffine(rotated, M, (cols,rows))
        
        img_height = new_img.shape[0]
        img_width = new_img.shape[1]
        
        rotated_list.append(new_img)
    
    print('calculating max height, width')
    #get max height, width
    for new_img in rotated_list:
        
        if new_img.shape[0]>img_height:
                
             img_height = max(new_img.shape[0], img_height)
             img_width = max(new_img.shape[1], img_width)
                
             img_height = int(np.ceil(img_height))
             img_width = int(np.ceil(img_width ))
             print(img_height, img_width)
             
    print('max:', img_height, img_width)
    
    
    #pad image with max height, width
    print('starting image padding')

    for new_img in rotated_list:
                    
        new_img = img_util.pad_img(new_img, des_height = img_height+50, des_width = img_width+50)
        rotated_pad_list.append(new_img)
        
    rotated_volume = np.stack(rotated_pad_list, axis=0)    
    print(rotated_volume.shape)
    
    img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_midline_stack_whole.tiff', 
                       img_data = rotated_volume)
   
    
    #y-direction translation
    print('starting y-translation registration')

    stack_volume_ZImg = ZImg('/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_midline_stack_whole.tiff')
    stack_volume = stack_volume_ZImg.data[0]
    DAPI_volume = np.squeeze(stack_volume)
    
    for idx in range(len(rotated_volume)-1):
        display(f'Running {idx} slice')
        
        fixedImg = ants.from_numpy(DAPI_volume[idx,:,:].astype('float32'))
        movingImg = ants.from_numpy(DAPI_volume[idx+1,:,:].astype('float32'))
        
        mytx = ants.registration(fixed=fixedImg , moving=movingImg, type_of_transform='Translation', 
                                 initial_transform = 'identity', restrict_deformation= '1x0')
        warped_moving = mytx['warpedmovout']
        
        DAPI_volume[idx+1,:,:] = warped_moving.numpy().astype('uint16')
        
        
        # # apply transform to other channel
        # for ch in range(stack_volume.shape[0]):
        #     if ch == 1:
        #         continue
        #     fixedImg = ants.from_numpy(stack_volume[ch,idx,:,:].astype('float32'))
        #     movingImg = ants.from_numpy(stack_volume[ch,idx+1,:,:].astype('float32'))
          
        #     trans = ants.apply_transforms(fixed=fixedImg, moving=movingImg, transformlist=mytx['fwdtransforms']) 
        
        #     # update stack_volume
        #     stack_volume[ch,idx+1,:,:] = trans.numpy().astype('uint16')

    img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/test_3D_midline.tiff', 
                       img_data = DAPI_volume)
    print('done')
print('done')    
        
     
       
        
    
    
    
