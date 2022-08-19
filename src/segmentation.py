#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:28:42 2021

@author: yoonkyoung
"""

template_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/registered_template_0308_8.tiff')
template_volume = template_volume_ZImg.data[0]
template_volume = np.squeeze(template_volume)
template_ants = ants.from_numpy(template_volume.astype('uint32'))
template_ants.set_spacing( (50., 25., 25. ))

moving_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/ARA_anno_rotated_25.mhd')
moving_template_volume = moving_volume_ZImg.data[0]
moving_template_volume = np.squeeze(moving_template_volume)
moving_template_ants = ants.from_numpy(moving_template_volume.astype('uint32'))
moving_template_ants.set_spacing( (25., 25., 25. ))

moving_volume_ZImg = ZImg('/Users/yoonkyoung/Desktop/041715_JK359-2_L/cortex_mask.tiff')
moving_template_volume = moving_volume_ZImg.data[0]
moving_template_volume = np.squeeze(moving_template_volume)
moving_template_ants = ants.from_numpy(moving_template_volume.astype('uint32'))
moving_template_ants.set_spacing( (25., 25., 25. ))


img = ants.apply_transforms(fixed=template_ants, moving=moving_template_ants, transformlist=mytx_affine5['fwdtransforms'], 
                            interpolator='genericLabel')

trans2 = ants.apply_transforms(fixed=template_ants, moving=img, transformlist=mytx_syn5['fwdtransforms'], 
                               interpolator='genericLabel')

trans_template = trans2.numpy().astype('uint16')
img_util.write_img(filename= '/Users/yoonkyoung/Desktop/041715_JK359-2_L/annotation_template_50_7.tiff', 
                   img_data = trans_template)




# trans = ants.apply_transforms(fixed=template_ants, moving=moving_template_ants, transformlist='cutting_angle_0308.mat',
#                               interpolator='genericLabel')

# img = ants.apply_transforms(fixed=template_ants, moving=trans, transformlist=mytx_affine5['fwdtransforms'], 
#                             interpolator='genericLabel')

# trans2 = ants.apply_transforms(fixed=template_ants, moving=img, transformlist=mytx_syn5['fwdtransforms'], 
#                                interpolator='genericLabel', whichtoinvert=[False,True])

# trans_template = trans2.numpy().astype('uint16')
# img_util.write_img(filename= '/Users/yoonkyoung/Desktop/041715_JK359-2_L/annotation_template_50_6.tiff', 
#                    img_data = trans_template)