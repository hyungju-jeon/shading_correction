import os

import cv2
import numpy as np
from imaris_ims_file_reader.ims import ims

from utils import shading_util as sc
from utils.imarispy import imaris

if __name__ == "__main__":
    img_folder = "/Volumes/shared_3/Personal/hyungju/chang/JE_Stitching"
    result_folder = "/Volumes/T7/HJ_correction"

    (_, _, file_list) = next(os.walk(img_folder))
    img_list = [fn for fn in file_list if ".ims" in fn]
    num_imgs = len(img_list)

    flatfield = []
    darkfield = []
    for ch in range(2):
        train_stack = []
        for img_idx in range(num_imgs):
            print(f"Loading {img_list[img_idx]} : {img_idx} out of {num_imgs}")
            img = ims(os.path.join(img_folder, img_list[img_idx]), ResolutionLevelLock=2)

            img_chs = img.Channels
            img_depth = img.shape[2]
            img_width = img.shape[3]
            img_height = img.shape[4]

            for z_idx in range(0, img_depth, 40):
                train_stack.append(img[0, 0, z_idx, :, :])

        train_stack = np.dstack(train_stack)
        train_stack = np.moveaxis(train_stack, -1, 0).copy(order="C")

        print(f"Estimating shading parameter for channel {ch}")
        flatfield_ch, _ = sc.BaSiC(train_stack, estimate_darkfield=False, working_size=512)
        flatfield_ch = cv2.resize(flatfield_ch, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
        # darkfield_ch = cv2.resize(darkfield_ch, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        flatfield.append(flatfield_ch)
        # darkfield.append(darkfield_ch)

    for img_idx in range(num_imgs):
        img = ims(os.path.join(img_folder, img_list[img_idx]), ResolutionLevelLock=0)
        # Imaris fileformat is [R,C,Z,X,Y] where R is resolution

        img_chs = img.Channels
        img_depth = img.shape[2]
        img_width = img.shape[3]
        img_height = img.shape[4]
        corrected_stack = np.zeros(shape=(1, 2, img_depth, img_width, img_height), dtype="uint16")
        for ch in range(2):
            for z_idx in range(img_depth):
                corrected_block = (img[0, ch, z_idx, :, :].astype(np.float64)) / flatfield[ch]
                corrected_block[corrected_block < 0] = 0
                corrected_stack[0, ch, z_idx, :, :] = corrected_block.astype("uint16")

        save_folder = result_folder
        imaris.ims_from_ims(
            corrected_stack,
            os.path.join(img_folder, img_list[img_idx]),
            os.path.join(save_folder, img_list[img_idx]),
        )
