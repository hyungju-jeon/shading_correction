import os

import cv2
import numpy as np
from imaris_ims_file_reader.ims import ims

from utils import shading_util as sc
from utils.imarispy import imaris


def stack_shading_correction(img_folder: str, result_folder: str = None):
    if not result_folder:
        result_folder = os.path.join(img_folder, "shading_corrected")
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # Load image tiles
    (_, _, file_list) = next(os.walk(img_folder))
    img_list = [fn for fn in file_list if ".lsm" in fn and "TileSelection" not in fn]
    nimgs = len(img_list)
    img_info = ZImg.readImgInfos(os.path.join(img_folder, img_list[0]))

    #
    flatfield = []
    darkfield = []
    train_stack = [[] for _ in range(img_info[0].numChannels)]
    for img_idx in range(nimgs):
        print(f"Loading {img_list[img_idx]} : {img_idx} out of {nimgs}")
        img_zimg = ZImg(os.path.join(img_folder, img_list[img_idx]), xRatio=2, yRatio=2)
        img = img_zimg.data[0].copy()

        img_chs = img.shape[0]
        img_depth = img.shape[1]
        img_width = img.shape[2]
        img_height = img.shape[3]

        for z_idx in range(0, img_depth, 10):
            for ch in range(img_info[0].numChannels):
                train_stack[ch].append(img[ch, z_idx, :, :])

    for ch in range(img_info[0].numChannels):
        train_stack_ch = np.dstack(train_stack[ch])
        train_stack_ch = np.moveaxis(train_stack_ch, -1, 0).copy(order="C")

        print(f"Estimating shading parameter for channel {ch}")
        flatfield_ch, darkfield_ch = sc.BaSiC(
            train_stack_ch, estimate_darkfield=False, working_size=256
        )
        flatfield_ch = cv2.resize(flatfield_ch, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
        # darkfield_ch = cv2.resize(darkfield_ch, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        flatfield.append(flatfield_ch)
        # darkfield.append(darkfield_ch)

    for img_idx in range(nimgs):
        print(f"Loading {img_list[img_idx]} : {img_idx} out of {nimgs}")
        img_zimg = ZImg(os.path.join(img_folder, img_list[img_idx]))
        img = img_zimg.data[0].copy()

        img_chs = img.shape[0]
        img_depth = img.shape[1]
        img_width = img.shape[2]
        img_height = img.shape[3]
        print(f"Correcting {img_list[img_idx]} : {img_idx} out of {nimgs}")
        for ch in range(img_chs):
            corrected_stack = np.zeros(
                shape=(img_chs, img_depth, img_width, img_height), dtype="uint8"
            )

            for z_idx in range(img_depth):
                corrected_block = (img[ch, z_idx, :, :].astype(np.float64)) / flatfield[ch]
                # corrected_block = ((img[ch, z_idx, :, :].astype(np.float64)-darkfield[ch]) / flatfield[ch])
                corrected_block[corrected_block < 0] = 0
                corrected_block[corrected_block >= 255] = 255
                img_zimg.data[0][ch, z_idx, :, :] = corrected_block.astype("uint8")

        save_folder = result_folder
        img_zimg.save(os.path.join(save_folder, f"{img_list[img_idx][:-4]}.tif"))


def imaris_shading_correction(
    img_folder: str, result_folder: str = None, channels: list = [0, 1], num_slices: int = 40
):
    (_, _, file_list) = next(os.walk(img_folder))
    img_list = [fn for fn in file_list if ".ims" in fn]
    num_imgs = len(img_list)

    flatfield = []
    darkfield = []
    for ch in channels:
        train_stack = []
        for img_idx in range(num_imgs):
            print(f"Loading {img_list[img_idx]} : {img_idx} out of {num_imgs}")
            img = ims(os.path.join(img_folder, img_list[img_idx]), ResolutionLevelLock=2)

            img_chs = img.Channels
            img_depth = img.shape[2]
            img_width = img.shape[3]
            img_height = img.shape[4]

            for z_idx in range(0, img_depth, num_slices):
                train_stack.append(img[0, 0, z_idx, :, :])

        train_stack = np.dstack(train_stack)
        train_stack = np.moveaxis(train_stack, -1, 0).copy(order="C")

        print(f"Estimating shading parameter for channel {ch}")
        flatfield_ch, _ = sc.BaSiC(train_stack, estimate_darkfield=False, working_size=512)
        flatfield_ch = cv2.resize(
            flatfield_ch, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC
        )
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
        for ch in channels:
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
