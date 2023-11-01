import os

import cv2
import numpy as np
import multiprocess as mp
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


def process_image(img_paths, channel, num_slices):
    resolution_level = 2
    train_stack = []
    for img_idx, img_path in enumerate(img_paths):
        print(f"Loading {img_path}: {img_idx} out of {len(img_paths)}")
        img = ims(img_path, ResolutionLevelLock=resolution_level)
        img_depth = img.shape[2]
        img_width = img.shape[3]
        img_height = img.shape[4]
        z_idx = [int(x) for x in np.linspace(0, img_depth - 1, num_slices)]
        for slice_count, z in enumerate(z_idx):
            print(f"Loading slice {z}. {slice_count} out of {num_slices}")
            train_stack.append(img[0, channel, z, :, :])
    train_stack = np.dstack(train_stack)
    train_stack = np.moveaxis(train_stack, -1, 0).copy(order="C")
    print(f"Running BaSic on channel {channel}")
    flatfield_ch, _ = sc.BaSiC(train_stack, estimate_darkfield=False, working_size=img_width)
    print(f"Finished estimating flatfield on {channel}")
    flatfield_ch = cv2.resize(
        flatfield_ch,
        dsize=(img_width << resolution_level, img_height << resolution_level),
        interpolation=cv2.INTER_CUBIC,
    )
    return flatfield_ch


def imaris_shading_correction(
    img_folder: str,
    result_folder: str = None,
    train_channels: list = [0, 1],
    correct_channels: list = [0, 1],
    ref_channel: int = 0,
    num_slices: int = 40,
):
    if not result_folder:
        result_folder = os.path.join(img_folder, "shading_corrected")
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

    (_, _, file_list) = next(os.walk(img_folder))
    img_list = [fn for fn in file_list if ".ims" in fn]
    num_imgs = len(img_list)

    img = ims(os.path.join(img_folder, img_list[0]))
    img_channels = img.Channels

    with mp.Pool(processes=len(train_channels)) as pool:
        flatfield = pool.starmap(
            process_image,
            [
                ([os.path.join(img_folder, x) for x in img_list], ch, num_slices)
                for ch in train_channels
            ],
        )
    flatfield_dict = {ch: flatfield_ch for (ch, flatfield_ch) in zip(train_channels, flatfield)}
    
    print(f"Applying shading Correction")
    for img_idx in range(num_imgs):
        img = ims(os.path.join(img_folder, img_list[img_idx]), ResolutionLevelLock=0)
        img_depth = img.shape[2]
        img_width = img.shape[3]
        img_height = img.shape[4]
        corrected_stack = np.zeros(
            shape=(1, img_channels, img_depth, img_width, img_height), dtype="uint16"
        )

        for ch in range(img_channels):
            if ch not in correct_channels:
                corrected_stack[0, ch, :, :, :] = img[0, ch, :, :, :]
                continue

            if ch in train_channels:
                flatfield_ch = flatfield_dict[ch]
            else:
                flatfield_ch = flatfield_dict[ref_channel]

            for i, z in enumerate(range(img_depth)):
                print(f"Correcting channel {ch} in {img_list[img_idx]} ({i} / {img_depth})")
                corrected_stack[0, ch, z, :, :] = np.clip(
                    (img[0, ch, z, :, :].astype(np.float64)) / flatfield_ch, 0, None
                ).astype("uint16")

        imaris.ims_from_ims(
            corrected_stack,
            os.path.join(img_folder, img_list[img_idx]),
            os.path.join(result_folder, img_list[img_idx]),
        )
