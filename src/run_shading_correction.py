from ..utils import shading_util

def stack_shading_correction():
    img_folder = '/Volumes/shared/Personal/Jihyun/mGRASPi/CONVERGENCE/4inputstoDG/20220603_oist_fig7_set1to4/confocal/7_3_1/IV/temp_1'
    result_folder = '/Volumes/shared/Personal/Jihyun/mGRASPi/CONVERGENCE/4inputstoDG/20220603_oist_fig7_set1to4/confocal/7_3_1/IV/temp_1' \
                    '/shading_corrected'

    (_, _, file_list) = next(os.walk(img_folder))
    img_list = [fn for fn in file_list if '.lsm' in fn and 'TileSelection' not in fn]
    nimgs = len(img_list)
    img_info = ZImg.readImgInfos(os.path.join(img_folder, img_list[0]))

    # TODO : Multi-stack shading Correction
    flatfield = []
    darkfield = []
    train_stack = [[] for _ in range(img_info[0].numChannels)]
    for img_idx in range(nimgs):
        print(f'Loading {img_list[img_idx]} : {img_idx} out of {nimgs}')
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
        train_stack_ch = np.moveaxis(train_stack_ch, -1, 0).copy(order='C')

        print(f'Estimating shading parameter for channel {ch}')
        flatfield_ch, darkfield_ch = sc.BaSiC(train_stack_ch, estimate_darkfield=False, working_size=256)
        flatfield_ch = cv2.resize(flatfield_ch, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
        # darkfield_ch = cv2.resize(darkfield_ch, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        flatfield.append(flatfield_ch)
        # darkfield.append(darkfield_ch)

    for img_idx in range(nimgs):
        print(f'Loading {img_list[img_idx]} : {img_idx} out of {nimgs}')
        img_zimg = ZImg(os.path.join(img_folder, img_list[img_idx]))
        img = img_zimg.data[0].copy()

        img_chs = img.shape[0]
        img_depth = img.shape[1]
        img_width = img.shape[2]
        img_height = img.shape[3]
        print(f'Correcting {img_list[img_idx]} : {img_idx} out of {nimgs}')
        for ch in range(img_chs):
            corrected_stack = np.zeros(shape=(img_chs, img_depth, img_width, img_height), dtype='uint8')

            for z_idx in range(img_depth):
                corrected_block = ((img[ch, z_idx, :, :].astype(np.float64)) / flatfield[ch])
                # corrected_block = ((img[ch, z_idx, :, :].astype(np.float64)-darkfield[ch]) / flatfield[ch])
                corrected_block[corrected_block < 0] = 0
                corrected_block[corrected_block >= 255] = 255
                img_zimg.data[0][ch, z_idx, :, :] = corrected_block.astype('uint8')

        save_folder = result_folder
        img_zimg.save(os.path.join(save_folder, f'{img_list[img_idx][:-4]}.tif'))