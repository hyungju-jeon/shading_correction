import os
import sys
import numpy as np
import scipy
import scipy.fft
import skimage.exposure
import logging
import multiprocessing
import math
import re

from zimg import *
from utils import io
from utils import img_util
from utils.brain_info import read_brain_info
from scipy import ndimage

logger = logging.getLogger(__name__)


def BaSiC_parameters():
    return {
        'lambdaa': None,  # default value estimated from input images directly, high values (eg. 9.5) increase the
        # spatial regularization strength, yielding a more smooth flatfield
        'estimation_mode': 'l0',
        'max_iterations': 500,
        'optimization_tol': 1e-6,
        'darkfield': False,  # whether you would like to estimate darkfield, keep 'false' if the input images are
        # brightfield images or bright fluoresence images, set 'true' only if only if the input images are fluorescence
        # images are dark and have a strong darkfield contribution.
        'lambda_darkfield': None,  # default value estimated from input images directly, , high values (eg. 9.5)
        # increase the spatial regularization strength, yielding a more smooth darkfield
        'working_size': 128,
        'max_reweightiterations': 10,
        'eplson': .1,  # reweighting parameter
        'varying_coeff': True,
        'reweight_tol': 1e-3,  # reweighting tolerance
    }


def inexact_alm_rspca_l1(D, *, lambdaa=None, lambda_darkfield=None, tol=1e-6,
                         maxIter=500, weight=1., estimatedarkfield=False, darkfieldlimit=1e7):
    '''
     l1 minimizatioin, background has a ratio, rank 1 but not the
     same
     This matlab code implements the inexact augmented Lagrange multiplier
     method for Sparse low rank matrix recovery

     modified from Robust PCA
     reference:
     Peng et al. "A BaSiC tool for background and shading correction
     of optical microscopy images" Nature Communications, 14836(2017)
     Candès, E., Li, X., Ma, Y. & Wright, J. "Robust Principal Component
     Analysis?" J. ACM (58) 2011

     D - n x m x m  matrix of observations/data (required input)

     while ~converged
       minimize (inexactly, update A and E only once)
       L(W, E,Y,u) = |E|_1+lambda * |W|_1 + <Y2,D-repmat(QWQ^T)-E> + +mu/2 * |D-repmat(QWQ^T)-E|_F^2;
       Y1 = Y1 + \mu * (D - repmat(QWQ^T) - E);
       \mu = \rho * \mu;
     end
    :param D: m x m x n  matrix of observations/data (required input)
    :return: A1_hat, E1_hat, A_offset
    '''
    p, q, n = D.shape
    m = p * q
    D = np.reshape(D, (m, n))
    if isinstance(weight, np.ndarray):
        weight = np.reshape(weight, D.shape)

    temp = np.linalg.svd(D, full_matrices=False, compute_uv=False, hermitian=False)
    norm_two = temp[0]
    Y1 = Y2 = 0.
    ent1 = 1.
    ent2 = 10.

    A1_hat = np.zeros(D.shape)
    E1_hat = np.zeros(D.shape)
    W_hat = scipy.fft.dctn(np.mean(np.reshape(A1_hat, (p, q, n)), axis=-1), norm='ortho')
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(D, 'fro')
    A1_coeff = np.ones(shape=(1, n))
    A_offset = np.zeros(shape=(m, 1))
    B1_uplimit = D.min()
    B1_offset = 0.
    A_uplimit = np.min(D, axis=-1)
    A_inmask = np.zeros(shape=(p, q))
    A_inmask[int(np.round(p / 6.)):int(np.round(p * 5. / 6)), int(np.round(q / 6.)):int(np.round(q * 5. / 6))] = 1
    # main iterations
    iter = 0
    total_svd = 0
    converged = False
    W_idct_hat = None
    while not converged:
        iter += 1
        W_idct_hat = scipy.fft.idctn(W_hat, norm='ortho')
        A1_hat = W_idct_hat.reshape((-1, 1)) @ A1_coeff.reshape((1, -1)) + A_offset.reshape((-1, 1))
        temp_W = (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1

        temp_W = np.mean(np.reshape(temp_W, (p, q, n)), axis=-1)
        W_hat += scipy.fft.dctn(temp_W, norm='ortho')
        W_hat = np.fmax(W_hat - lambdaa / (ent1 * mu), 0.) + np.fmin(W_hat + lambdaa / (ent1 * mu), 0.)
        W_idct_hat = scipy.fft.idctn(W_hat, norm='ortho')
        A1_hat = W_idct_hat.reshape((-1, 1)) @ A1_coeff.reshape((1, -1)) + A_offset.reshape((-1, 1))
        # update E1 using l0 norm
        E1_hat = E1_hat + (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        E1_hat = np.fmax(E1_hat - weight / (ent1 * mu), 0.) + np.fmin(E1_hat + weight / (ent1 * mu), 0.)
        # update A1_coeff, A2_coeff and A_offset
        R1 = D - E1_hat
        A1_coeff = np.mean(R1, axis=0) / np.mean(R1)
        A1_coeff[A1_coeff < 0] = 0
        if estimatedarkfield:
            validA1coeff_idx = A1_coeff < 1.
            W_idct_hat_flatten = W_idct_hat.flatten()
            B1_coeff = (np.mean(R1[W_idct_hat_flatten > np.mean(W_idct_hat_flatten) - 1e-6][:, validA1coeff_idx], axis=0) -
                        np.mean(R1[W_idct_hat_flatten < np.mean(W_idct_hat_flatten) + 1e-6][:, validA1coeff_idx], axis=0)) / \
                       np.mean(R1)
            k = validA1coeff_idx.sum()
            temp1 = np.sum(np.square(A1_coeff[validA1coeff_idx]), axis=0)
            temp2 = np.sum(A1_coeff[validA1coeff_idx], axis=0)
            temp3 = np.sum(B1_coeff, axis=0)
            temp4 = np.sum(A1_coeff[validA1coeff_idx] * B1_coeff, axis=0)
            temp5 = temp2 * temp3 - k * temp4
            if temp5 == 0:
                B1_offset = 0.
            else:
                B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
            # limit B1_offset: 0<B1_offset<B1_uplimit
            B1_offset = np.fmax(B1_offset, 0.0)
            B1_offset = np.fmin(B1_offset, B1_uplimit / np.mean(W_idct_hat_flatten))
            B_offset = B1_offset * np.mean(W_idct_hat_flatten) - B1_offset * W_idct_hat_flatten
            A1_offset = np.mean(R1[:, validA1coeff_idx], axis=1) - np.mean(A1_coeff[validA1coeff_idx]) * W_idct_hat_flatten
            A1_offset -= np.mean(A1_offset)
            A_offset = A1_offset - np.mean(A1_offset) - B_offset
            # smooth A_offset
            W_offset = scipy.fft.dctn(np.reshape(A_offset, (p, q)), norm='ortho')
            W_offset = np.fmax(W_offset - lambda_darkfield / (ent2 * mu), 0) + \
                       np.fmin(W_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = scipy.fft.idctn(W_offset, norm='ortho')
            A_offset = A_offset.flatten()
            # encourage sparse A_offset
            A_offset = np.fmax(A_offset - lambda_darkfield / (ent2 * mu), 0) + \
                       np.fmin(A_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = A_offset + B_offset
        Z1 = D - A1_hat - E1_hat
        Y1 = Y1 + mu * Z1
        mu = min(mu * rho, mu_bar)

        # stop criterion
        stopCriterion = np.linalg.norm(Z1, 'fro')/ d_norm
        if stopCriterion < tol:
            converged = True

        # if np.mod(total_svd, 10) == 0:
        #     logger.info(f'Iteration {iter} |W|_0 {(np.abs(W_hat) > 0).sum()} |E1|_0 {(np.abs(E1_hat) > 0).sum()}'
        #                 f' stopCriterion {stopCriterion} B1_offset {B1_offset}')

        if not converged and iter >= maxIter:
            logger.info('Maximum iterations reached')
            converged = True

    A_offset = np.squeeze(A_offset)
    A_offset += B1_offset * W_idct_hat.flatten()
    return A1_hat, E1_hat, A_offset


def BaSiC(img_tiles: np.ndarray, *, lambdaa=None, estimation_mode='l0', max_iterations=500, optimization_tol=1e-6,
          estimate_darkfield=False, lambda_darkfield=None, working_size=128, max_reweightiterations=10, eplson=.1,
          varying_coeff=True, reweight_tol=1e-3):
    '''
    Estimation of flatfield for optical microscopy. Apply to a collection of
    monochromatic images. Multi-channel images should be separated, and each
    channel corrected separately.

    :param IF: nimg x nrows x ncols ndarray
    :param lambdaa: default value estimated from input images directly, high values (eg. 9.5) increase the spatial regularization strength, yielding a more smooth flatfield
    :param estimation_mode: default l0
    :param max_iterations: default 500
    :param optimization_tol: default 1e-6
    :param estimate_darkfield: whether you would like to estimate darkfield, keep 'false' if the input images are brightfield images or bright fluoresence images, set 'true' only if only if the input images are fluorescence images are dark and have a strong darkfield contribution.
    :param lambda_darkfield: default value estimated from input images directly, , high values (eg. 9.5) increase the spatial regularization strength, yielding a more smooth darkfield
    :param working_size: downsample to working_size x working_size before processing
    :param max_reweightiterations: default 10
    :param eplson: default .1, reweighting parameter
    :param varying_coeff:
    :param reweight_tol: default 1e-3, reweighting tolerance
    :return:
        - flatfield: estimated flatfield
        - darkfield: estimated darkfield

    reference: Peng et al. "A BaSiC tool for background and shading correction of optical microscopy images" Nature Communications, 14836(2017)
    '''
    tile_height, tile_width = img_tiles.shape[1:]
    D = np.moveaxis(img_util.imresize(img_tiles, des_height=working_size, des_width=working_size,
                                      interpolant=Interpolant.Linear).astype(np.float64),
                    0, -1)
    nrows, ncols, nimgs = D.shape
    meanD = np.mean(D, axis=-1)
    meanD /= np.mean(meanD)
    W_meanD = scipy.fft.dctn(meanD, norm='ortho')

    if lambdaa is None:
        lambdaa = np.abs(W_meanD).sum() / 400. * 0.5
    if lambda_darkfield is None:
        lambda_darkfield = np.abs(W_meanD).sum() / 400. * 0.2

    D.sort(axis=-1)
    XAoffset = np.zeros(shape=meanD.shape)

    weight = np.ones(shape=D.shape)
    i = 0
    flag_reweighting = True
    flatfield_last = np.ones(shape=meanD.shape)
    darkfield_last = np.random.randn(meanD.shape[0], meanD.shape[1])
    XA = None
    while flag_reweighting:
        i += 1
        logger.info(f'Reweighting Iteration {i}')
        X_k_A, X_k_E, X_k_Aoffset = inexact_alm_rspca_l1(D, lambdaa=lambdaa, lambda_darkfield=lambda_darkfield,
                                                         tol=optimization_tol, maxIter=max_iterations,
                                                         weight=weight, estimatedarkfield=estimate_darkfield)
        XA = X_k_A.reshape((nrows, ncols, -1))
        XE = X_k_E.reshape((nrows, ncols, -1))
        XAoffset = X_k_Aoffset.reshape((nrows, ncols))
        XE_norm = XE / (np.mean(XA, axis=(0,1), keepdims=True) + 1e-6)
        weight = 1. / (np.abs(XE_norm) + eplson)
        weight = weight * weight.size / weight.sum()
        temp = np.mean(XA, axis=-1) - XAoffset
        flatfield_current = temp / np.mean(temp)
        darkfield_current = XAoffset
        mad_flatfield = np.abs(flatfield_current - flatfield_last).sum() / np.abs(flatfield_last).sum()
        temp_diff = np.abs(darkfield_current - darkfield_last).sum()
        if temp_diff < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = temp_diff / max(1e-6, np.abs(darkfield_last).sum())
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if max(mad_flatfield, mad_darkfield) <= reweight_tol or i > max_reweightiterations:
            flag_reweighting = False

    # print(XA.shape, XAoffset.shape)
    shading = np.mean(XA, axis=-1) - XAoffset
    flatfield = img_util.imresize(shading, des_height=tile_height, des_width=tile_width)
    flatfield /= np.mean(flatfield)
    XAoffset = img_util.imresize(XAoffset, des_height=tile_height, des_width=tile_width)
    darkfield = XAoffset if estimate_darkfield else None
    return flatfield, darkfield


def BaSiC_basefluor(img_tiles: np.ndarray, flatfield: np.ndarray, *, darkfield=None, working_size=128):
    '''
    Estimation of background fluoresence signal for time-lapse movie. Used in conjunction with BaSiC

    :param IF: nimg x nrows x ncols ndarray
    :param flatfield: estimated flatfield
    :param darkfield: to supply your darkfield, note it should be the same size as your fieldfield
    :param working_size: downsample to working_size x working_size before processing
    :return:
        - fi_base: estimated background

    reference: Peng et al. "A BaSiC tool for background and shading correction of optical microscopy images" Nature Communications, 14836(2017)
    '''
    tile_height, tile_width = img_tiles.shape[1:]
    D = np.moveaxis(img_util.imresize(img_tiles, des_height=working_size, des_width=working_size,
                                      interpolant=Interpolant.Linear).astype(np.float64),
                    0, -1)
    nrows, ncols, nimgs = D.shape
    D = D.reshape((nrows * ncols, -1))
    flatfield = img_util.imresize(flatfield, des_height=working_size, des_width=working_size,
                                  interpolant=Interpolant.Linear).astype(np.float64)
    if darkfield is None:
        darkfield = np.zeros(shape=flatfield.shape)
    else:
        darkfield = img_util.imresize(darkfield, des_height=working_size, des_width=working_size,
                                      interpolant=Interpolant.Linear).astype(np.float64)

    weight = np.ones(shape=D.shape)
    eplson = 0.1
    tol = 1e-6
    for reweighting_iter in range(5):
        W_idct_hat = flatfield.flatten()
        A_offset = darkfield.flatten()
        A1_coeff = np.mean(D, axis=0)
        # main iteration loop starts
        temp = np.linalg.svd(D, full_matrices=False, compute_uv=False, hermitian=False)
        norm_two = temp[0]
        mu = 12.5/norm_two  # this one can be tuned
        mu_bar = mu * 1e7
        rho = 1.5  # this one can be tuned
        d_norm = np.linalg.norm(D, 'fro')
        ent1 = 1
        iter = 0
        total_svd = 0
        converged = False
        A1_hat = np.zeros(shape=D.shape)
        E1_hat = np.zeros(shape=D.shape)
        Y1 = 0
        while not converged:
            iter += 1
            A1_hat = W_idct_hat.reshape((-1, 1)) @ A1_coeff.reshape((1, -1)) + A_offset.reshape((-1, 1))
            # update E1 using l0 norm
            E1_hat = E1_hat + (D - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
            E1_hat = np.fmax(E1_hat - weight / (ent1 * mu), 0.) + np.fmin(E1_hat + weight / (ent1 * mu), 0.)
            # update A1_coeff, A2_coeff and A_offset
            R1 = D - E1_hat
            A1_coeff = np.mean(R1, axis=0) - np.mean(A_offset)
            A1_coeff[A1_coeff < 0] = 0

            Z1 = D - A1_hat - E1_hat
            Y1 = Y1 + mu * Z1
            mu = min(mu * rho, mu_bar)

            # stop criterion
            stopCriterion = np.linalg.norm(Z1, 'fro')/ d_norm
            if stopCriterion < tol:
                converged = True

            if np.mod(total_svd, 10) == 0:
                logger.info(f'Iteration {iter} |E1|_0 {(np.abs(E1_hat) > 0).sum()}'
                            f' stopCriterion {stopCriterion}')

        # update weight
        # XE_norm = bsxfun(@ldivide, E1_hat, mean(A1_hat))
        XE_norm = np.mean(A1_hat, axis=0) / E1_hat
        weight = 1. / (np.abs(XE_norm) + eplson)
        weight = weight * weight.size / weight.sum()

    fi_base = A1_coeff
    return fi_base


def correct_shading(input_filename, scene: int, *,
                    output_filename: str=None,
                    inverse_channels: tuple = tuple(),
                    correct_background_channels: tuple = tuple(),
                    correct_background_method: str = 'annotation',  # annotation or BaSiC or annotation_interpolate
                    correct_background_annotation: str = None,
                    correct_background_annotation_slice_idx=None,
                    correct_background_mask: np.ndarray = None,
                    tile_overlap_ratio: float = 0.05,
                    normalize_intensity_channels: tuple = tuple(),
                    normalize_intensity_clip_limit: float = 0.1,
                    ):
    infoList = ZImg.readImgInfos(input_filename)
    logger.info(f'Running {os.path.basename(input_filename)}, scene {scene}')
    if os.path.exists(output_filename):
        logger.info(f'{os.path.basename(output_filename)} already exist')
        return
    if len(infoList) <= scene:
        logger.info(f'Scene {scene} does not exist')
        return
    # print('image', infoList[scene])
    blockList = ZImg.getInternalSubRegions(input_filename)
    # np.set_printoptions(threshold=sys.maxsize)
    # print('czi blocks in image', blockList[scene])
    tile_width = blockList[scene][0].end.x - blockList[scene][0].start.x
    tile_height = blockList[scene][0].end.y - blockList[scene][0].start.y
    nchs = blockList[scene][0].end.c - blockList[scene][0].start.c
    ntiles = len(blockList[scene])
    # print(tile_width, tile_height, nchs)
    tile_img = ZImg(input_filename, region=blockList[scene][0], scene=scene)
    img_dtype = tile_img.data[0].dtype
    stacked_tiles = np.zeros(shape=(nchs, ntiles, tile_height, tile_width), dtype=img_dtype)
    res_mask = np.zeros(shape=(infoList[scene].depth, infoList[scene].height, infoList[scene].width),
                        dtype=np.uint8)
    for tile_idx, tile in enumerate(blockList[scene]):
        tile_img = ZImg(input_filename, region=tile, scene=scene)
        stacked_tiles[:, tile_idx, :, :] = tile_img.data[0][:, 0, :, :]
        res_mask[tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += 1
    res_mask[res_mask == 0] = 1
    res_img = np.zeros(shape=(nchs, infoList[scene].depth, infoList[scene].height, infoList[scene].width),
                       dtype=np.float64)

    for ch in inverse_channels:
        # print(ch, np.iinfo(img_dtype).max)
        stacked_tiles[ch, :, :, :] = np.iinfo(img_dtype).max - stacked_tiles[ch, :, :, :]

    if correct_background_method == 'annotation' or correct_background_method == 'annotation_interpolate':
        assert correct_background_annotation is not None or correct_background_mask is not None

    for ch in range(nchs):
        flatfield, darkfield = BaSiC(stacked_tiles[ch, :, :, :], estimate_darkfield=True)
        if ch in correct_background_channels:
            if correct_background_method == 'BaSiC':
                basefluor =  BaSiC_basefluor(stacked_tiles[ch, :, :, :], flatfield=flatfield, darkfield=darkfield)
                for tile_idx, tile in enumerate(blockList[scene]):
                    corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield - basefluor[tile_idx]
                    res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
            elif correct_background_method == 'annotation':
                ra_dict = region_annotation.read_region_annotation(correct_background_annotation)
                annotation_mask = region_annotation.convert_region_annotation_dict_to_binary_mask(ra_dict,
                                                                                                  height=infoList[scene].height,
                                                                                                  width=infoList[scene].width,
                                                                                                  slice_idx=correct_background_annotation_slice_idx)
                for tile_idx, tile in enumerate(blockList[scene]):
                    corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield
                    tile_region_mask = annotation_mask[tile.start.y:tile.end.y, tile.start.x:tile.end.x]
                    if tile_region_mask.sum() / tile_region_mask.size < 0.9:
                        corrected_tile[np.logical_not(tile_region_mask)] -= \
                            np.median(corrected_tile[np.logical_not(tile_region_mask)])
                    res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
            elif correct_background_method == 'annotation_interpolate':
                if correct_background_mask is not None:
                    assert correct_background_mask.ndim == 2 and \
                           correct_background_mask.shape[0] == infoList[scene].height and \
                           correct_background_mask.shape[1] == infoList[scene].width
                    annotation_mask = correct_background_mask.astype(np.bool)
                else:
                    ra_dict = region_annotation.read_region_annotation(correct_background_annotation)
                    annotation_mask = region_annotation.convert_region_annotation_dict_to_binary_mask(ra_dict,
                                                                                                      height=infoList[scene].height,
                                                                                                      width=infoList[scene].width,
                                                                                                      slice_idx=correct_background_annotation_slice_idx)

                annotation_mask = ndimage.binary_dilation(annotation_mask,
                                                          structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(
                                                              bool), iterations=200)
                annotation_mask = ndimage.binary_fill_holes(annotation_mask)

                for tile_idx, tile in enumerate(blockList[scene]):
                    tile_width = tile.end.x - tile.start.x
                    tile_height = tile.end.y - tile.start.y
                    break

                def get_grid_xy(tile):
                    grid_y = int((tile.end.y + tile.start.y) / 2. - tile_height * tile_overlap_ratio) // int(
                        tile_height * (1 - 2 * tile_overlap_ratio))
                    grid_x = int((tile.end.x + tile.start.x) / 2. - tile_width * tile_overlap_ratio) // int(
                        tile_width * (1 - 2 * tile_overlap_ratio))
                    return grid_x, grid_y

                max_grid_x = 0
                max_grid_y = 0
                for tile_idx, tile in enumerate(blockList[scene]):
                    grid_x, grid_y = get_grid_xy(tile)
                    max_grid_x = max(max_grid_x, grid_x)
                    max_grid_y = max(max_grid_y, grid_y)
                background_grid = np.zeros(shape=(max_grid_y + 1, max_grid_x + 1), dtype=np.float64)

                for tile_idx, tile in enumerate(blockList[scene]):
                    grid_x, grid_y = get_grid_xy(tile)
                    corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield
                    tile_region_mask = annotation_mask[tile.start.y:tile.end.y, tile.start.x:tile.end.x]
                    if tile_region_mask.sum() / tile_region_mask.size < 0.9:
                        background_grid[grid_y, grid_x] = np.median(corrected_tile[np.logical_not(tile_region_mask)])

                import scipy.sparse
                import scipy.interpolate
                coo = scipy.sparse.coo_matrix(background_grid)
                zfun_smooth_rbf = scipy.interpolate.Rbf(coo.row, coo.col, coo.data, function='cubic',
                                                        smooth=0)  # default smooth=0 for interpolation
                background_grid_out = np.fromfunction(np.vectorize(zfun_smooth_rbf), (max_grid_y + 1, max_grid_x + 1),
                                                      dtype=np.float64)

                for tile_idx, tile in enumerate(blockList[scene]):
                    grid_x, grid_y = get_grid_xy(tile)
                    corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield
                    estimated_background = background_grid_out[grid_y, grid_x]
                    corrected_tile -= estimated_background
                    res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
            else:
                assert False, f'unknown background correction method: {correct_background_method}'
        else:
            for tile_idx, tile in enumerate(blockList[scene]):
                corrected_tile = (stacked_tiles[ch, tile_idx, :, :].astype(np.float64) - darkfield) / flatfield
                res_img[ch, tile.start.z:tile.end.z, tile.start.y:tile.end.y, tile.start.x:tile.end.x] += corrected_tile
        res_img[ch, :, :, :] /= res_mask.astype(np.float64)

    res_img = np.clip(res_img, a_min=np.iinfo(img_dtype).min, a_max=np.iinfo(img_dtype).max).astype(img_dtype)

    for ch in normalize_intensity_channels:
        for d in range(res_img.shape[1]):
            img_adapteq = skimage.exposure.equalize_adapthist(res_img[ch, d, :, :],
                                                              clip_limit=normalize_intensity_clip_limit)
            res_img[ch, d, :, :] = (img_adapteq * np.iinfo(img_dtype).max).astype(img_dtype)

    for ch in inverse_channels:
        res_img[ch, :, :, :][res_img[ch, :, :, :] == np.iinfo(img_dtype).min] = np.iinfo(img_dtype).max
        res_img[ch, :, :, :] = np.iinfo(img_dtype).max - res_img[ch, :, :, :]

    if output_filename is not None:
        img = ZImg(res_img, infoList[scene])
        img.save(output_filename)
    return res_img, infoList[scene]


if __name__ == "__main__":
    from utils import logger as logg
    logg.setup_logger()

    # compare with matlab
    # import glob
    # folder = os.path.join(os.path.expanduser('~/code/matlab/BaSiC-master/Demoexamples/WSI_Brain'), 'Uncorrected_tiles')
    # filelist = glob.glob(os.path.join(folder, '*.tif'))
    # img = ZImg(filelist, catDim=Dimension.Z)
    # flatfield, darkfield = BaSiC(img.data[0][0], estimate_darkfield=True)

    if False:
        folder = os.path.join(io.fs3017_data_dir(), 'lemur', 'Hotsauce_334A',
                              '181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        czi_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_21.czi')
        scene = 0
        output_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_21_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename)

    if False:
        folder = os.path.join('/Volumes/T7Touch/Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1')
        czi_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_37.czi')
        scene = 1
        annotation_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_37_scene{scene}.reganno')
        output_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_37_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                        inverse_channels=(0,), correct_background_channels=(1,),
                        correct_background_annotation=annotation_filename)

    if False:
        folder = os.path.join('/Volumes/T7Touch/Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1')
        czi_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_24.czi')
        scene = 3
        annotation_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_24_scene{scene}.reganno')
        output_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_24_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                        inverse_channels=(0,), correct_background_channels=(1,),
                        correct_background_annotation=annotation_filename)

    if False:
        folder = os.path.join('/Volumes/T7Touch/Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1')
        czi_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_27.czi')
        scene = 3
        annotation_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_27_scene{scene}.reganno')
        output_filename = os.path.join(folder, f'Lemur-J_vGluT2_SMI32_vGluT1_27_scene{scene}_shading_correction.nim')

        correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                        inverse_channels=(0,), correct_background_channels=(1,),
                        correct_background_annotation=annotation_filename)

    if False:
        for scene in range(4):
            folder = os.path.join('/Users/feng/Documents/181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
            czi_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_23.czi')
            annotation_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_23_scene{scene}.reganno')
            output_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_23_scene{scene}_shading_correction.nim')

            correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                            inverse_channels=(0,), correct_background_channels=(1,),
                            correct_background_annotation=annotation_filename)


    if True:
        folder_list = ['Fig_325AA/180918_Lemur-Fig_PV_TH_NeuN',
                       'Fig_325AA/180914_fig_SMI99_NeuN_VGlut2',
                       'Garlic_320CA/181023_Lemur-Garlic_SMI99_VGluT2_M2',
                       'Hotsauce_334A/181016_Lemur-Hotsauce_PV_TH_NeuN',
                       'Hotsauce_334A/181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN',
                       'Icecream_225BD/190221_icecream_PV_TH_NeuN',
                       'Icecream_225BD/20190218_icecream_SMI99_NeuN_VGlut2',
                       'Jellybean_289BD/20190813_jellybean_FOXP2_SMI32_NeuN',
                       'Jellybean_289BD/20190827_jellybean_vGluT2_SMI32_vGluT1']
        prefix_list = ['Lemur-F_PV_TH_NeuN',
                       'Lemur-F_SMI99_NeuN_VGlut2',
                       'Lemur-G_SMI99_VGluT2_M2',
                       'Lemur-H_PV_TH_NeuN',
                       'Lemur-H_SMI99_VGluT2_NeuN',
                       'Lemur-I_PV_TH_NeuN',
                       'Lemur-I_SMI99_VGluT2_NeuN',
                       'Lemur-J_FOXP2_SMI32_NeuN',
                       'Lemur-J_vGluT2_SMI32_vGluT1']

        lemur_folder = os.path.join(io.fs3017_data_dir(), 'lemur')
        for idx in [6]:
            logger.info(f'Running {folder_list[idx]}')
            folder = os.path.join(lemur_folder, folder_list[idx])
            stack_anno_filename = os.path.join(folder, '00_stacked_annotation.reganno')
            # ra_dict = region_annotation.read_region_annotation(stack_anno_filename)
            info_filename = os.path.join(folder, 'info.txt')

            if os.path.exists(info_filename):
                brain_info = read_brain_info(info_filename)
            else:
                brain_info = None
            if brain_info is None:
                # param_set (czi_file, scene, annotation_file, result_file)
                # param_set (slice_idx, annotation_file, result_file)
                slice_list = list(range(190))
                czi_file_list = [math.ceil((idx + 1) / 4) for idx in slice_list]
                scene_list = [idx % 4 for idx in slice_list]
            else:
                slice_list = list(range(len(brain_info['filename'])))
                czi_file_list = [int(re.split('^(.*)_([0-9]+).czi$', fn)[2]) for fn in brain_info['filename']]
                scene_list = [int(fn)-1 for fn in brain_info['scene']]
            param_set = [(slice_list[idx], czi_file_list[idx], scene_list[idx]) for idx in range(len(slice_list))]

            def pool_correct_shading(parameter_tuple: list):
                slice_idx = parameter_tuple[0]
                czi_file_idx = parameter_tuple[1]
                scene_idx = parameter_tuple[2]+1
                czi_filename = os.path.join(folder, f'{prefix_list[idx]}_{czi_file_idx:02}.czi')
                output_filename = os.path.join(folder, 'background_corrected', f'{prefix_list[idx]}'
                                                                               f'_{czi_file_idx:02}_scene'
                                                                               f'{scene_idx}_background_corrected.nim')
                annotation_filename = os.path.join(folder, 'annotation', f'{prefix_list[idx]}_{czi_file_idx:02}_scene'
                                    f'{scene_idx}.reganno')
                if not os.path.exists(annotation_filename):
                    slice_dict = region_annotation.extract_region_annotation_slice(ra_dict, slice_idx)
                    region_annotation.write_region_annotation_dict(slice_dict,annotation_filename)

                correct_shading(czi_filename, scene=scene_idx-1, output_filename=output_filename,
                                inverse_channels=(0,), correct_background_channels=(1,),
                                correct_background_method='annotation_interpolate',
                                correct_background_annotation=annotation_filename,
                                tile_overlap_ratio=0.05,
                                normalize_intensity_channels=(1,),
                )

            with multiprocessing.Pool(5) as pool:
                pool.map_async(pool_correct_shading, param_set,
                               chunksize=1, callback=None).wait()
    if False:
        folder = os.path.join('/Users/feng/Documents/181005_Lemur-Hotsauce_SMI99_VGluT2_NeuN')
        for czi_file_idx in range(45):
            czi_filename = os.path.join(folder, f'Lemur-H_SMI99_VGluT2_NeuN_{czi_file_idx + 1:02}.czi')
            for scene in range(4):
                annotation_filename = os.path.join(folder, 'annotation',
                                                   f'Lemur-H_SMI99_VGluT2_NeuN_{czi_file_idx + 1:02}_scene{scene + 1}.reganno')
                output_filename = os.path.join(folder, 'background_corrected',
                                               f'Lemur-H_SMI99_VGluT2_NeuN_{czi_file_idx + 1:02}_scene{scene + 1}_background_corrected.nim')

                if os.path.exists(output_filename):
                    continue

                correct_shading(czi_filename, scene=scene, output_filename=output_filename,
                                inverse_channels=(0,), correct_background_channels=(1,),
                                correct_background_method='annotation_interpolate',
                                correct_background_annotation=annotation_filename,
                                tile_overlap_ratio=0.05,
                                normalize_intensity_channels=(1,),
                )

            with multiprocessing.Pool(5) as pool:
                pool.map_async(pool_correct_shading, param_set,
                               chunksize=1, callback=None).wait()



