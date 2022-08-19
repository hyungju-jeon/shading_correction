__all__ = ["read_brain_info"]

import os

def read_brain_info(info_file:str):
    """
    Read brain infofile and return dict with filename, scene, flipped, damaged as keys

    Arguments
    ---------
        info_file: filename with path to brain info file

    Returns
    -------
    dict containing following key/value pairs:
        `filename`  : List of filenames with path included
        `scene`     : List of scence number. Note that scence number starts from 1 
        `flipped`   : List of flipped flag. flipped = 1 denotes a flipped image 
        `damaged`   : List of damaged flag. flipped = 1 denotes a damaged image 

    Notes
    -----

    Example
    -------
    >>> 
    """
    file_folder = os.path.dirname(info_file)
    brain_info = {}
    with open(info_file, 'rt') as file:
        line = next(file)

        keyList = line.strip().replace(" ", "").split(',')
        for key in keyList:
            brain_info[key] = []
        for line in file:
            if line == '\n':
                continue
            line = line.strip()
            values = line.replace(" ", "").split(',')
            values[0] = os.path.join(file_folder, values[0])

            for idx in range(len(keyList)):
                brain_info[keyList[idx]].append(values[idx])

    return brain_info

def read_midline_info(info_file:str, ratio:int = 1):
    """
    Read brain midline info and return dict with filename, scene, midline points as keys

    Arguments
    ---------
        info_file   : filename with path to brain midline info file
        ratio       : (optional) Scaling ratio

    Returns
    -------
    dict containing following key/value pairs:
        `filename`  : List of filenames with path included
        `scene`     : List of scence number. Note that scence number starts from 1 
        `pts`       : List of coordinate for points used to describe midline. 
                      Lines are determined by two points (x1,y1) and (x2,y2)
                      Data are arranged in [x1, x2, y1, y2]

    Notes
    -----

    Example
    -------
    >>> 
    """
    file_folder = os.path.dirname(info_file)
    midline_info = {}
    with open(info_file, 'rt') as file:
        keyList = ['filename', 'scene', 'pts']
        for key in keyList:
            midline_info[key] = []
        for line in file:
            if line == '\n':
                continue
            line = line.strip()
            values = line.replace(" ", "").split(',')
            values[0] = os.path.join(file_folder, values[0])

            midline_pts = [];
            for idx in range(len(values)):
                if idx > 1:
                    midline_pts.append(float(values[idx])*64/ratio)
                else:
                    midline_info[keyList[idx]].append(values[idx])
            midline_info['pts'].append(midline_pts)

    return midline_info

