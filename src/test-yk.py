
import sys
sys.path.append('../')

from utils import brain_info
from utils import img_util

if __name__ == "__main__":
    info_file = '/users/Yoonkyoung/Desktop/041715_JK359-2_L/info.txt'
    brain_info = brain_info.read_brain_info(info_file)

    for i in range(0,6):
        name = brain_info['filename'][i]
        filename = name.split('/')[5]
        scene = brain_info['scene'][i]
        
        print(filename)

