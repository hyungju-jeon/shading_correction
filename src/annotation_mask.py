import os 
import sys
sys.path.append('../')

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree
from allensdk.core.reference_space import ReferenceSpace

from zimg import *
from utils import img_util
from utils import brain_info

import pandas as pd
import math
import numpy as np
import ants
import nrrd
import matplotlib.pyplot as plt
import functools

if __name__ == "__main__":
    
    os.chdir('/users/Yoonkyoung/Desktop/041715_JK359-2_L/')

    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])  # 1 is the id of the adult mouse structure graph
    structure_graph = StructureTree.clean_structures(structure_graph)  
    
    tree = StructureTree(structure_graph)
    tree.get_structures_by_name(['Isocortex'])
    
    annotation_volume_ZImg = ZImg('/users/Yoonkyoung/Desktop/041715_JK359-2_L/ARA_anno_rotated_25.mhd')
    annotation_volume = annotation_volume_ZImg.data[0]
    annotation_volume = np.squeeze(annotation_volume)
    rsp = ReferenceSpace(tree, annotation_volume, [25, 25, 25])
    
    whole_cortex_mask = rsp.make_structure_mask([567]) #id
    # whole_cortex_mask = np.moveaxis(whole_cortex_mask, [0,1,2], [2,1,0])
    
    img_util.write_img(filename= '/users/Yoonkyoung/Desktop/041715_JK359-2_L/cortex_mask.tiff', 
                          img_data = whole_cortex_mask)
  