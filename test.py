import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap

from hloc.utils.read_write_model import read_model
from hloc.utils.parsers import (
    parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)


reference_sfm='outputs/huawei/sfm_superpoint+superglue_huawei/model'
_, db_images, points3D = read_model(str(reference_sfm), '.bin')

db_id=1089
print(db_images[db_id])
points3D_ids = db_images[db_id].point3D_ids
print(len(points3D_ids))


