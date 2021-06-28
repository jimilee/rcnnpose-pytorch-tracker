import os.path as osp
import pathlib


########################################################################################
# Specification of paths where dataset and output results will be stored
########################################################################################
TARGET_DATASET = {'MOT16'}

CHALLENGE_PATH = 'C:/Users/CVPR_JIMILEE/Desktop/motchallenge-devkit/res/MOT16/data/'

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()

PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT16_dets')

MODEL_SAVE_PATH = osp.join(PROJECT_PATH, 'backup','keypoint', 'all')

# Absolute path where datasets and processed data (e.g. precomputed embeddings) will be stored
DATA_PATH = None

# Absolute path where results (e.g. output result files, model weights) will be stored
OUTPUT_PATH = None

if DATA_PATH is None:
    DATA_PATH = osp.join(PROJECT_PATH)

if OUTPUT_PATH is None:
    OUTPUT_PATH = osp.join(PROJECT_PATH, 'output')

T = {'01':[0.0, 0.1, 0.2, 0.3, 0.4], #2
     '02':[0.0, 0.1, 0.2, 0.3, 0.4], ##train
     '03':[0.0, 0.2, 0.2, 0.3, 0.3], #4
     '04':[0.0, 0.2, 0.2, 0.3, 0.3], ## fix
     '05':[0.0, 0.5, 0.1, 0.2, 0.2], ## fix
     '06':[0.0, 0.0, 0.3, 0.4, 0.3], #5
     '07':[0.0, 0.4, 0.2, 0.2, 0.2], #10
     '08':[0.0, 0.0, 0.3, 0.4, 0.3], #11
     '09':[0.0, 0.1, 0.2, 0.3, 0.4], ##
     '10':[0.0, 0.4, 0.2, 0.2, 0.2], ## fix
     '11':[0.0, 0.1, 0.2, 0.3, 0.4], ##
     '12':[0.0, 0.0, 0.3, 0.4, 0.3], #11
     '13':[0.0, 0.4, 0.2, 0.2, 0.2], ##
     '14':[0.0, 0.4, 0.2, 0.2, 0.2]} #13