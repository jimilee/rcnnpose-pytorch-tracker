import os.path as osp
import pathlib


########################################################################################
# Specification of paths where dataset and output results will be stored
########################################################################################
TARGET_DATASET = {'MOT16'}

CHALLENGE_PATH = 'C:/Users/CVPR_JIMILEE/Desktop/motchallenge-devkit/res/MOT16/data/'

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()

PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT16_dets')#det
#PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT20', 'train')#gt

# Absolute path where datasets and processed data (e.g. precomputed embeddings) will be stored
DATA_PATH = None

# Absolute path where results (e.g. output result files, model weights) will be stored
OUTPUT_PATH = None

if DATA_PATH is None:
    DATA_PATH = osp.join(PROJECT_PATH)

if OUTPUT_PATH is None:
    OUTPUT_PATH = osp.join(PROJECT_PATH, 'output')

SC1 = 0.5 # geometric score
SC2 = 0.5 # simsiam score

detTH = 0.6
ovlTH = 0.5
updateTH = 0.5
ageTH = 20

hierarchy = 0.5

# T = {'02':[0.2, 0.2, 0.6], ##train
#      '04':[0.2, 0.2, 0.6], ## fix
#      '05':[0.2, 0.6, 0.2], ## fix
#      '09':[0.2, 0.6, 0.2], ##
#      '10':[0.2, 0.6, 0.2], ## fix
#      '11':[0.2, 0.6, 0.2], ##
#      '13':[0.2, 0.0, 0.8], ##
#
#      '01':[0.0, 0.6, 0.4], #2
#      '03':[0.0, 0.4, 0.6], #4
#      '06':[0.0, 0.4, 0.6], #5
#      '07':[0.0, 0.4, 0.6], #10
#      '08':[0.0, 0.2, 0.8], #11
#      '12':[0.0, 0.3, 0.7], #11
#      '14':[0.0, 0.3, 0.7]} #13

T = {'02':[0.0, 1.0, 0.0], ##train
     '04':[0.0, 0.9, 0.1], ## fix
     '05':[0.0, 0.9, 0.1], ## fix
     '09':[0.0, 0.9, 0.1], ##
     '10':[0.0, 0.9, 0.1], ## fix
     '11':[0.0, 0.9, 0.1], ##
     '13':[0.0, 0.9, 0.1], ##

     '01':[0.0, 0.6, 0.4], #2
     '03':[0.0, 0.4, 0.6], #4
     '06':[0.0, 0.4, 0.6], #5
     '07':[0.0, 0.4, 0.6], #10
     '08':[0.0, 0.2, 0.8], #11
     '12':[0.0, 0.3, 0.7], #11
     '14':[0.0, 0.3, 0.7]} #13