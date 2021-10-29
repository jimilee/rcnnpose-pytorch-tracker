import os.path as osp
import os
import pathlib


########################################################################################
# Specification of paths where dataset and output results will be stored
########################################################################################
TARGET_DATASET = {'MOT20'}
TARGET_DATA = 'MOT20'

PROJECT_PATH = pathlib.Path(__file__).parent.absolute()
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

print(BASE_DIR)
CHALLENGE_PATH = BASE_DIR +'/data/'+TARGET_DATA+'-train/SSL_MOT/'
CKPT = 'MOT Aug( w cutout, jitter0.2)+Market C-Pose-GAN Aug (w cutout.pt'
SIMSIAM_PATH = BASE_DIR + '/simsiam/ckpt/'+CKPT

PREDATA_PATH = osp.join(PROJECT_PATH, TARGET_DATA+'_dets')#MOT17, MOT20 det
# PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT16_dets')#det
#PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT20', 'train')#gt (for making traindata)

# Absolute path where datasets and processed data (e.g. precomputed embeddings) will be stored
DATA_PATH = None

# Absolute path where results (e.g. output result files, model weights) will be stored
OUTPUT_PATH = None

if DATA_PATH is None:
    DATA_PATH = osp.join(PROJECT_PATH)

if OUTPUT_PATH is None:
    OUTPUT_PATH = osp.join(PROJECT_PATH, 'output')

SC1 = 0.4 # geometric score
SC2 = 0.6 # simsiam score

detTH = 0.6 # MOT16
ovlTH = 0.5
updateTH = 0.5
ageTH = 20

hierarchy = 0.9 #MOT20
'''
Fol MOT Challenge Eval
'''
EVAL_CONFIG = {'USE_PARALLEL': False,
               'NUM_PARALLEL_CORES': 1,
               'BREAK_ON_ERROR': True,
               'RETURN_ON_ERROR': False,
               'LOG_ON_ERROR': 'C:/Users/CVPR_JIMILEE/Desktop/TrackEval-master/error_log.txt',
               'PRINT_RESULTS': True,
               'PRINT_ONLY_COMBINED': False,
               'PRINT_CONFIG': False,
               'TIME_PROGRESS': False,
               'DISPLAY_LESS_PROGRESS': False,
               'OUTPUT_SUMMARY': False,
               'OUTPUT_EMPTY_CLASSES': False,
               'OUTPUT_DETAILED': False,
               'PLOT_CURVES': False}

DATASET_CONFIG = {'PRINT_CONFIG': False,
                  'GT_FOLDER': 'E:/_workspace/rcnnpose-pytorch-tracker/data/gt/mot_challenge/',
                  'TRACKERS_FOLDER': 'E:/_workspace/rcnnpose-pytorch-tracker/data/trackers/mot_challenge/',
                  'OUTPUT_FOLDER': None,
                  'TRACKERS_TO_EVAL': ['SSL_MOT'],
                  'CLASSES_TO_EVAL': ['pedestrian'],
                  'BENCHMARK': TARGET_DATA,
                  'SPLIT_TO_EVAL': 'train',
                  'INPUT_AS_ZIP': False,
                  'DO_PREPROC': True,
                  'TRACKER_SUB_FOLDER': 'data',
                  'OUTPUT_SUB_FOLDER': '',
                  'TRACKER_DISPLAY_NAMES': None,
                  'SEQMAP_FOLDER': None,
                  'SEQMAP_FILE': None,
                  'SEQ_INFO': None,
                  'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                  'SKIP_SPLIT_FOL': False}

METRICS_CONFIG = {'METRICS': ['CLEAR', 'Identity'], 'THRESHOLD': 0.5, 'PRINT_CONFIG': False}

Kalman = False


T = {
    '01': [0.4, 0.4, 0.6],
    #'02': [0.5, 0.4, 0.6],##best for MOT17
    #'03': [0.6, 0.4, 0.6],
    #'04': [0.7, 0.4, 0.6],
    #'05': [0.9, 0.4, 0.6], ##best for MOT20
    #'06': [1.0, 0.4, 0.6],
    #'07': [0.9, 0.4, 0.6],
    #'08': [2.5, 0.4, 0.6],
    #'09': [3.0, 0.4, 0.6]
    #'10': [0.9, 0.4, 0.6],
    #'11': [1.0, 0.4, 0.6]
}

