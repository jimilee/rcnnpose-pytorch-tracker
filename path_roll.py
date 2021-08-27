import os.path as osp
import pathlib


########################################################################################
# Specification of paths where dataset and output results will be stored
########################################################################################
TARGET_DATASET = {'MOT20'}
TARGET_DATA = 'MOT20'

#CHALLENGE_PATH = 'C:/Users/CVPR_JIMILEE/Desktop/motchallenge-devkit/res/MOT16/data/'
CHALLENGE_PATH = 'E:/_workspace/rcnnpose-pytorch-tracker/data/trackers/mot_challenge/'+TARGET_DATA+'-train/SSL_MOT/'
CKPT = 'MOT Aug( w cutout, jitter0.2)+Market C-Pose-GAN Aug (w cutout.pt'
# CKPT = 'table3_simsiam.pt'
SIMSIAM_PATH = 'E:/_workspace/rcnnpose-pytorch-tracker/simsiam/ckpt/'+CKPT

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute()

PREDATA_PATH = osp.join(PROJECT_PATH, TARGET_DATA+'_dets')#det
# PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT16_dets')#det
#PREDATA_PATH = osp.join(PROJECT_PATH, 'MOT20', 'train')#gt

# Absolute path where datasets and processed data (e.g. precomputed embeddings) will be stored
DATA_PATH = None

# Absolute path where results (e.g. output result files, model weights) will be stored
OUTPUT_PATH = None

if DATA_PATH is None:
    DATA_PATH = osp.join(PROJECT_PATH)

if OUTPUT_PATH is None:
    OUTPUT_PATH = osp.join(PROJECT_PATH, 'output')

SC1 = 0.6 # geometric score
SC2 = 0.4 # simsiam score

detTH = 0.6 # MOT16
ovlTH = 0.5
updateTH = 0.5
ageTH = 20

hierarchy = 0.5

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
'''
#MOT20
T = {'01':[-1, 0.5, 0.5], #2 static
     '02':[-1, 0.5, 0.5], ##train static
     '03':[-1, 0.5, 0.5], #4 static
     '05':[-1, 0.5, 0.5], ## fix moving

     '04':[0, 0.5, 0.5], ## fix static
     '06':[0, 0.5, 0.5], #5 moving
     '07':[0, 0.5, 0.5], #10 moving
     '08':[0, 0.5, 0.5], #11 static
     }
'''
'''
T = {'02':[0, 0.5, 0.5], ##train static
     '04':[0, 0.5, 0.5], ## fix static
     '05':[-1, 0.4, 0.6], ## fix moving
     '09':[0, 0.2, 0.8], ## static
     '10':[-1, 0.0, 1.0], ## fix moving
     '11':[-1, 0.4, 0.6], ## moving
     '13':[-1, 0.3, 0.7], ## moving

     '01':[0, 0.5, 0.5], #2 static
     '03':[0, 0.5, 0.5], #4 static
     '06':[-1, 0.4, 0.6], #5 moving
     '07':[-1, 0.0, 1.0], #10 moving
     '08':[0, 0.2, 0.8], #11 static
     '12':[-1, 0.4, 0.6], #11 moving
     '14':[-1, 0.3, 0.7]} #13 moving
'''
Kalman = False
T = {
    #'01': [0, 0.0, 1.0],
    #'02': [0, 0.1, 0.9],
    #'03': [0, 0.2, 0.8],
    #'04': [0, 0.3, 0.7],
    '05': [0, 0.4, 0.6] ##best for MOT20
    #'06': [0, 0.5, 0.5],
    #'07': [0, 0.6, 0.4],
    #'08': [0, 0.7, 0.3],
    #'09': [0, 0.8, 0.2],
    #'10': [0, 0.9, 0.1],
    #'11': [0, 1.0, 0.0],
}