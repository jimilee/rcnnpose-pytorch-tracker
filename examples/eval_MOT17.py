import time
from collections import deque

from tqdm import tqdm

import winsound as sd
import path_roll as roll
import os
import os.path as osp
import cv2
import numpy as np
import trackeval

from examples.tracker import tracker
from examples.tracker_utils import print_tracking_result, save_crop_bbox_img
from rcnnpose.utils import draw_tracker_boxes
import openpyxl
#
def beepsound():
    fr = 400    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def track_all_seq(target_='train', show = False):
    TEST_ROLL = roll.T.keys()

    MOT_DATA = roll.TARGET_DATASET
    st = tracker()
    proctime = 0
    for dataset in MOT_DATA:
        print(osp.join(roll.DATA_PATH, dataset, target_))
        for seq in os.listdir(osp.join(roll.DATA_PATH, dataset, target_)):
            file_target = os.path.join(osp.join(roll.PREDATA_PATH, seq + '.txt'))
            #file_target = os.path.join(osp.join(roll.PREDATA_PATH, seq, 'gt', 'gt.txt'))
            img_target = osp.join(roll.DATA_PATH, dataset, target_, seq, seq)

            cnt_id_dict = {}
            print('img_target ',img_target)
            print('target_file is :', file_target)
            if not (
                    osp.join(roll.CHALLENGE_PATH, seq[9:])):  # 폴더 경로 생성
                os.makedirs(osp.join(roll.CHALLENGE_PATH, seq[9:]))

            for test_idx in TEST_ROLL:

                challenge_path = roll.CHALLENGE_PATH + 'data_' + test_idx + '/'

                if not os.path.exists(challenge_path):
                    os.makedirs(challenge_path)
                challenge_path = osp.join(challenge_path, '{0}.txt'.format(seq))

                if os.path.isfile(challenge_path):
                    #print(challenge_path, 'is exist! make new.')  # 이미 챌린지 출력결과가 있을경우, 해당 파일 삭제.
                    #os.remove(challenge_path)
                    print(challenge_path, 'is exist! pass this seq.')  # 패스.
                    continue

                print('challenge_path : ',challenge_path)

                bbox = []
                with open(file_target, 'r', encoding='utf-8') as f:
                    while True:
                        line = f.readline()
                        if not line: break
                        frame, id1, x, y, w, h, score, n1, n2, n3 = line.split(sep=',')
                        # frame, id1, x, y, w, h, score, cls, vis = line.split(sep=',') #gt
                        # if int(cls) == 1 and float(vis) > 0.4 :  # gt.
                        # if float(score) > roll.detTH:
                        # print(frame, id1, x, y, w, h, score, cls, vis)
                        bbox.append((frame, id1, x, y, w, h, score))

                bbox = sorted(bbox, key=lambda x: int(x[0]))  # 프레임순 정렬.
                dq = deque(bbox)
                frame_cnt = 0

                st.init_id_tracker(st.max_tracker, test_idx)

                for img in tqdm(os.listdir(img_target)):
                    frame_cnt = frame_cnt + 1
                    src = cv2.imread(osp.join(img_target, img))
                    det_boxes = []
                    gt_boxes = []
                    while (len(dq)):
                        bbox = dq.popleft()
                        if int(bbox[0]) == frame_cnt:
                            id = int(bbox[1])
                            sx = int(float(bbox[2]))
                            sy = int(float(bbox[3]))
                            bx = sx + int(float(bbox[4]))
                            by = sy + int(float(bbox[5]))
                            det_boxes.append([sx,sy,bx,by])
                            gt_boxes.append([id,sx,sy,bx,by])
                        else:
                            dq.appendleft(bbox)  # 다음 프레임은 다시 넣음
                            break
                    # print(type(np.array(det_boxes)))
                    # save_crop_bbox_img(src, np.array(gt_boxes), frame_cnt, seq)
                    #start = time.time()
                    target = st.tracking(np.array(det_boxes), src, frame_cnt)
                    #end = time.time()
                    #proctime += end-start
                    print_tracking_result(target, challenge_path, frame_cnt)

                    # # 트래킹오버레이 확인용 화면 출력.
                    if(show):
                        overlay_tk = draw_tracker_boxes(src, target, frame_cnt)
                        cv2.imshow('Video Demo', overlay_tk)
                        if cv2.waitKey(20) & 0xff == 27:  # exit if pressed `ESC`
                            break

                print('total proctime : ',proctime)
    return proctime
#

target_seq = 'test'
total_frame = 0
tracking_time = track_all_seq(target_ = target_seq, show=False)
if roll.TARGET_DATASET == {'MOT16'}:
    total_frame = 5316 if target_seq == 'train' else 5919
if roll.TARGET_DATASET == {'MOT17'}:
    total_frame = 15948 if target_seq == 'train' else 17757
if roll.TARGET_DATASET == {'MOT20'}:
    total_frame = 8931 if target_seq == 'train' else 4479

if tracking_time:
    print("done. total process time : {0}, FPS : {1}".format(tracking_time, float(1/(tracking_time/total_frame)))) #5316 is total frame of MOT train-set

# make exel file.
# exel_path = 'C:/Users/CVPR_JIMILEE/Desktop/motchallenge-devkit/result.xlsx'
# wb = openpyxl.load_workbook(exel_path)
# sheet = wb.active
# sheet.append({'A':'IDF1','B':'MOTA','C':roll.CKPT,'D':roll.SC1,'E':roll.SC2,'F':roll.detTH,'G':roll.ovlTH,'H':roll.updateTH,'I':roll.ageTH,'J':roll.hierarchy})
#
# wb.save(exel_path)
# print("exel result saved. {0}".format(exel_path))

# MOT evaluation.
best_MOTA = {}
best_IDF1 = {}
for test_idx in roll.T.keys():
    dataset_config = roll.DATASET_CONFIG
    dataset_config['TRACKER_SUB_FOLDER'] = 'data_'+test_idx
    print('\n Testing data_{0} . . . '.format(test_idx))
    evaluator = trackeval.Evaluator(roll.EVAL_CONFIG)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in roll.METRICS_CONFIG['METRICS']:
            metrics_list.append(metric(roll.METRICS_CONFIG))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg  = evaluator.evaluate(dataset_list, metrics_list)
    MOTA = output_res['MotChallenge2DBox']['SSL_MOT']['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
    IDF1 = output_res['MotChallenge2DBox']['SSL_MOT']['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']

    best_MOTA[test_idx] = MOTA
    best_IDF1[test_idx] = IDF1

print(best_MOTA)
print(best_IDF1)

beepsound()
    # print(times)
    # m, s = divmod(sum(times.values()), 60)
    # print('Completed after ', sum(times.values()), ' / {0}:{1}'.format(int(m), int(s)))