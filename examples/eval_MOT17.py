import time
from collections import deque

from tqdm import tqdm

import winsound as sd
import path_roll as roll
import os
import os.path as osp
import cv2
import numpy as np
from examples.tracker import tracker
from examples.tracker_utils import print_tracking_result, save_crop_bbox_img
from rcnnpose.utils import draw_tracker_boxes
import openpyxl

#

def beepsound():
    fr = 500    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def track_all_seq(target_='train', show = False):
    MOT_DATA = roll.TARGET_DATASET
    st = tracker()
    proctime = 0
    for dataset in MOT_DATA:
        print(osp.join(roll.DATA_PATH, dataset, target_))
        for seq in os.listdir(osp.join(roll.DATA_PATH, dataset, target_)):

            T_seq = seq[6:8]
            st.init_id_tracker(st.max_tracker, T_seq)
            challenge_path = osp.join(roll.CHALLENGE_PATH, seq[9:], '{0}.txt'.format(seq))
            file_target = os.path.join(osp.join(roll.PREDATA_PATH, seq + '.txt'))
            #file_target = os.path.join(osp.join(roll.PREDATA_PATH, seq, 'gt', 'gt.txt'))
            img_target = osp.join(roll.DATA_PATH, dataset, target_, seq, seq)  # MOT17

            cnt_id_dict = {}
            print('img_target ',img_target)
            print('target_file is :', file_target)
            if not (
                    osp.join(roll.CHALLENGE_PATH, seq[9:])):  # 폴더 경로 생성
                os.makedirs(osp.join(roll.CHALLENGE_PATH, seq[9:]))

            if os.path.isfile(challenge_path):
                print(challenge_path, 'is exist! make new.')  # 이미 챌린지 출력결과가 있을경우, 해당 파일 삭제.
                os.remove(challenge_path)
                # print(challenge_path, 'is exist! pass this seq.')   # 패스.
                # continue

            bbox = []
            with open(file_target, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    frame, id1, x, y, w, h, score, n1, n2, n3 = line.split(sep=',')
                    #frame, id1, x, y, w, h, score, cls, vis = line.split(sep=',') #gt
                    #if int(cls) == 1 and float(vis) > 0.4 :  # gt.
                    #if float(score) > roll.detTH:
                        # print(frame, id1, x, y, w, h, score, cls, vis)
                    bbox.append((frame, id1, x, y, w, h, score))

            bbox = sorted(bbox, key=lambda x: int(x[0]))  # 프레임순 정렬.
            dq = deque(bbox)
            frame_cnt = 0
            idx = 0


            for img in tqdm(os.listdir(img_target)):
                frame_cnt = frame_cnt + 1
                src = cv2.imread(osp.join(img_target, img))
                det_boxes = []
                gt_boxes = []
                while (len(dq)):
                    bbox = dq.popleft()
                    # print(bbox)
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
                start = time.time()
                target = st.tracking(np.array(det_boxes), src, frame_cnt)
                end = time.time()
                proctime += end-start
                # # 트래킹오버레이 확인용 화면 출력.
                if(show):
                    overlay_tk = draw_tracker_boxes(src, target, frame_cnt)
                    cv2.imshow('Video Demo', overlay_tk)
                    if cv2.waitKey(20) & 0xff == 27:  # exit if pressed `ESC`
                        break

                print_tracking_result(target, challenge_path, frame_cnt)
    print(proctime)
    return proctime

tracking_time = track_all_seq()

print("done. total process time : {0}, FPS : {1}".format(tracking_time, float(1/(tracking_time/5316)))) #5316 is total frame of MOT train-set
# exel_path = 'C:/Users/CVPR_JIMILEE/Desktop/motchallenge-devkit/result.xlsx'
# wb = openpyxl.load_workbook(exel_path)
# sheet = wb.active
# sheet.append({'A':'IDF1','B':'MOTA','C':roll.CKPT,'D':roll.SC1,'E':roll.SC2,'F':roll.detTH,'G':roll.ovlTH,'H':roll.updateTH,'I':roll.ageTH,'J':roll.hierarchy})
#
# wb.save(exel_path)
# print("exel result saved. {0}".format(exel_path))

beepsound()
    # print(times)
    # m, s = divmod(sum(times.values()), 60)
    # print('Completed after ', sum(times.values()), ' / {0}:{1}'.format(int(m), int(s)))