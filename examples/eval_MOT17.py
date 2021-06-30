from collections import deque

import path_roll as roll
import os
import os.path as osp
import cv2
import numpy as np

from examples.tracker_demo import simple_tracker
from examples.tracker_utils import print_tracking_result
from rcnnpose.utils import draw_tracker_boxes
st = simple_tracker()
MOT_DATA = roll.TARGET_DATASET
def track_all_seq(target_='train'):
    for dataset in MOT_DATA:
        print(osp.join(roll.DATA_PATH, dataset, target_))
        for seq in os.listdir(osp.join(roll.DATA_PATH, dataset, target_)):
            st.init_id_tracker(st.max_tracker)
            path = osp.join(roll.PREDATA_PATH, target_, str(seq), str(seq))
            challenge_path = osp.join(roll.CHALLENGE_PATH, seq[9:], 'MOT16-{0}.txt'.format(seq[6:8]))
            file_target = os.path.join(osp.join(roll.PREDATA_PATH, seq + '_det.txt'))
            img_target = osp.join(roll.DATA_PATH, dataset, target_, seq, seq)  # MOT17
            T_seq = seq[6:8]
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
                    if float(score) > 0.65:  # gt.
                        bbox.append((frame, id1, x, y, w, h, score))

            bbox = sorted(bbox, key=lambda x: int(x[0]))  # 프레임순 정렬.
            dq = deque(bbox)
            frame_cnt = 0
            idx = 0
            for img in os.listdir(img_target):
                frame_cnt = frame_cnt + 1
                src = cv2.imread(osp.join(img_target, img))
                det_boxes = []
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

                    else:
                        dq.appendleft(bbox)  # 다음 프레임은 다시 넣음
                        break
                # print(type(np.array(det_boxes)))
                target = st.tracking(np.array(det_boxes), src, frame_cnt)
                # overlay_tk = draw_tracker_boxes(src, target, frame_cnt)
                # cv2.imshow('Video Demo', overlay_tk)
                # if cv2.waitKey(20) & 0xff == 27:  # exit if pressed `ESC`
                #     break
                print_tracking_result(target, challenge_path, frame_cnt)

track_all_seq()
    # print(times)
    # m, s = divmod(sum(times.values()), 60)
    # print('Completed after ', sum(times.values()), ' / {0}:{1}'.format(int(m), int(s)))