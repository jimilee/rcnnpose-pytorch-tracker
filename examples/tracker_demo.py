import numpy as np
import cv2
import random, math, time, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.optimize import linear_sum_assignment

class simple_tracker():
    def __init__(self):
        self.trackers = {}
        self.init_id_tracker()
        self.last_id, self.max_tracker=0,5
        self.minmax_scaler = MinMaxScaler()
        self.stand_scaler = StandardScaler()

    def init_id_tracker(self):
        for i in range(0, 5):
            color = []
            for c in range(3): color.append(random.randrange(0, 256))
            self.trackers[i] = {'id': i, 'stat': False, 'feat': 0, 'frame': 0, 'hist': 0, 'rgb': color}

    def cal_histogram(self,img):
        return cv2.calcHist([img],[0],None,[256],[0,256])

    def cos_sim(self, A, B):
        return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

    def euclid_sim(self, A, B):
        return np.sqrt(np.sum((A-B)**2))

    def dist_sim(self, A, B):
        Ax1, Ay1, Ax2, Ay2 = A
        Bx1, By1, Bx2, By2 = B
        Acx = Ax1 + ((Ax2-Ax1) / 2)
        Acy = Ay1 + ((Ay2 - Ay1) / 2)
        Bcx = Bx1 + ((Bx2-Bx1) / 2)
        Bcy = By1 + ((By2 - By1) / 2)
        point_com_sxy = [min(Ax1, Bx1), min(Ay1, By1)]
        point_com_bxy = [max(Ax2, Bx2), max(Ay2, By2)]
        point_ovl_sxy = [max(Ax1, Bx1), max(Ay1, By1)]
        point_ovl_bxy = [min(Ax2, Bx2), min(Ay2, By2)]

        a_center_pt = [Acx, Acy]
        b_center_pt = [Bcx, Bcy]

        comb_area = (point_com_bxy[0] - point_com_sxy[0]) * (point_com_bxy[1] - point_com_sxy[1])
        ovl_area = max(point_ovl_bxy[0] - point_ovl_sxy[0], 0) * max(point_ovl_bxy[1] - point_ovl_sxy[1], 0)
        ovl_score = ovl_area / comb_area
        result = float((
                               (1 - math.sqrt(math.pow(a_center_pt[0] - b_center_pt[0], 2) + math.pow(a_center_pt[1] - b_center_pt[1], 2)))
                               + ovl_score)/2)
        return result

    def bbox_sim_score(self, target):
        dist, euclid, hist, ovl = [], [], [], []
        target_trk = [state for id, state in self.trackers.items() if state['stat'] is True]
        len_trk = len(target_trk)
        dist_score = np.ones(shape=(len_trk,), dtype=np.float32)
        euclid_score = np.ones(shape=(len_trk,), dtype=np.float32)

        for j, trk in enumerate(target_trk):
            # print(target)
            dist.append(self.dist_sim(target['box'], trk['box']))
            euclid.append(self.euclid_sim(target['feat'], trk['feat']))

        if (len(dist) > 0):
            dist = np.array(dist).reshape(len(dist), 1)
            euclid = np.array(euclid).reshape(len(dist), 1)
            self.minmax_scaler.fit(dist)
            self.stand_scaler.fit(euclid)
            result = self.minmax_scaler.transform(dist)
            eu_result = self.stand_scaler.transform(euclid)

        for j, trk in enumerate(target_trk):
            dist_score[j] = result[j]
            euclid_score[j] = 1 - eu_result[j]
        return dist_score, euclid_score


    # 트래커 초기화 및 업데이트.
    # target_ : 업데이트 시킬 타겟,
    # num_trk : id 할당 우선순위용.
    # t_id, update : 트래커 업데이트 시, 사용.
    def tracker_provider(self, target_, t_id=None, update=False):
        print(self.last_id, self.max_tracker)
        if self.last_id >= self.max_tracker - 1:
            self.last_id = 0

        if update and t_id is not None:
            rgb = self.trackers[t_id]['rgb']
            if not self.occluded_tracker(target_):
                tracker_feat = target_['feat']
                tracker_hist = target_['hist']

            else:
                tracker_feat = self.trackers[t_id]['feat']
                tracker_hist = self.trackers[t_id]['hist']

            self.trackers[t_id] = {'id': t_id, 'frame': target_['frame'], 'stat': True,
                                   'box': target_['box'],
                                   'hist': tracker_hist,
                                   'feat': tracker_feat,
                                   'rgb': rgb}

        # if d_id != -1:
        #     self.id_table[d_id]['stat'] = False
        elif self.last_id < self.max_tracker:
            for id, state in self.trackers.items():
                rgb = self.trackers[id]['rgb']
                if self.trackers[id]['stat'] is False and id >= self.last_id:
                    self.last_id = id

                    self.trackers[id] = {'id': id, 'frame': target_['frame'], 'stat': True,
                                         'box': target_['box'],
                                         'hist': target_['hist'],
                                         'feat': target_['feat'],
                                         'rgb': rgb}  # id 할당되면 true.

                    # print(id, '트래커 할당.', self.trackers[id])
                    return int(id)

    def occluded_tracker(self, target, ovl_th = 0.5):
        # print(target)
        target_sx, target_bx, target_sy, target_by = target['box']

        for idx, tracker in self.trackers.items():
            if self.trackers[idx]['stat'] is True: # and self.id_table[idx]['frame'] == cur_frame
                Bx1, By1, Bx2, By2 = self.trackers[idx]['box']
                point_ovl_sxy = [max(target_sx, Bx1), max(target_sy, By1)]
                point_ovl_bxy = [min(target_bx, Bx2), min(target_by, By2)]

                target_area = (target_bx - target_sx) * (target_by - target_sy)
                ovl_area = max(point_ovl_bxy[0] - point_ovl_sxy[0], 0) * max(point_ovl_bxy[1] - point_ovl_sxy[1], 0)
                ovl_score = ovl_area / target_area

                if ovl_score > ovl_th and target_by-target_sy < (By2 - By1):
                    return True
    #트래커 제거자.
    def tracker_eliminator(self, cur_frame):
        age_TH = 3
        for idx, tracker in self.trackers.items():
            if self.trackers[idx]['stat'] is True:
                # if self.occluded_tracker(idx):
                #     age_TH = 15
                if int(cur_frame - tracker['frame']) > age_TH:
                    # del self.id_table[idx]
                    print('트래커 삭제', self.trackers[idx]['id'], cur_frame - tracker['frame'])
                    self.trackers[idx]['stat'] = False

    def tracking(self, det_boxes, image, frame_cnt):
        final_targets = []
        target_det = []
        src = image.copy()
        matrix_size = len(det_boxes) if len(self.trackers) < len(det_boxes) else len(self.trackers)
        score_matrix = np.full((matrix_size + 1, matrix_size + 1), 1.0, dtype=float)
        for i, det in enumerate(det_boxes):
            x1, y1 = det[:2]
            x2, y2 = det[2:]

            if (x2-x1) < 8 or (y2-y1) < 8:
                continue
            # feat = src.crop((max(x1, 0), max(y1, 0), min(x2, src.size[0]), min(y2, src.size[1])))
            feat = src[y1:y2, x1:x2]
            roi_hsv = cv2.cvtColor(feat, cv2.COLOR_BGR2HSV)
            # HS 히스토그램 계산
            channels = [0, 1]
            ranges = [0, 180, 0, 256]
            det_hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

            # print(len(det_boxes))
            det_data = {'id': -1,
                        'frame': frame_cnt,
                        'box': det,
                        'hist': det_hist,
                        'feat': det_hist}

            if frame_cnt == 0:
                det_data['id'] = self.tracker_provider(det_data)  # 트래커 생성
                continue


            dist_score, euclid_score = self.bbox_sim_score(det_data)


            for j, trk in self.trackers.items():
                if self.trackers[j]['stat']:
                    try:
                        score_matrix[j][i] = 1 -((dist_score[j] * 0.5) + ( euclid_score[j]* 0.5))  #(self.euclid_sim(det_data['feat'], self.trackers[j]['feat']))*0.0 +
                    except:
                        pass
                        # print(j)
            target_det.append(det_data)

        # print('=====================================================================================')

        if frame_cnt == 0:
            final_targets = [state for id, state in self.trackers.items() if state['stat'] is True]
            return final_targets

        score_matrix[np.isnan(score_matrix)] = 1.0
        # print(score_matrix)
        row_ind, col_ind = linear_sum_assignment(score_matrix)  # hungarian.
        # print(row_ind, col_ind)
        hungarian_result = col_ind[:self.max_tracker]
        # print(hungarian_result)
        # id -> idx 로 저장해야됨.
        # print(len(target_det))
        for id, idx in enumerate(hungarian_result):  # id_update.
            # print('업데이트 타겟. idx : {0}, id: {1}'.format(idx, self.trackers[id]['id']))
            if idx < len(target_det):
                if  score_matrix[idx][id] < 0.5:  # and score_matrix[idx][id] < 0.5  self.trackers[id]['frame'] >= frame_cnt-1
                    self.tracker_provider(target_=target_det[idx],
                                          t_id=self.trackers[id]['id'], update=True)
                    target_det[idx]['id'] = self.trackers[id]['id']

                if target_det[idx]['id'] == -1 and not self.occluded_tracker(target=target_det[idx]):# 타겟 아이디가 -1 일때.
                    target_det[idx]['id'] = self.tracker_provider(target_det[idx])  # 트래커 생성

        self.tracker_eliminator(frame_cnt)

        final_targets = [state for id, state in self.trackers.items() if state['stat'] is True]

        return final_targets

                # print('idx :',idx,'id : ', target_frame[idx]['id'], ' -->  idx :', id,'id : ', target_frame[id]['id'],'\n')

        # final_targets.clear()
        # for n, target in enumerate(target_det):
        #     # print('최종 타겟들 추가하기.',len(target_frame), num_trackers)
        #     # print(n, ' 타겟들. target_frames : {0}, id: {1}'.format(target['frame'],
        #     #                                                                 target['id']))
        #     if n >= num_trackers:
        #         if target['id'] == -1 or target['id'] == None:
        #             target['id'] = self.tracker_provider(target)  # 트래커 생성
        #         if target['id'] != self.ghost_id:
        #             # print(n, ' 추가된 디텍션들. final_target_frames : {0}, id: {1}'.format(target['frame'],
        #             #                                                                 target['id']))
        #             final_targets.append(target)


