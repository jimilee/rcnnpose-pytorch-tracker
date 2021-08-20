import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as f
import random, math, time, sys
import path_roll as roll
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tracker_utils import cal_histogram, cos_sim, euclid_sim, ext_ovl_sim, print_tracking_result, centroid_box, \
    convert_img_tensor, ext_dist_sim, ext_atan2_sim, atan2, get_center_pt, center_pt_2_bbox
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from PIL import Image
from simsiam.simsiam_standalone import SimsiamSA
from KalmanFilter import KalmanFilter

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
challenge_path = 'MOT16-11.txt'
class tracker():
    def __init__(self):
        print('init...')
        self.model = SimsiamSA()
        self.trackers = {}
        self.online_trk = []
        self.last_id, self.max_tracker=0,70
        self.minmax_scaler = MinMaxScaler()
        self.stand_scaler = StandardScaler()

    def init_id_tracker(self, max_tracker, T_seq):
        self.T1, self.T2, self.T3 = roll.T[T_seq]
        self.last_id = 0
        for i in range(0, max_tracker):
            color = []
            for c in range(3): color.append(random.randrange(0, 256))
            KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
            self.trackers[i] = {'id': i,
                                'stat': False,
                                'feat': 0,
                                'frame': 0,
                                'hist': 0,
                                #'atan': 0,
                                'rgb': color,
                                'occ': False,
                                'KF': KF}


    def bbox_sim_score(self, target):
        dist, euclid, degree = [], [], []
        target_trk = self.online_trk
        len_trk = len(target_trk)
        dist_score = np.ones(shape=(len_trk,), dtype=np.float32)
        euclid_score = np.ones(shape=(len_trk,), dtype=np.float32)
        degree_score = np.ones(shape=(len_trk,), dtype=np.float32)

        for j, trk in enumerate(target_trk):
            # print(target)
            dist.append(ext_dist_sim(target['box'], trk['box']))
            euclid.append(euclid_sim(target['hist'], trk['hist']))
            #degree.append(ext_atan2_sim(target['box'], trk['box'], trk['atan']))

        if (len(dist) > 0):
            dist = np.array(dist).reshape(len_trk, 1)
            euclid = np.array(euclid).reshape(len_trk, 1)
            #degree = np.array(degree).reshape(len_trk, 1)

            self.minmax_scaler.fit(euclid)
            euclid_score = self.minmax_scaler.transform(euclid)
            euclid_score = 1 - euclid_score

            #self.minmax_scaler.fit(degree)
            #degree_score = self.minmax_scaler.transform(degree)
            #degree_score = 1 - degree_score

        for j, trk in enumerate(target_trk):
            dist_score[j] = (1 - dist[j])

        # print('dist_score', dist_score)
        # print('euclid_score', euclid_score)
        # print('sim_result', sim_result)  #, sim_result
        return dist_score, euclid_score, degree_score


    # 트래커 초기화 및 업데이트.
    # target_ : 업데이트 시킬 타겟,
    # num_trk : id 할당 우선순위용.
    # t_id, update : 트래커 업데이트 시, 사용.
    def tracker_provider(self, target_, t_id=None, update=False):
        if self.last_id >= self.max_tracker - 1:
            self.last_id = 0
        # update tracker
        if update and t_id is not None:
            rgb = self.trackers[t_id]['rgb']
            #tracker_atan = atan2(target_['box'], self.trackers[t_id]['box'])

            tracker_KF = self.trackers[t_id]['KF']
            tracker_cp = get_center_pt(self.trackers[t_id]['box'])
            target_cp = get_center_pt(target_['box'])
            if not self.occluded_tracker(target_['box'], ovl_th=roll.ovlTH, t_id = t_id):
                tracker_feat = target_['feat']
                tracker_hist = target_['hist']
                #tracker_bbox = centroid_box(target_['box'], self.trackers[t_id]['box'])
                (x, y) = tracker_KF.predict()
                (x1, y1) = tracker_KF.update(target_cp)
                tracker_bbox = target_['box']
                trk_occ = False

            else:
                tracker_feat = self.trackers[t_id]['feat']
                tracker_hist = self.trackers[t_id]['hist']
                #tracker_bbox = centroid_box(target_['box'], self.trackers[t_id]['box'])
                #tracker_bbox = target_['box']
                (x, y) = tracker_KF.predict()
                (x1, y1) = tracker_KF.update(tracker_cp)
                if not self.T1 == -1: #static
                    tracker_bbox = center_pt_2_bbox(self.trackers[t_id]['box'],x,y)
                else:
                    tracker_bbox = self.trackers[t_id]['box']
                trk_occ = True
                # print('트래커 업데이트 id :{0}, frame : {1}, cur_frame: {2}'.format(self.trackers[t_id]['id'],
            #                                                            self.trackers[t_id]['frame'], target_['frame']))
            self.trackers[t_id] = {'id': t_id, 'frame': target_['frame'], 'stat': True,
                                   'box': tracker_bbox,
                                   'hist': tracker_hist,
                                   'feat': tracker_feat,
                                   'rgb': rgb,
                                   #'atan': tracker_atan,
                                   'occ' : trk_occ,
                                   'KF' : tracker_KF}
        # generate tracker
        # if d_id != -1:
        #     self.id_table[d_id]['stat'] = False
        elif self.last_id < self.max_tracker: #self.last_id < self.max_tracker and
            for id, state in self.trackers.items():
                rgb = self.trackers[id]['rgb']
                if self.trackers[id]['stat'] is False and id > self.last_id:#
                    self.last_id = id
                    tracker_KF = self.trackers[id]['KF']
                    tracker_cp = get_center_pt(target_['box'])
                    (x, y) = tracker_KF.predict()
                    (x1, y1) = tracker_KF.update(tracker_cp)
                    self.trackers[id] = {'id': id, 'frame': target_['frame'], 'stat': True, # id 할당되면 true.
                                         'box': target_['box'],
                                         'hist': target_['hist'],
                                         'feat': target_['feat'],
                                         'rgb': rgb,
                                         'occ': False,
                                         #'atan': -1,
                                         'KF':tracker_KF}

                    # print(id, '트래커 할당.', self.trackers[id])
                    return int(id)

    def occluded_tracker(self, target, ovl_th=0.5, t_id=-1):
        Ax1, Ay1, Ax2, Ay2 = target
        for tracker in self.online_trk:
            if tracker['stat'] is True and tracker['id']!= t_id:  # and self.id_table[idx]['frame'] == cur_frame
                Bx1, By1, Bx2, By2 = tracker['box']

                # box = (x1, y1, x2, y2)
                box1_area = (Ax2 - Ax1 + 1) * (Ay2 - Ay1 + 1)
                box2_area = (Bx2 - Bx1 + 1) * (By2 - By1 + 1)

                # obtain x1, y1, x2, y2 of the intersection
                Ox1 = max(Ax1, Bx1)
                Oy1 = max(Ay1, By1)
                Ox2 = min(Ax2, Bx2)
                Oy2 = min(Ay2, By2)

                # compute the width and height of the intersection
                w = max(0, Ox2 - Ox1 + 1)
                h = max(0, Oy2 - Oy1 + 1)

                inter = w * h
                iou = inter / (box1_area + box2_area - inter)
                #if iou > 0.8: continue #매칭 될 타겟으로 간주.
                ovl_score = iou

                if ovl_score > ovl_th and ((Ay2 - Ay1) < (By2 - By1) or Ay2 < By2): #
                    return True #많이 겹치면 True

        return False
    #트래커 제거자.
    def tracker_eliminator(self, cur_frame):
        age_TH = 10
        for idx, tracker in self.trackers.items():
            if self.trackers[idx]['stat'] is True:
                if int(cur_frame - self.trackers[idx]['frame']) > age_TH:
                    self.trackers[idx]['stat'] = False

    def get_score_matrix(self, data_dets, score_matrix):
        target_trk = [state['feat'].unsqueeze(0) for state in self.online_trk]
        len_trk = len(target_trk)
        simscore = []
        if (len_trk > 0):
            trackers = torch.cat(target_trk, dim=0).cuda(non_blocking=True)
            detectors = torch.cat([feat['feat'].unsqueeze(0) for feat in data_dets], dim=0).cuda(non_blocking=True)
            # trackers = torch.cat(target_trk, dim=0)
            # detectors = torch.cat([feat['feat'].unsqueeze(0) for feat in data_dets], dim=0)
            # print(trackers.shape, detectors.shape)
            ass_mat = self.model.get_association_matrix(self.model.backbone(), trackers,
                                                        detectors, k=len_trk)

            indicies = ass_mat['indicies']
            scores = ass_mat['scores']
            for ind, s in zip(indicies, scores):
                simsiam_score = np.zeros(shape=(len_trk,), dtype=np.float32)
                if(len_trk==1):
                    x = torch.stack([ind.squeeze().float(),s.squeeze().float()], dim=0).tolist()
                    simsiam_score[int(x[0])] = 0.5
                else:
                    x = torch.stack([ind.squeeze().float(), s.squeeze().float()],
                                    dim=1).tolist()
                    y = np.array(x)[:, 1]
                    y -= y.min()  # bring the lower range to 0
                    y /= y.max()  # bring the upper range to 1
                    # x = torch.stack([ind.squeeze().float(), torch.Tensor(y).cuda()],
                    #                 dim=1).tolist()
                    x = torch.stack([ind.squeeze().float(), torch.Tensor(y)],
                                    dim=1).tolist()
                    for i, score in x:
                        simsiam_score[int(i)] = score
                simscore.append(simsiam_score)
            for i, sim in enumerate(simscore): #i는 디텍션
                for j, trk in enumerate(self.online_trk):
                    if score_matrix[i][trk['id']] < roll.hierarchy:
                        score_matrix[i][trk['id']] = 1 - (((1 - score_matrix[i][trk['id']])* roll.SC1) + (sim[j] * roll.SC2))
                    else:
                        score_matrix[i][trk['id']] = 10.0
                        # score_matrix[i][trk['id']] = 1 - sim[j]
        return score_matrix

                    # try:
                    # if dist_score[j] > 0.6 :
                    # score_matrix[i][trk['id']] += (simscore[i][j] * 0.3)


    def tracking(self, det_boxes, image, frame_cnt):
        target_det = []
        src = image.copy()
        matrix_size = len(det_boxes) if len(self.trackers) < len(det_boxes) else len(self.trackers)
        score_matrix = np.full((matrix_size + 1, matrix_size + 1), 10.0, dtype=float)
        # print(
        #     '\n Start frame : {0} ===================================================================================='.format(
        #         frame_cnt))
        for i, det in enumerate(det_boxes):
            x1, y1 = det[:2]
            x2, y2 = det[2:]

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x1 > src.shape[1]: x1 = src.shape[1]-1
            if y1 > src.shape[0]: y1 = src.shape[0]-1

            if x2 < 0: x2 = 0
            if y2 < 0: y2 = 0
            if x2 > src.shape[1]: x2 = src.shape[1]-1
            if y2 > src.shape[0]: y2 = src.shape[0]-1
            if x1 == x2: continue
            elif y1 == y2: continue

            det = np.array([x1,y1,x2,y2], dtype=int)

            #if (x2-x1) < 30 or (y2-y1) < 30:
            #    continue
            # feat = src.crop((max(x1, 0), max(y1, 0), min(x2, src.size[0]), min(y2, src.size[1])))
            feat = src[y1:y2, x1:x2]
            roi_hsv = cv2.cvtColor(feat, cv2.COLOR_BGR2HSV)
            tensor_src = convert_img_tensor(feat)

            # HSV 히스토그램 계산
            channels = [0, 1]
            ranges = [0, 180, 0, 256]
            det_hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

            # print(len(det_boxes))
            det_data = {'id': -1,
                        'frame': frame_cnt,
                        'box': det,
                        'hist': det_hist,
                        'feat': tensor_src}

            if frame_cnt == 0:
                det_data['id'] = self.tracker_provider(det_data)  # 트래커 생성
                continue

            dist_score, euclid_score, degree_score = self.bbox_sim_score(det_data)
            # # sim_score = self.simsiam_sim_score(tensor_src)
            # # print(len(dist_score), len(sim_score), len(self.online_trk))
            for j, trk in enumerate(self.online_trk):
                # try:
                #if dist_score[j] > 0.6 :
                score_matrix[i][trk['id']] = 1 - ((dist_score[j] * self.T2) + (euclid_score[j] * self.T3))

                # except:
                #     print(j,i ,'is passed.')
                #     pass
            target_det.append(det_data)
        #simsiam
        if len(target_det)>0:
            score_matrix = self.get_score_matrix(target_det, score_matrix)

        if frame_cnt == 0:
            self.online_trk = [state for id, state in self.trackers.items() if state['stat'] is True]
            return self.online_trk

        score_matrix[np.isnan(score_matrix)] = 10.0
        # print(score_matrix)
        row_ind, col_ind = linear_sum_assignment(score_matrix)  # hungarian.
        hungarian_result = col_ind[:len(target_det)]
        # print(row_ind)
        # print(col_ind)
        # print(hungarian_result)
        # id -> idx 로 저장해야됨.
        for idx, id in enumerate(hungarian_result):  # id_update.
            if idx < len(target_det) and id < self.max_tracker:
                if self.trackers[id]['stat'] == True and score_matrix[idx][id] < roll.updateTH:  # and score_matrix[idx][id] < 0.5  self.trackers[id]['frame'] >= frame_cnt-1
                    # print('업데이트 타겟. id : {0} -> idx: {1}'.format(self.trackers[id]['id'], idx))
                    self.tracker_provider(target_=target_det[idx],
                                          t_id=self.trackers[id]['id'], update=True) # 트래커 업데이트
                    target_det[idx]['id'] = self.trackers[id]['id']

                if target_det[idx]['id'] == -1 and not self.occluded_tracker(target=target_det[idx]['box'], ovl_th= 0.5):# 타겟 아이디가 -1 일때.(아직 할당 x, 새로 생긴 객체)
                    target_det[idx]['id'] = self.tracker_provider(target_det[idx])  # 트래커 생성
                    # print('트래커 생성. id:', target_det[idx]['id'] ,target_det[idx]['id'] == -1,
                    #       self.occluded_tracker(target=target_det[idx]['box'], ovl_th=0.3))

        self.tracker_eliminator(frame_cnt)
        self.online_trk = [state for id, state in self.trackers.items() if state['stat'] is True] # and (frame_cnt - state['frame']) < 2
        # 결과값 저장.
        # print_tracking_result(self.online_trk, challenge_path, frame_cnt)
        return self.online_trk



