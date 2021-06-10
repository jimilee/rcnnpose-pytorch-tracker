import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as f
import random, math, time, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tracker_utils import cal_histogram, cos_sim, euclid_sim, dist_sim
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from PIL import Image
from simsiam.simsiam_standalone import SimsiamSA

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

class simple_tracker():
    def __init__(self):
        self.simsiam = SimsiamSA()
        self.trackers = {}
        self.online_trk = []
        self.init_id_tracker()
        self.last_id, self.max_tracker=0,10
        self.minmax_scaler = MinMaxScaler()
        self.stand_scaler = StandardScaler()

    def init_id_tracker(self):
        for i in range(0, 10):
            color = []
            for c in range(3): color.append(random.randrange(0, 256))
            self.trackers[i] = {'id': i, 'stat': False, 'feat': 0, 'frame': 0, 'hist': 0, 'rgb': color}


    def bbox_sim_score(self, target):
        dist, euclid, hist, ovl = [], [], [], []
        target_trk = self.online_trk
        len_trk = len(target_trk)
        dist_score = np.ones(shape=(len_trk,), dtype=np.float32)
        euclid_score = np.ones(shape=(len_trk,), dtype=np.float32)

        for j, trk in enumerate(target_trk):
            # print(target)
            dist.append(dist_sim(target['box'], trk['box']))
            euclid.append(euclid_sim(target['hist'], trk['hist']))

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
            # print('트래커 업데이트 id :{0}, frame : {1}, cur_frame: {2}'.format(self.trackers[t_id]['id'],
            #                                                            self.trackers[t_id]['frame'], target_['frame']))
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

    def occluded_tracker(self, target, ovl_th=0.5):
        target_sx, target_bx, target_sy, target_by = target['box']

        for idx, tracker in self.trackers.items():
            if self.trackers[idx]['stat'] is True:  # and self.id_table[idx]['frame'] == cur_frame
                Bx1, By1, Bx2, By2 = self.trackers[idx]['box']
                point_ovl_sxy = [max(target_sx, Bx1), max(target_sy, By1)]
                point_ovl_bxy = [min(target_bx, Bx2), min(target_by, By2)]

                target_area = (target_bx - target_sx) * (target_by - target_sy)
                if target_area == 0: continue

                ovl_area = max(point_ovl_bxy[0] - point_ovl_sxy[0], 0) * max(point_ovl_bxy[1] - point_ovl_sxy[1], 0)
                ovl_score = 0
                try:
                    ovl_score = ovl_area / target_area
                except:
                    print(target_area)

                if ovl_score > ovl_th and target_by - target_sy < (By2 - By1):
                    return True
    #트래커 제거자.
    def tracker_eliminator(self, cur_frame):
        age_TH = 3
        for idx, tracker in self.trackers.items():
            if self.trackers[idx]['stat'] is True:
                # if self.occluded_tracker(self.trackers[idx]):
                #     age_TH = 5
                if int(cur_frame - self.trackers[idx]['frame']) > age_TH:
                    #
                    # print('트래커 제거 id :{0}, frame : {1}, cur_frame: {2}'.format(
                    #                                                               self.trackers[idx]['id'],
                    #                                                               self.trackers[idx]['frame'],
                    #                                                               cur_frame))
                    self.trackers[idx]['stat'] = False

    def convert_img_tensor(self, src):
        color_cvt = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pil_src = Image.fromarray(color_cvt)
        trans = transforms.Compose([transforms.Resize((224,112)),
                                  transforms.ToTensor()])
        trans_target = trans(pil_src)
        # print(trans_target.shape)
        return trans_target

    def simsiam_sim_score(self, target):
        sim_score = []
        target_trk = [state['feat'].unsqueeze(0) for id, state in self.trackers.items() if state['stat'] is True]
        len_trk = len(target_trk)
        if(len_trk > 0):
            trackers = torch.cat(target_trk, dim=0).cuda(non_blocking=True)
            ass_mat = self.simsiam.get_association_matrix(self.simsiam.backbone(), trackers, target.unsqueeze(0).cuda(non_blocking=True), k=len_trk)
            #score 노말라이즈.
            # print(type(ass_mat['indicies']), type(ass_mat['scores']))
            # print(ass_mat['indicies'].squeeze().shape, ass_mat['scores'].squeeze().shape)
            # print(torch.stack([ass_mat['indicies'].squeeze().float(),ass_mat['scores'].squeeze().float()], dim=1))
            x = torch.stack([ass_mat['indicies'].squeeze().float(),ass_mat['scores'].squeeze().float()], dim=1).tolist()
            s = sorted(x, key=lambda x: x[0])

            x = np.array(s)[:, 1]
            x -= x.min()  # bring the lower range to 0
            x /= x.max()  # bring the upper range to 1
            # sim_score = f.normalize(ass_mat['scores'], dim=1).squeeze().tolist()
            sim_score = np.array(x,dtype=np.float32)
            sim_score = [0 if i < 0.7 else i for i in sim_score] # 일정 임계값 이하는 컷

        return sim_score

    def tracking(self, det_boxes, image, frame_cnt):
        target_det = []
        src = image.copy()
        matrix_size = len(det_boxes) if len(self.trackers) < len(det_boxes) else len(self.trackers)
        score_matrix = np.full((matrix_size + 1, matrix_size + 1), 10.0, dtype=float)
        for i, det in enumerate(det_boxes):
            x1, y1 = det[:2]
            x2, y2 = det[2:]

            if (x2-x1) < 8 or (y2-y1) < 8:
                continue
            # feat = src.crop((max(x1, 0), max(y1, 0), min(x2, src.size[0]), min(y2, src.size[1])))
            feat = src[y1:y2, x1:x2]
            roi_hsv = cv2.cvtColor(feat, cv2.COLOR_BGR2HSV)
            tensor_src = self.convert_img_tensor(feat)

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


            dist_score, euclid_score = self.bbox_sim_score(det_data)
            sim_score = self.simsiam_sim_score(tensor_src)
            # print(len(dist_score), len(sim_score), len(self.online_trk))
            for j, trk in enumerate(self.online_trk):
                # try:
                score_matrix[j][i] = 1 - ((dist_score[j] * 0.5) + (sim_score[j] * 0.5))
                # except:
                #     print(j,i ,'is passed.')
                #     pass
            target_det.append(det_data)

        # print('=====================================================================================')

        if frame_cnt == 0:
            self.online_trk = [state for id, state in self.trackers.items() if state['stat'] is True]
            return self.online_trk

        score_matrix[np.isnan(score_matrix)] = 10.0

        # print(score_matrix)
        row_ind, col_ind = linear_sum_assignment(score_matrix)  # hungarian.
        hungarian_result = col_ind[:self.max_tracker]

        # id -> idx 로 저장해야됨.
        for id, idx in enumerate(hungarian_result):  # id_update.
            # print('업데이트 타겟. id : {0} -> idx: {1}'.format(self.trackers[id]['id'], idx))
            if idx < len(target_det):
                if score_matrix[idx][id] < 0.6:  # and score_matrix[idx][id] < 0.5  self.trackers[id]['frame'] >= frame_cnt-1
                    self.tracker_provider(target_=target_det[idx],
                                          t_id=self.trackers[id]['id'], update=True) # 트래커 업데이트
                    target_det[idx]['id'] = self.trackers[id]['id']

                if target_det[idx]['id'] == -1 and not self.occluded_tracker(target=target_det[idx]):# 타겟 아이디가 -1 일때.(아직 할당 x, 새로 생긴 객체)
                    target_det[idx]['id'] = self.tracker_provider(target_det[idx])  # 트래커 생성

        self.tracker_eliminator(frame_cnt)
        self.online_trk = [state for id, state in self.trackers.items() if state['stat'] is True]
        return self.online_trk



