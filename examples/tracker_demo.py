import numpy as np
import cv2
import random, math, time, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class simple_tracker(object):
    def __init__(self):
        self.trackers = {}
        self.last_id = 0
        self.color_table = []
        self.max_tracker = 5
        self.minmax_scaler = MinMaxScaler()
        self.stand_scaler = StandardScaler()

    def init_id_tracker(self):
        self.last_id = 0
        for i in range(0, self.max_tracker):
            self.trackers[i] = {'id': i, 'stat': False, 'feat': 0, 'frame': 0, 'hist': 0}

    def init_color_table(self):
        for i in range(0, self.max_tracker):
            color = []
            for c in range(3): color.append(random.randrange(0, 256))
            self.color_table.append(color)

    def cal_histogram(self,img):
        return cv2.calcHist([img],[0],None,[256],[0,256])

    def cos_sim(self, A, B):
        return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    def euclid_sim(self, A, B):
        return np.sqrt(np.sum((A-B)**2))

    def bbox_sim_score(self, targets, edges):
        dist, euclid, hist, ovl = [], [], [], []
        dist_score = np.ones(shape=(len(edges[0]),), dtype=np.float32)
        euclid_score = np.ones(shape=(len(edges[0]),), dtype=np.float32)
        hist_score = np.ones(shape=(len(edges[0]),), dtype=np.float32)
        overlap_score = np.ones(shape=(len(edges[0]),), dtype=np.float32)
        for a, b, k in zip(edges[0], edges[1], range(0, len(edges[0]))):
            point_com_sxy = [min(targets[a]['sx'], targets[b]['sx']), min(targets[a]['sy'], targets[b]['sy'])]
            point_com_bxy = [max(targets[a]['bx'], targets[b]['bx']), max(targets[a]['by'], targets[b]['by'])]
            point_ovl_sxy = [max(targets[a]['sx'], targets[b]['sx']), max(targets[a]['sy'], targets[b]['sy'])]
            point_ovl_bxy = [min(targets[a]['bx'], targets[b]['bx']), min(targets[a]['by'], targets[b]['by'])]

            comb_area = (point_com_bxy[0] - point_com_sxy[0]) * (point_com_bxy[1] - point_com_sxy[1])
            ovl_area = max(point_ovl_bxy[0] - point_ovl_sxy[0], 0) * max(point_ovl_bxy[1] - point_ovl_sxy[1], 0)
            ovl_score = ovl_area / comb_area

            a_center_pt = [(targets[a]['sx'] + targets[a]['bx']) / 2, (targets[a]['sy'] + targets[a]['by']) / 2]
            b_center_pt = [(targets[b]['sx'] + targets[b]['bx']) / 2, (targets[b]['sy'] + targets[b]['by']) / 2]
            dist.append(math.sqrt(
                math.pow(a_center_pt[0] - b_center_pt[0], 2) + math.pow(a_center_pt[1] - b_center_pt[1], 2)))
            # euclid.append(self.euclid_sim(np.hstack([targets[a]['keypoints'],targets[a]['feat']]),
            #                               np.hstack([targets[b]['keypoints'],targets[b]['feat']])))
            euclid.append(self.euclid_sim(targets[a]['feat'], targets[b]['feat']))
            hist.append((cv2.compareHist(targets[a]['hist'], targets[b]['hist'], cv2.HISTCMP_INTERSECT)) / np.sum(
                targets[a]['hist']))
            ovl.append(ovl_score)

        if (len(dist) > 0):
            dist = np.array(dist).reshape(len(dist), 1)
            euclid = np.array(euclid).reshape(len(dist), 1)
            # cosin = np.array(cosin).reshape(len(dist),1)
            self.minmax_scaler.fit(dist)
            self.stand_scaler.fit(euclid)
            result = self.minmax_scaler.transform(dist)
            eu_result = self.stand_scaler.transform(euclid)
            # self.stand_scaler.fit(cosin)
            # co_result = self.stand_scaler.transform(cosin)
        cnt = 0
        for a, b, k in zip(edges[0], edges[1], range(0, len(edges[0]))):
            overlap_score[k] = ovl[cnt]
            dist_score[k] = 1 - result[cnt]
            euclid_score[k] = 1 - eu_result[cnt]
            hist_score[k] = hist[cnt]
            cnt += 1
        return dist_score, euclid_score, overlap_score, hist_score
    # 트래커 초기화 및 업데이트.
    # target_ : 업데이트 시킬 타겟,
    # num_trk : id 할당 우선순위용.
    # t_id, update : 트래커 업데이트 시, 사용.
    def tracker_provider(self, target_, t_id=None, update=False):
        if self.last_id == self.max_tracker - 1:
            self.last_id = 0

        if update and t_id is not None:
            if not self.occluded_tracker(t_id):
                tracker_feat = target_['feat']
                tracker_hist = target_['hist']

            else:
                tracker_feat = self.trackers[t_id]['feat']
                tracker_hist = self.trackers[t_id]['hist']

            self.trackers[t_id] = {'id': t_id, 'frame': target_['frame'], 'stat': True,
                                   'box': target_['box'],
                                   'hist': tracker_hist,
                                   'feat': tracker_feat}

        # if d_id != -1:
        #     self.id_table[d_id]['stat'] = False
        else:
            for id, state in self.trackers.items():
                if self.trackers[id]['stat'] is False and id > self.last_id:
                    self.last_id = id

                    self.trackers[id] = {'id': id, 'frame': target_['frame'], 'stat': True,
                                         'box': target_['box'],
                                         'hist': target_['hist'],
                                         'feat': target_['feat']}  # id 할당되면 true.

                    return int(id)
    def tracking(self, det_boxes, image, frame_cnt):
        final_targets = []
        target_det = []
        src = image.copy()

        matrix_size = len(det_boxes) if len(self.trackers) < len(det_boxes) else len(self.trackers)
        score_matrix = np.full((matrix_size + 1, matrix_size + 1), 1.0, dtype=float)
        for i, det in enumerate(det_boxes):
            x1, y1 = det[:2]
            x2, y2 = det[2:]
            feat = src.crop((max(x1, 0), max(y1, 0), min(x2, src.size[0]), min(y2, src.size[1])))

            det_hist = self.cal_histogram(feat)
            print(det_hist)
            det_data = {'frame': frame_cnt,
                        'box': det,
                        'hist': det_hist,
                        'feat': det_hist}

            for j, trk in enumerate(self.trackers):
                score = self.euclid_sim(trk['feat'], det_data['feat'])
                score_matrix[i][j] = score




        final_targets.clear()
        for n, target in enumerate(target_det):
            # print('최종 타겟들 추가하기.',len(target_frame), num_trackers)
            # print(n, ' 타겟들. target_frames : {0}, id: {1}'.format(target['frame'],
            #                                                                 target['id']))
            if n >= num_trackers:
                if target['id'] == -1 or target['id'] == None:
                    target['id'] = self.tracker_provider(target)  # 트래커 생성
                if target['id'] != self.ghost_id:
                    # print(n, ' 추가된 디텍션들. final_target_frames : {0}, id: {1}'.format(target['frame'],
                    #                                                                 target['id']))
                    final_targets.append(target)


