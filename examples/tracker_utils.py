import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as f
import random, math, time, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from PIL import Image
from simsiam.simsiam_standalone import SimsiamSA

def cal_histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def cos_sim(A, B):
    return np.dot(A, B) /(np.linalg.norm(A) *np.linalg.norm(B))

def euclid_sim(A, B):
    return np.sqrt(np.sum((A-B)**2))

def dist_sim(A, B): # (ovl_score + dist_score) / 2
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B

    # combine 2 box = (x1, y1, x2, y2)
    Cx1, Cy1, Cx2, Cy2 = min(Ax1,Bx1), min(Ay1, By1), max(Ax2, Bx2), max(Ay2,By2)

    # obtain x1, y1, x2, y2 of the intersection
    Ox1 = max(Ax1, Bx1)
    Oy1 = max(Ay1, By1)
    Ox2 = min(Ax2, Bx2)
    Oy2 = min(Ay2, By2)

    # compute the width and height of the intersection
    w = max(0, Ox2 - Ox1 + 1)
    h = max(0, Oy2 - Oy1 + 1)

    inter = w * h
    if inter == 0:
        return float(0.0)
    iou = inter / ((Cx2-Cx1) * (Cy2 - Cy1))
    ovl_score = iou


    Acx = Ax1 + ((Ax2 - Ax1) / 2)
    Acy = Ay1 + ((Ay2 - Ay1) / 2)
    Bcx = Bx1 + ((Bx2 - Bx1) / 2)
    Bcy = By1 + ((By2 - By1) / 2)

    a_center_pt = [Acx, Acy]
    b_center_pt = [Bcx, Bcy]

    if (a_center_pt < [Bx2,By2] and a_center_pt > [Bx1,By1]) \
            or (b_center_pt < [Ax2,Ay2] and b_center_pt > [Ax1,Ay1]):
        return float((ovl_score * 1.0))

    return 0.0

# 챌린지 출력파일 저장.
def print_tracking_result(data, path, this_frame):
    f = open(path, 'a')
    # print('this_frame : ', this_frame)
    for p_data in data:
        x1, y1, x2, y2 = p_data['box']
        # if not self.occluded_tracker(this_frame, p_data['id'], 0.5):
        p_data_width = int(x2 - x1)
        p_data_height = int(y2 - y1)
        # print("{0},{1},{2},{3},{4},{5},-1,-1,-1,-1".format(p_data['frame'], p_data['id'], p_data['sx'], p_data['sy'], p_data_width, p_data_height))
        if this_frame <= p_data['frame'] + 2:
            f.write("{0}, {1}, {2}, {3}, {4}, {5}, -1, -1, -1, -1\n".format(this_frame, p_data['id'], x1,
                                                                            y1, p_data_width,
                                                                            p_data_height))

    f.close()