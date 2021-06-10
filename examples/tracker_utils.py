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
    Acx = Ax1 + ((Ax2 - Ax1) / 2)
    Acy = Ay1 + ((Ay2 - Ay1) / 2)
    Bcx = Bx1 + ((Bx2 - Bx1) / 2)
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
    result = float(((1 - math.sqrt(math.pow(a_center_pt[0] - b_center_pt[0], 2)
                                   + math.pow(a_center_pt[1] - b_center_pt[1], 2)))
                    + ovl_score ) /2)
    return result