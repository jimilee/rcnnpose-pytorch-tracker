import os

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


def convert_img_tensor(src):
    color_cvt = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(color_cvt)
    # trans = transforms.Compose([transforms.Resize((224,112)),
    #                           transforms.ToTensor()])
    # trans = transforms.Compose([transforms.Resize((128,64)),
    #                           transforms.ToTensor()])
    trans = transforms.Compose([transforms.Resize((256, 128)),
                                transforms.ToTensor()])
    trans_target = trans(pil_src)
    return trans_target

def cal_histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def cos_sim(A, B):
    return np.dot(A, B) /(np.linalg.norm(A) *np.linalg.norm(B))

def euclid_sim(A, B):
    return np.sqrt(np.sum((A-B)**2))

def centroid_box(A, B):
    C = (A + B)/2
    return np.array(C, dtype=int)

def bbox_reloc(det, shape):
    x1, y1 = det[:2]
    x2, y2 = det[2:]

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x1 > shape[1]: x1 = shape[1] - 1
    if y1 > shape[0]: y1 = shape[0] - 1

    if x2 < 0: x2 = 0
    if y2 < 0: y2 = 0
    if x2 > shape[1]: x2 = shape[1]
    if y2 > shape[0]: y2 = shape[0]
    if x2 - x1 <= 2:
        return None
    elif y2 - y1 <= 2:
        return None
    return np.array([x1, y1, x2, y2], dtype=int)

def atan2(A,B):
    Ax1, Ay1, Ax2, Ay2 = A #target
    Bx1, By1, Bx2, By2 = B #tracker
    Acx, Acy = Ax1 + ((Ax2 - Ax1) / 2), Ay1 + ((Ay2 - Ay1) / 2)
    Bcx, Bcy = (Bx1 + ((Bx2 - Bx1) / 2)), (By1 + ((By2 - By1) / 2))
    Dx, Dy = 0, 0
    Dx = (Acx - Bcx) #if Acx > Bcx else (Bcx - Acx)
    Dy = (Acy - Bcy) #if Acy > Bcy else (Bcy - Acy)
    return math.atan2(Dy,Dx)

def ext_atan2_sim(A, B, atan):
    if atan == -1: return 0 #?????? ?????????
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B
    Acx, Acy = Ax1+((Ax2-Ax1)/2), Ay1 + ((Ay2- Ay1)/2)
    Bcx, Bcy = (Bx1+((Bx2-Bx1)/2)), (By1 + ((By2- By1)/2))
    Dx,Dy = 0,0
    Dx = (Acx - Bcx) #if Acx > Bcx else (Bcx - Acx)
    Dy = (Acy - Bcy) #if Acy > Bcy else (Bcy - Acy)
    result = math.atan2(Dy,Dx)
    #print(result, atan, 'atan result. : ', abs(result - atan))
    return abs(result - atan)

def get_center_pt(A):
    Ax1, Ay1, Ax2, Ay2 = A
    Acx, Acy = Ax1+((Ax2-Ax1)/2), Ay1 + ((Ay2- Ay1)/2)
    return np.array([[Acx], [Acy]])

def center_pt_2_bbox(A, cx, cy):
    Ax1, Ay1, Ax2, Ay2 = A
    w, h = (Ax2-Ax1),(Ay2-Ay1)
    w = w/2
    h = h/2
    cx = cx.getA1()[0]
    cy = cy.getA1()[0]
    return np.array([cx-w, cy-h, cx+w, cy+h], dtype=int)

def ext_dist_sim(A, B):
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B

    Acx, Acy = Ax1+((Ax2-Ax1)/2), Ay1 + ((Ay2- Ay1)/2)
    Bcx, Bcy = Bx1+((Bx2-Bx1)/2), By1 + ((By2- By1)/2)
    p = Acx - Bcx
    q = Acy - Bcy
    dist = math.sqrt((p**2) + (q**2))
    distTH = ((Ax2 - Ax1) * (Ay2 - Ay1))/3

    if dist<distTH:
        return (float)(dist/distTH)
    else:
        return 1

def ext_ovl_sim(A, B): # (ovl_score + dist_score) / 2
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

    # if (a_center_pt < [Bx2,By2] and a_center_pt > [Bx1,By1]) \
    #         or (b_center_pt < [Ax2,Ay2] and b_center_pt > [Ax1,Ay1]):
    # if(Acx < Bx2 and Acx > Bx1):
    return float((ovl_score * 1.0))

    # return 0.0
# ??????????????? ?????????_ using gt
def save_crop_bbox_img(src, bboxes, this_frame, seq):
    for bbox in bboxes:
        id, x1, y1, x2, y2 = bbox
        data = src[y1:y2, x1:x2]
        print('result/cropped/MOT20/' + str(seq) + '20_{0:04}_{1:04}'.format(this_frame, id) + '.jpg')
        try:
            cv2.imwrite('result/cropped/MOT20/' + str(seq) + '20_{0:04}_{1:04}'.format(this_frame, id) + '.jpg', data)
        except:
            continue

# ????????? ???????????? ??????.
def print_tracking_result(data, path, this_frame):
    f = open(path, 'a')
    for p_data in data:
        x1, y1, x2, y2 = p_data['box']
        p_data_width = int(x2 - x1)
        p_data_height = int(y2 - y1)

        if this_frame <= p_data['frame'] + 2:
            f.write("{0}, {1}, {2}, {3}, {4}, {5}, -1, -1, -1, -1\n".format(this_frame, p_data['id'], x1,
                                                                            y1, p_data_width,
                                                                            p_data_height))

    f.close()