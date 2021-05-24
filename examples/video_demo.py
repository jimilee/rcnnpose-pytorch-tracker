import sys
import time
sys.path.append('../')

import cv2
import numpy as np
from rcnnpose.estimator import BodyPoseEstimator
from rcnnpose.utils import draw_body_connections, draw_keypoints, draw_masks, _draw_box, draw_boxes

estimator = BodyPoseEstimator(pretrained=True)
videoclip = cv2.VideoCapture('media/mot16-11.wmv')
total_proctime = 0.0
while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    starttime = time.time()
    pred_dict = estimator(frame, masks=False, keypoints=True)
    # print(pred_dict['estimator_k'])
    boxes = estimator.get_boxes(pred_dict['estimator_k'], score_threshold=0.5)
    # print(pred_dict['estimator_k']['boxes'])
    # masks = estimator.get_masks(pred_dict['estimator_m'], score_threshold=0.80)
    # keypoints = estimator.get_keypoints(pred_dict['estimator_k'], score_threshold=0.8)
    # for i in (pred_dict['estimator_k']['labels'] > 0).nonzero().view(-1):
    #     print(pred_dict['estimator_k']['keypoints'][i].detach().cpu().squeeze().numpy())
    #     print(pred_dict['estimator_k']['keypoints_scores'][i].detach().cpu().squeeze().numpy())

    frame_dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_dst = cv2.merge([frame_dst] * 3)
    # overlay_m = draw_masks(frame_dst, masks, color=(0, 255, 0), alpha=0.5)
    # overlay_k = draw_body_connections(frame, keypoints, thickness=1, alpha=0.7)
    overlay_k = draw_boxes(frame_dst, boxes)



    # frame_dst = np.hstack((frame, overlay_m, overlay_k))
    frame_dst = np.hstack((frame, overlay_k))
    total_proctime = total_proctime+(time.time() - starttime)
    cv2.putText(frame_dst, str((time.time() - starttime)), (600,400), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)


    cv2.imshow('Video Demo', frame_dst)
    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break

print('total time : '+str(total_proctime/300))
# videoclip.release()
# cv2.destroyAllWindows()
