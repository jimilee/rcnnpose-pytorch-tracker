import sys
import time
sys.path.append('../')

import cv2
import numpy as np
from rcnnpose.estimator import BodyPoseEstimator
from rcnnpose.utils import draw_body_connections, draw_keypoints, draw_masks, _draw_box, draw_boxes, draw_tracker_boxes
from examples.tracker_demo import simple_tracker

estimator = BodyPoseEstimator(pretrained=True)
videoclip = cv2.VideoCapture('media/mot16-11.wmv')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#파일 stream 생성
# out = cv2.VideoWriter('result/test_Result_MOT.avi',fourcc, videoclip.get(cv2.CAP_PROP_FPS), (int(videoclip.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoclip.get(cv2.CAP_PROP_FRAME_HEIGHT))) )
total_proctime = 0.0
st = simple_tracker()

frame_cnt = 0
while videoclip.isOpened():
    flag, frame = videoclip.read()
    if not flag:
        break
    starttime = time.time()
    pred_dict = estimator(frame, masks=False, keypoints=True)
    # print(pred_dict['estimator_k'])
    boxes = estimator.get_boxes(pred_dict['estimator_k'], score_threshold=0.5)
    # print(boxes)
    # masks = estimator.get_masks(pred_dict['estimator_m'], score_threshold=0.80)
    # keypoints = estimator.get_keypoints(pred_dict['estimator_k'], score_threshold=0.8)
    # for i in (pred_dict['estimator_k']['labels'] > 0).nonzero().view(-1):
    #     print(pred_dict['estimator_k']['keypoints'][i].detach().cpu().squeeze().numpy())
    #     print(pred_dict['estimator_k']['keypoints_scores'][i].detach().cpu().squeeze().numpy())

    frame_dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_dst = cv2.merge([frame_dst] * 3)
    # overlay_m = draw_masks(frame_dst, masks, color=(0, 255, 0), alpha=0.5)
    # overlay_k = draw_body_connections(frame, keypoints, thickness=1, alpha=0.7)
    overlay_d = draw_boxes(frame_dst, boxes)

    starttime_trk = time.time()
    target = st.tracking(boxes, frame, frame_cnt)
    endtime_trk = time.time()
    overlay_tk = draw_tracker_boxes(frame, target, frame_cnt)

    # frame_dst = np.hstack((frame, overlay_m, overlay_k))
    frame_dst = np.hstack((frame, overlay_d, overlay_tk))
    total_proctime = total_proctime+(time.time() - starttime)

    # try:
    #     cv2.putText(frame_dst, str(1/(endtime_trk - starttime)), (500, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    #     cv2.putText(frame_dst, str(1/(time.time() - starttime_trk)), (500, 380), cv2.FONT_HERSHEY_PLAIN, 1, (100, 0, 0), 2)
    # except:
    #     pass
    cv2.imwrite('result/img/'+'{0:04}'.format(frame_cnt) + '.jpg',frame_dst)
    # out.write(overlay_tk)
    cv2.imshow('Video Demo', overlay_tk)
    frame_cnt+=1
    if cv2.waitKey(20) & 0xff == 27: # exit if pressed `ESC`
        break

# out.release()
print('total time : '+str(total_proctime/300))
# videoclip.release()
# cv2.destroyAllWindows()
