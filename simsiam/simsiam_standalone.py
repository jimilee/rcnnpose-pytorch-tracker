import torch
import torch.nn.functional as F
from simsiam.models import SimSiam

# trackers = torch.rand([10, 3, 224, 112]).cuda(non_blocking=True)  # RGB, 0~1
# detections = torch.rand([8, 3, 224, 112]).cuda(non_blocking=True)
#
# ass_mat = get_association_matrix(model.backbone, trackers, detections, k=min(len(trackers), 5))
# print(ass_mat.keys())

class SimsiamSA():
    def __init__(self):
        self.model = SimSiam().cuda()
        self.model.load_state_dict(torch.load('E:/_workspace/rcnnpose-pytorch-tracker/simsiam/ckpt/simsiam-new.pt'))
        self.model.eval()

    def backbone(self):
        return self.model.backbone

    def get_association_matrix(self, net, trackers, detections, k, t=0.1):
        with torch.no_grad():
            z_trackers = net(trackers)
            z_trackers = F.normalize(z_trackers, dim=1)

            z_detections = net(detections)
            z_detections = F.normalize(z_detections, dim=1)

            ass_mat = torch.mm(z_detections, z_trackers.t())

            top_scores, top_indicies = ass_mat.topk(k=k, dim=-1)

            top_scores = (top_scores / t).exp()

            return {
                'ass_mat': ass_mat, # ass_mat : 디텍션x트래커 매트릭스
                'scores': top_scores, # scores : 디텍션xTopK트래커 스코어
                'indicies': top_indicies # indicies : 디텍션xTopK트래커 인덱스
            }

