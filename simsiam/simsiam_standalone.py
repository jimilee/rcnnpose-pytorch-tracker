import torch
import path_roll as roll
import torch.nn.functional as F
from simsiam.models import SimSiam
from simsiam.models import BarlowTwins
from simsiam.models import BYOL

# trackers = torch.rand([10, 3, 224, 112]).cuda(non_blocking=True)  # RGB, 0~1
# detections = torch.rand([8, 3, 224, 112]).cuda(non_blocking=True)
#
# ass_mat = get_association_matrix(model.backbone, trackers, detections, k=min(len(trackers), 5))
# print(ass_mat.keys())
#'E:/_workspace/rcnnpose-pytorch-tracker/simsiam/ckpt/MOT Aug( w cutout, jitter0.2)+Market C-Pose-GAN Aug (w cutout.pt'
class SimsiamSA():
    def __init__(self):
        self.model = SimSiam().cuda()
        self.model.load_state_dict(torch.load(roll.SIMSIAM_PATH))
        self.model.eval()

        # 테스트 BTwins
        # self.model = BarlowTwins('8192-8192-8192')
        # self.model.load_state_dict(torch.load('E:/_workspace/rcnnpose-pytorch-tracker/simsiam/ckpt/table3_btwins.pt'))
        # self.model.backbone  # 2048

        # 테스트 BYOL
        # self.model = BYOL(hidden_layer="avgpool")
        # self.model.load_state_dict(torch.load('E:/_workspace/rcnnpose-pytorch-tracker/simsiam/ckpt/table3_byol.pt'))
        # self.model.online_encoder  # 256

    def online_encoder(self):
        return self.model.online_encoder

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

