import torch
import torch.nn.functional as F
from models import SimSiam


# trackers = torch.rand([10, 3, 224, 112]).cuda(non_blocking=True)  # RGB, 0~1
# detections = torch.rand([8, 3, 224, 112]).cuda(non_blocking=True)
#
# ass_mat = get_association_matrix(model.backbone, trackers, detections, k=min(len(trackers), 5))
# print(ass_mat.keys())

class Simsiam():
    def __init__(self):
        model = SimSiam().cuda()
        model.load_state_dict(torch.load('ckpt/simsiam2.pt'))
        model.eval()

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
                'ass_mat': ass_mat,
                'scores': top_scores,
                'indicies': top_indicies
            }

