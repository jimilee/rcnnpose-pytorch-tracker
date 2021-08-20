import torch
import torch.nn as nn
import torchvision

class BarlowTwins(nn.Module):
    def __init__(self, projector):
        super().__init__()
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
#
# if __name__ == '__main__':
#     btwins = BarlowTwins('8192-8192-8192')
#     btwins.load_state_dict(torch.load('E:/_workspace/rcnnpose-pytorch-tracker/simsiam/ckpt/table3_btwins.pt'))
#     btwins.backbone # 2048