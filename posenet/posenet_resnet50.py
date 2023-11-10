import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50
# from googlenet import GoogLeNet

class PoseNet(nn.Module):
    def __init__(self, with_embedding=False):
        super(PoseNet, self).__init__()

        # self.backbone = GoogLeNet(with_aux=True)
        self.regressor1 = Regression('regress1')
        self.regressor2 = Regression('regress2')
        self.regressor3 = Regression('regress3', with_embedding=with_embedding)
        # self.training =  True


        resnet = resnet50(pretrained=True)
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone

        # self.backbone = resnet50(pretrained=True)


    def forward(self, x):

        x = self.backbone(x)
        pose = self.regressor3(x)

        return pose[0]

    # def loss_(self, batch):

class Regression(nn.Module):
    """Pose regression module.
    Args:
        regid: id to map the length of the last dimension of the inputfeature maps.
        with_embedding: if set True, output activations before pose regression 
                        together with regressed poses, otherwise only poses.
    Return:
        xyz: global camera position.
        wpqr: global camera orientation in quaternion.
    """
    def __init__(self, regid, with_embedding=False):
        super(Regression, self).__init__()
        conv_in = {"regress1": 512, "regress2": 528}
        self.with_embedding = with_embedding
        if regid != "regress3":
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                            nn.Conv2d(conv_in[regid], 128, kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024),
                                            nn.ReLU(),
                                            nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(1024, 3)
            self.regress_fc_wpqr = nn.Linear(1024, 4)
        else: # 'regress3'
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 2048),
                                            nn.ReLU(),
                                            nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(2048, 2)
            self.regress_fc_wpqr = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.projection(x)
        x = self.regress_fc_pose(x.view(x.size(0), 1, -1))
        xyz = self.regress_fc_xyz(x)
        # wpqr = self.regress_fc_wpqr(x)
        # wpqr = F.normalize(wpqr, p=2, dim=1)
        if self.with_embedding:
            return (xyz, None, x)
        return (xyz, None)
    

if __name__ == '__main__':
    net = PoseNet().cuda()
    from thop import profile
    pseudo_input = torch.randn(32,3,224,224).cuda()

    flops, params = profile(net, inputs =(pseudo_input,))

    print(flops)
    print(params)