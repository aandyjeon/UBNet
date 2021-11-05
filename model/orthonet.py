import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Dict, Any, cast
from model.module import CodeReduction

class OrthoNet(nn.Module):
    def __init__(self, num_classes: int = 2) :
        super(OrthoNet, self).__init__()
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.test = nn.Conv2d(64,64,kernel_size = 1)

        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )
        self.trans5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size = 1),
            nn.LeakyReLU(0.1)
        )

        self.reduction = CodeReduction(c_dim = 64*5,  feat_hw = 7, blocks = 5) 

        self.fc1 = nn.Linear(64, self.num_classes)
        self.fc2 = nn.Linear(64, self.num_classes)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.fc4 = nn.Linear(64, self.num_classes)
        self.fc5 = nn.Linear(64, self.num_classes)

        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x: Dict[str, torch.Tensor])->torch.Tensor:

        x1, x2, x3, x4, x5 = x['out1'], x['out2'], x['out3'], x['out4'], x['out5']

        out1 = self.avgpool(x1) 
        out1 = self.trans1(out1) 

        out2 = self.avgpool(x2)
        out2 = self.trans2(out2)

        out3 = self.avgpool(x3)
        out3 = self.trans3(out3)

        out4 = self.avgpool(x4)
        out4 = self.trans4(out4)

        out5 = self.avgpool(x5)
        out5 = self.trans5(out5)
        
        out_concat = torch.cat((out1, out2, out3, out4, out5), axis = 1) 
        out, loss_conv, loss_trans = self.reduction(out_concat)

        out1_, out2_, out3_, out4_, out5_= torch.split(out, [out.shape[1]//5]*5, dim = 1)

        out1_ = self.fc1(out1_)
        out2_ = self.fc2(out2_)
        out3_ = self.fc3(out3_)
        out4_ = self.fc4(out4_)
        out5_ = self.fc5(out5_)

        out = (out1_ + out2_ + out3_ + out4_ + out5_)/5
        out = self.softmax(out)


        return out, loss_conv, loss_trans
