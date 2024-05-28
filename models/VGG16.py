from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnsn import CNSN, CrossNorm, SelfNorm
from models.simam import SimAM


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        self.cnsn = CNSN(None, SelfNorm(3))
        self.simam = SimAM()


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.SELU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        self.fc0 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
        )

        self.fc = nn.Linear(in_features=256, out_features=10, bias=True)

    def forward(self, x, return_feat=False):
        layers_output_dict = OrderedDict()
        x = x.float()

        out = self.cnsn(x)
        layers_output_dict["layer0"] = out
        out = self.conv1(out)
        
        layers_output_dict["layer1"] = out
        out=self.simam(out)
        out = self.conv2(out)
        layers_output_dict["layer2"] = out
        out=self.simam(out)
        out = self.conv3(out)
        layers_output_dict["layer3"] = out
        out=self.simam(out)
        out = self.conv4(out)
        layers_output_dict["layer4"] = out
        out=self.simam(out)
        out = self.conv5(out)
        layers_output_dict["layer5"] = out

        out = out.view(x.size(0), -1)
        out = self.fc0(out)

        if return_feat:
            return x, self.fc(out)
        else:
            return self.fc(out)
