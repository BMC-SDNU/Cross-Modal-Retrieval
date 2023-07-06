import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImgNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(ImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.decode = nn.Linear(code_len, txt_feat_len)
        self.alpha = 1.0

    def forward(self, x):
        # xs = []
        for layer in self.alexnet.features:
            x = layer(x)
        # x = self.alexnet.features(x)

        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        code_s = torch.sign(code)
        decoded = self.decode(code_s)
        return (x, feat), hid, code, decoded

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len, image_size=4096):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        # self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(4096, code_len)
        self.decode = nn.Linear(code_len, image_size)
        self.alpha = 1.0

    def forward(self, x):
        feat1 = self.fc1(x)
        feat = F.relu(feat1)
        hid = self.fc2(feat)

        code = torch.tanh(self.alpha * hid)
        code_s = torch.sign(code)
        decoded = self.decode(code_s)
        return feat, hid, code, decoded

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
