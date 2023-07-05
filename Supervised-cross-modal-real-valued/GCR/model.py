import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import torch
import numpy as np




class GraphConvLayer(nn.Module):

    def __init__(self, input_dim=4096, output_dim=1024, dropout=0.1, negative_slope=0.2, bias = True):
        super(GraphConvLayer, self).__init__()

        self.dropout = dropout
        self.ac = nn.LeakyReLU(negative_slope, inplace=True)
        self.bias = bias

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        init.kaiming_uniform_(self.weight, a=np.math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj=None, adjflag=False):

        if self.training:
            x = F.dropout(x, self.dropout)

        view1_feature = torch.mm(x, self.weight.t())

        if adjflag:
            view1_feature = adj.mm(view1_feature)

        if self.bias is not None:
            view1_feature += self.bias

        return self.ac(view1_feature)

class CrossGCN(nn.Module):

    def __init__(self, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10, adj=None):
        super(CrossGCN, self).__init__()
        self.imgLayer = GraphConvLayer(img_input_dim, img_output_dim)
        self.textLayer = GraphConvLayer(text_input_dim, text_output_dim)

        self.shareLayer = GraphConvLayer(img_output_dim, minus_one_dim)

        self.shareClassifier = nn.Linear(minus_one_dim, output_dim)


    def forward(self, img, text, adj=None, adjflag=False):
        imgH = self.imgLayer(img, adj, adjflag)
        textH = self.textLayer(text, adj, adjflag)

        imgH2 = self.shareLayer(imgH, adj, adjflag)
        textH2 = self.shareLayer(textH, adj, adjflag)

        img_predict = self.shareClassifier(imgH2)
        text_predict = self.shareClassifier(textH2)

        return imgH2, textH2, img_predict, text_predict


class GeneratorV(nn.Module):
    def __init__(self):
        super(GeneratorV, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(4096, 2000, normalize=False),
            *block(2000, 800, normalize=False),
            nn.Linear(800, 1024),
            nn.Tanh()
        )

    def forward(self, z):

        img = self.model(z)

        return img


class DiscriminatorV(nn.Module):
    def __init__(self):
        super(DiscriminatorV, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):

        validity = self.model(img)

        return validity


class GeneratorT(nn.Module):
    def __init__(self):
        super(GeneratorT, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        # 1386 for flicker
        self.model = nn.Sequential(
            *block(300, 2000, normalize=False),
            *block(2000, 1000, normalize=False),
            *block(1000, 500, normalize=False),
            nn.Linear(500, 1024),
            nn.Tanh()
        )

    def forward(self, z):

        txt = self.model(z)

        return txt


class DiscriminatorT(nn.Module):
    def __init__(self):
        super(DiscriminatorT, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 812),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(812, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, txt):

        validity = self.model(txt)

        return validity