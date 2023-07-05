"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2018, April).
    Finding beans in burgers: Deep semantic-visual embedding with localization.
    In Proceedings of CVPR (pp. 3984-3993)

Author: Martin Engilberge
"""

import torch
import torch.nn as nn
import torchvision.models as models


##########################################################
# translated from torch version:                         #
# https://github.com/durandtibo/weldon.resnet.pytorch    #
##########################################################


class WeldonPooling(nn.Module):  #
    # Pytorch implementation of WELDON pooling

    def __init__(self, nMax=1, nMin=None):
        super(WeldonPooling, self).__init__()
        self.nMax = nMax
        if(nMin is None):
            self.nMin = nMax
        else:
            self.nMin = nMin

        self.input = torch.Tensor()
        self.output = torch.Tensor()
        self.indicesMax = torch.Tensor()
        self.indicesMin = torch.Tensor()

    def forward(self, input):

        self.batchSize = 0
        self.numChannels = 0
        self.h = 0
        self.w = 0

        if input.dim() == 4:
            self.batchSize = input.size(0)
            self.numChannels = input.size(1)
            self.h = input.size(2)
            self.w = input.size(3)
        elif input.dim() == 3:
            self.batchSize = 1
            self.numChannels = input.size(0)
            self.h = input.size(1)
            self.w = input.size(2)
        else:
            print('error in WeldonPooling:forward - incorrect input size')

        self.input = input

        nMax = self.nMax
        if nMax <= 0:
            nMax = 0
        elif nMax < 1:
            nMax = torch.clamp(torch.floor(nMax * self.h * self.w), min=1)

        nMin = self.nMin
        if nMin <= 0:
            nMin = 0
        elif nMin < 1:
            nMin = torch.clamp(torch.floor(nMin * self.h * self.w), min=1)

        x = input.view(self.batchSize, self.numChannels, self.h * self.w)

        # sort scores by decreasing order
        scoreSorted, indices = torch.sort(x, x.dim() - 1, True)

        # compute top max
        self.indicesMax = indices[:, :, 0:nMax]
        self.output = torch.sum(scoreSorted[:, :, 0:nMax], dim=2, keepdim=True)
        self.output = self.output.div(nMax)

        # compute top min
        if nMin > 0:
            self.indicesMin = indices[
                :, :, self.h * self.w - nMin:self.h * self.w]
            yMin = torch.sum(
                scoreSorted[:, :, self.h * self.w - nMin:self.h * self.w], 2, keepdim=True).div(nMin)
            self.output = torch.add(self.output, yMin)

        if input.dim() == 4:
            self.output = self.output.view(
                self.batchSize, self.numChannels, 1, 1)
        elif input.dim() == 3:
            self.output = self.output.view(self.numChannels, 1, 1)

        return self.output

    def backward(self, grad_output, _indices_grad=None):
        nMax = self.nMax
        if nMax <= 0:
            nMax = 0
        elif nMax < 1:
            nMax = torch.clamp(torch.floor(nMax * self.h * self.w), min=1)

        nMin = self.nMin
        if nMin <= 0:
            nMin = 0
        elif nMin < 1:
            nMin = torch.clamp(torch.floor(nMin * self.h * self.w), min=1)

        yMax = grad_output.clone().view(self.batchSize, self.numChannels,
                                        1).expand(self.batchSize, self.numChannels, nMax)
        z = torch.zeros(self.batchSize, self.numChannels,
                        self.h * self.w).type_as(self.input)
        z = z.scatter_(2, self.indicesMax, yMax).div(nMax)

        if nMin > 0:
            yMin = grad_output.clone().view(self.batchSize, self.numChannels, 1).div(
                nMin).expand(self.batchSize, self.numChannels, nMin)
            self.gradInput = z.scatter_(2, self.indicesMin, yMin).view(
                self.batchSize, self.numChannels, self.h, self.w)
        else:
            self.gradInput = z.view(
                self.batchSize, self.numChannels, self.h, self.w)

        if self.input.dim() == 3:
            self.gradInput = self.gradInput.view(
                self.numChannels, self.h, self.w)

        return self.gradInput


class ResNet_weldon(nn.Module):

    def __init__(self, args, pretrained=True, weldon_pretrained_path=None):
        super(ResNet_weldon, self).__init__()

        resnet = models.resnet152(pretrained=pretrained)

        self.base_layer = nn.Sequential(*list(resnet.children())[:-2])
        self.spaConv = nn.Conv2d(2048, 2400, 1,)

        # add spatial aggregation layer
        self.wldPool = WeldonPooling(15)
        # Linear layer for imagenet classification
        self.fc = nn.Linear(2400, 1000)

        # Loading pretrained weights of resnet weldon on imagenet classification
        if pretrained:
            try:
                state_di = torch.load(
                    weldon_pretrained_path, map_location=lambda storage, loc: storage)['state_dict']
                self.load_state_dict(state_di)
            except Exception:
                print("Error when loading pretrained resnet weldon")

    def forward(self, x):
        x = self.base_layer(x)
        x = self.spaConv(x)
        x = self.wldPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
